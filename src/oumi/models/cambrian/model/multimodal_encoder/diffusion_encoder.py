from typing import Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from torchvision import transforms

from .base_encoder import BaseVisionTower, ProcessorWrapper


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`):
                (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`):
                (batch, sequence_length, feature_dim) encoder hidden states
        """
        # By default samples have to be AT least a multiple of the overall upsampling
        # factor.The overall upsampling factor is equal to 2 ** (#upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any i
        # upsampling size on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of
        # `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # project
        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, None)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            _has_attr = hasattr(downsample_block, "has_cross_attention")
            if _has_attr and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            _has_attr = hasattr(upsample_block, "has_cross_attention")
            if _has_attr and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                    upsample_size=upsample_size,
                    attention_mask=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            if i in up_ft_indices:
                up_ft[i] = sample

        output = {}
        output["up_ft"] = up_ft
        return output


class OneStepSDPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        device = self._execution_device

        scale_factor = self.vae.config.scaling_factor
        latents = scale_factor * self.vae.encode(img_tensor).latent_dist.mode()

        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy, t, up_ft_indices, encoder_hidden_states=prompt_embeds
        )
        return unet_output


class DiffusionVisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(DiffusionVisionTower, self).__init__(vision_tower, args, delay_load)

        if not self.delay_load:
            self.load_model()

    def extract_features(self, images, time_step=250, output="dense", layers=[1, 2, 3]):

        batch_size = images.shape[0]

        # Repeat the empty prompt embeddings to match the batch size
        prompt_embeds = (self.empty_prompt_embeds.repeat(batch_size, 1, 1)).to(device=self.device, dtype=self.dtype)

        # Pass the images and prompts through the model
        with torch.no_grad():
            # what was happening in the pipeline
            scale_factor = self.vae.config.scaling_factor
            latents = scale_factor * self.vae.encode(images).latent_dist.mode()

            t = torch.tensor(time_step, dtype=torch.long, device=self.device)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t).to(dtype=self.dtype, device=self.device)

            #print("image dtype, latent dtype, noise dtype, t dtype, latents_noisy dtype:", images.dtype, latents.dtype, noise.dtype, t.dtype, latents_noisy.dtype)

            unet_output = self.unet(
                latents_noisy, t, self.up_ft_index, encoder_hidden_states=prompt_embeds.detach()
            )

        unet_output = unet_output["up_ft"]
        #print(len(unet_output))

        # Process the extracted features
        features = []
        for idx in unet_output:
            layer_output = unet_output[idx]
            if output == "gap":
                features.append(layer_output.mean(dim=(2, 3)))
            elif output == "dense":
                h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
                features.append(torch.nn.functional.interpolate(layer_output, size=(h, w), mode="bilinear", align_corners=False))

        # Concatenate features from different layers along the channel dimension
        concatenated_features = torch.cat(features, dim=1)

        # Reshape the features to (batch_size, number_of_tokens, token_dimension)
        batch_size = concatenated_features.shape[0]
        number_of_tokens = concatenated_features.shape[2] * concatenated_features.shape[3]
        token_dimension = concatenated_features.shape[1]
        reshaped_features = concatenated_features.permute(0, 2, 3, 1).reshape(batch_size, number_of_tokens, token_dimension)

        return reshaped_features

    def load_model(self, device_map=None):
        self.vision_model = "diffusion"

        # self.vision_tower = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        sd_id = "stabilityai/stable-diffusion-2-1"

        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(
            sd_id, unet=unet, safety_checker=None
        )
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )

        self.onestep_pipe = onestep_pipe

        self.vae = onestep_pipe.vae
        self.unet = onestep_pipe.unet
        self.scheduler = onestep_pipe.scheduler

        self.up_ft_index = [0, 1, 2, 3]
        self.onestep_pipe.output_tokens = True

        # Encode the empty prompt once and reuse it in extract_features
        with torch.no_grad():
            self.empty_prompt_embeds = self.onestep_pipe.encode_prompt(
                [""],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            self.empty_prompt_embeds = self.empty_prompt_embeds[0]
        # print("text embeds", self.empty_prompt_embeds.shape)

        self._hidden_size = 3520
        self._image_size = 512
        self._patch_size = 16
        # print(self._image_size, self._patch_size)
        preprocess = transforms.Compose([
            transforms.Resize(512),                 # Resize the shorter side to 512 pixels
            transforms.CenterCrop(512),             # Crop the center to make it 512x512
            transforms.ToTensor(),                  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalize the tensor
                                 std=[0.5, 0.5, 0.5])
        ])

        self.image_processor = ProcessorWrapper(preprocess, height=self._image_size, width=self._image_size)

        # freeze or unfreeze the unet
        self.unet.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        # print(self.device, self.dtype)
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.extract_features(images.to(device=self.device, dtype=self.dtype)).to(images.dtype)
            # print("Image output shape, device and dtype:", image_features.shape, image_features.device, image_features.dtype)
            # image_features = image_features.to(device=self.device, dtype=self.dtype)
            return image_features

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vae, 'dtype'):
            return self.vae.dtype
        else:
            params = list(self.vae.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vae, 'device'):
            return self.vae.device
        else:
            params = list(self.vae.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu") # Default to CPU if no parameters
