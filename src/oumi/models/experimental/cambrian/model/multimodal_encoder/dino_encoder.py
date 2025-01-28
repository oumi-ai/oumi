# Used in the following published models:
# nyu-visionx/cambrian-phi3-3b
# nyu-visionx/cambrian-8b
# nyu-visionx/cambrian-13b
# nyu-visionx/cambrian-34b

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model

from oumi.utils.logging import logger

from .base_encoder import BaseVisionTower


def extract_res_interp(model_name):
    valid_model_prefixes = [
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/dinov2-giant-imagenet1k-1-layer",
        "facebook/dinov2-giant",
    ]

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = prefix
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    res = None
    interp = None

    parts = model_name[len(base_model_name) :].split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        """try to extract an image resolution and interpolation res from the model name string

        valid model names:
            facebook/dinov2-small
            facebook/dinov2-base
            facebook/dinov2-large
            facebook/dinov2-giant
            facebook/dinov2-giant-imagenet1k-1-layer

        res pattern: <model_name>-res<res>-interp<interp>

        eg: facebook/dinov2-small-res518-interp224
        """

        # extract image resolution from model name
        # if self.vision_tower_name.startswith("facebook/dinov2-"):
        #     parts = self.vision_tower_name.split("-res")
        #     if len(parts) == 2:
        #         self._image_size = int(parts[1])
        #         logger.warning(f"Extracted image size {self._image_size} from passed model name '{self.vision_tower_name}' (using {parts[0]})")
        #         self.vision_tower_name = parts[0]
        #     else:
        #         self._image_size = None  # default is 518px
        base_model_name, res, interp = extract_res_interp(self.vision_tower_name)
        self._vision_tower_name = vision_tower
        self.vision_tower_name = base_model_name
        self._image_size = res
        self._interp_size = interp
        self._patch_size = 14  # default patch size

        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)
        """ValueError: Dinov2Model does not support `device_map='auto'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."""
        self.vision_tower._no_split_modules = ["Dinov2SwiGLUFFN"]

        _image_size = self.vision_tower.config.image_size
        if self._image_size is None:
            self._image_size = _image_size
        else:
            logger.warning(
                f"Overriding DinoVisionTower image size of {_image_size} with {self._image_size}"
            )

        # increase shortest edge to prevent edge case crops
        default_shortest_ratio = 8 / 7  # 224/256
        # shortest_edge = int(default_shortest_ratio * self._image_size)
        shortest_edge = self._image_size

        processor = AutoImageProcessor.from_pretrained(
            self.vision_tower_name,
            crop_size=dict(height=self._image_size, width=self._image_size),
            size=dict(shortest_edge=shortest_edge),
        )

        logger.info(f"Dino Vision Processor: {processor}")
        self.image_processor = processor

        # Assign the output channels of the projection convolution as the hidden size
        self._hidden_size = (
            self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        )
        # Assign the first value of the stride of the projection convolution as the patch size
        self._patch_size = (
            self.vision_tower.embeddings.patch_embeddings.projection.stride[0]
        )

        # print(self._hidden_size, self._patch_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs[
            "last_hidden_state"
        ]  # batch_size, sequence_length, hidden_size

        if self.select_feature == "cls_patch":
            image_features = sequence_output
        elif self.select_feature == "patch":
            image_features = sequence_output[:, 1:]
        elif self.select_feature == "cls":
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images):
        # logger.warning(f"images shape: {images.shape}")
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype)
            )
            # logger.warning(f"image_forward_outs shape: {image_forward_outs['last_hidden_state'].shape}")
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # logger.warning(f"image_features shape: {image_features.shape}")
            interp_features = self.interpolate(image_features)
            # logger.warning(f"interp_features shape: {interp_features.shape}")
            return interp_features

    @property
    def num_patches_per_side(self):
        return int(self.num_patches**0.5)

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self._image_size // self._patch_size) ** 2
        else:
            return self._interp_size
