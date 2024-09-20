import numpy as np
import torch
import torch.nn.functional as F
from ezcolorlog import root_logger as logger

from .base_encoder import BaseVisionTower
from .sam.encoder import SAM_MODEL_CONFIG, create_sam_vit
from .sam.transforms import ResizeLongestSide


class SAMProcessor:
    def __init__(self, height=378, width=378):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = ResizeLongestSide(height)

        self.image_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors="pt"):
        input_image = self._transforms.apply_image(np.array(image))
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]
        # Normalize colors
        image_mean = torch.Tensor(self.image_mean).view(1, -1, 1, 1)
        pixel_std = torch.Tensor(self.pixel_std).view(1, -1, 1, 1)
        input_image_torch = (input_image_torch - image_mean) / pixel_std

        # Pad
        h, w = input_image_torch.shape[-2:]
        padh = self._crop_size["height"] - h
        padw = self._crop_size["width"] - w
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
        output = {}
        output["pixel_values"] = [input_image_torch[0]]
        return output


def extract_res_interp(model_name):
    # ty claude
    valid_model_names = ["sam_vit_b", "sam_b_downsample", "sam_vit_l", "sam_vit_h"]

    model_parts = model_name.split("-")
    base_model_name = model_parts[0]

    if base_model_name not in valid_model_names:
        raise ValueError(
            f"Invalid model name: {base_model_name}. Valid model names are: {', '.join(valid_model_names)}"
        )

    res = 1024
    interp = 1024

    for part in model_parts[1:]:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class SAMVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        """try to extract an image res + interp res from the model name

        valid model names:
            sam_vit_b
            sam_b_downsample
            sam_vit_l
            sam_vit_h

        res pattern: <model_name>-res<res>-interp<interp>

        eg: sam_vit_b-res512-intp336
        """

        # extract image resolution from model name
        if not "sam" in self.vision_tower_name.lower():
            raise ValueError(f"Unknown vision tower: {self.vision_tower_name}")

        base_model_name, res, interp = extract_res_interp(self.vision_tower_name)
        self.vision_tower_name = base_model_name
        self._image_size = res
        self._num_patches = interp
        self._num_patches_per_side = int(self.num_patches**0.5)

        self._hidden_size = SAM_MODEL_CONFIG[self.vision_tower_name]["width"]
        self.hidden_size_single = SAM_MODEL_CONFIG[self.vision_tower_name]["width"]

        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.debug(
                f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping."
            )
            return

        self.image_processor = SAMProcessor(
            height=self._image_size, width=self._image_size
        )
        self.vision_tower = create_sam_vit(self.vision_tower_name)
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)

        self.is_loaded = True

    def interpolate(self, image_features):
        target_h = target_w = self._num_patches_per_side
        image_features_flatten = F.interpolate(
            image_features.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=image_features.dtype)
        image_features_flatten = image_features_flatten.flatten(2, 3).permute(0, 2, 1)
        return image_features_flatten.contiguous()

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            with torch.cpu.amp.autocast(cache_enabled=True, dtype=torch.bfloat16):
                x = self.vision_tower.patch_embed(
                    images.to(device=self.device, dtype=self.dtype)
                )
                if self.vision_tower.pos_embed is not None:
                    # assert x.shape == self.vision_tower.pos_embed.shape, f"Shape mismatch: {x.shape} != {self.vision_tower.pos_embed.shape}"
                    x = x + self.vision_tower.pos_embed

                for blk in self.vision_tower.blocks:
                    x = blk(x)

                image_features = x.permute(0, 3, 1, 2).contiguous()
                return self.interpolate(image_features)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def image_size(self):  # resolution
        return self._image_size

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_patches_per_side(self):
        return self._num_patches_per_side

    @property
    def num_patches(self):
        return self._num_patches
