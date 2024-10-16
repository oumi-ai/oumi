from abc import abstractmethod

import torch
import torch.nn as nn

from oumi.utils.logging import logger


class ProcessorWrapper:
    def __init__(
        self,
        transform,
        height=378,
        width=378,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
    ):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        # print(transform)
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors="pt"):
        # Ensure image is a PIL Image
        output = {}
        output["pixel_values"] = [self._transforms(image)]
        return output


class BaseVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super().__init__()

        logger.info(f"Initializing Cambrian MM Vision Tower: {type(self)}...")

        self.is_loaded = False
        self.args = args

        self.vision_tower_name = vision_tower_name
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.unfreeze_mm_vision_tower = getattr(args, "unfreeze_mm_vision_tower", False)
        logger.warning(f"Unfreezing MM Vision Tower: {self.unfreeze_mm_vision_tower}")
        self.delay_load = delay_load

    @abstractmethod
    def load_model(self, device_map=None):
        raise NotImplementedError("Subclasses must implement load_model")

    @abstractmethod
    def _forward(self, images):
        raise NotImplementedError("Subclasses must implement forward")

    def forward(self, images):
        if type(images) is list:
            image_features = [self._forward(image.unsqueeze(0)) for image in images]
        else:
            image_features = self._forward(images)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "dtype"):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].dtype if len(params) > 0 else torch.float32
            )  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "device"):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].device if len(params) > 0 else torch.device("cpu")
            )  # Default to CPU if no parameters

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size

    @property
    def image_size(self):  # resolution
        # return self.config.image_size
        try:
            return self.config.image_size
        except:
            return self._image_size

    @property
    def patch_size(self):
        # return self.config.patch_size
        try:
            return self.config.patch_size
        except:
            return self._patch_size

    @property
    def num_patches_per_side(self):
        if self._interp_size is not None:
            return int(self._interp_size**0.5)
        try:
            return self.image_size // self.patch_size
        except:
            return self._num_patches_per_side

    @property
    def num_patches(self):
        if self._interp_size is not None:
            return self._interp_size
        try:
            return self.num_patches_per_side**2
        except:
            return self._num_patches
