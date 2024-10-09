# Used in the following published models:
# nyu-visionx/cambrian-phi3-3b
# nyu-visionx/cambrian-8b
# nyu-visionx/cambrian-13b
# nyu-visionx/cambrian-34b

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from oumi.models.experimental.cambrian.utils import IS_XLA_AVAILABLE
from oumi.utils.logging import logger

from .base_encoder import BaseVisionTower


def extract_interp(model_name):
    interp = None
    base_model_name = model_name

    if "interp" in model_name:
        base_model_name = model_name.split("-interp")[0]

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, interp


class ClipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super().__init__(vision_tower_name, args, delay_load)
        base_model_name, interp = extract_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.debug(
                f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping."
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            from torch_xla.utils.checkpoint import checkpoint

            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = (
                checkpoint
            )

    def _feature_select(self, image_features):
        if self.select_feature == "patch":
            features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return self._feature_select(image_features)

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
        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint

            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = (
                checkpoint
            )

        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            interp_features = self.interpolate(image_features)
            return interp_features
