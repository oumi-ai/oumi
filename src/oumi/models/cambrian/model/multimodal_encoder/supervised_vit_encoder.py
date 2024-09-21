import timm
import torch

# from ezcolorlog import root_logger as logger
from oumi.utils.logging import logger

from .base_encoder import BaseVisionTower, ProcessorWrapper


class SupervisedViT_VisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(SupervisedViT_VisionTower, self).__init__(vision_tower, args, delay_load)

        # extract image resolution from model name
        if self.vision_tower_name.startswith("mae"):
            self._image_size = 224

        if not self.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        self.vision_model = "supervised"

        if self.vision_tower_name.lower() == "supervised-vit-h-14-in21k":
            self.vision_tower = timm.create_model('vit_huge_patch14_224.orig_in21k', pretrained=True)
            # ps = 14
            # hs = 1280
            # res = 224
        elif self.vision_tower_name.lower()=="supervised-vit-l-16-in21k":
            self.vision_tower = timm.create_model('vit_large_patch16_224.orig_in21k', pretrained=True)
            # ps = 16
            # hs = 1024
            # res = 224
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]

        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.vision_tower)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        self.image_processor = ProcessorWrapper(transforms, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def feature_select(self, feats):
        if self.select_feature == 'cls_patch':
            image_features = feats
        elif self.select_feature == 'patch':
            image_features = feats[:, 1:]
        elif self.select_feature == 'cls':
            image_features = feats[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_feature_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_feature_outs).to(images.dtype)
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
