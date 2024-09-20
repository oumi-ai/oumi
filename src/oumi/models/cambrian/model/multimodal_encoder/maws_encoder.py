import torch
from torchvision import transforms

from .base_encoder import BaseVisionTower, ProcessorWrapper


class MawsVisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(MawsVisionTower, self).__init__(vision_tower, args, delay_load)

        # extract image resolution from model name
        if self.vision_tower_name.startswith("maws"):
            self._image_size = 224

        if not self.delay_load:
            self.load_model()

    def load_model(self):
        self.vision_model = "maws"

        # ps = 14
        # hs = 1280
        # res = 224
        if self.vision_tower_name.lower() == "maws/mae-h-14":
            self.vision_tower = torch.hub.load("facebookresearch/maws", model="vit_h14_mae")

        elif self.vision_tower_name.lower() == "maws/maws-h-14":
            self.vision_tower = torch.hub.load("facebookresearch/maws", model="vit_h14_maws")

        # ps = 16
        # hs = 1024
        # res = 224
        elif self.vision_tower_name.lower() == "maws/maws-l-16":
            self.vision_tower = torch.hub.load("facebookresearch/maws", model="vit_l16_maws")

        elif self.vision_tower_name.lower() == "maws/mae-l-16":
            self.vision_tower = torch.hub.load("facebookresearch/maws", model="vit_l16_mae")

        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        # print(self._image_size, self._patch_size)
        preprocess = transforms.Compose([
            transforms.Resize(256),            # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),        # Crop the center 224x224 pixels
            transforms.ToTensor(),             # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor
                                 std=[0.229, 0.224, 0.225])
        ])

        self.image_processor = ProcessorWrapper(preprocess, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _feature_select(self, image_features):
        if self.select_feature == 'patch':  # default
            features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return features

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            # image_features = image_features[:, 1:, :]
            image_features = self._feature_select(image_features)
            # print(image_features.shape)
            return image_features

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2
