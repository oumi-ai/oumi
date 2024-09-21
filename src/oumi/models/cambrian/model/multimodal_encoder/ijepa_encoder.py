import os

import torch
from torchvision import transforms

# from ezcolorlog import root_logger as logger
from oumi.utils.logging import logger

from .base_encoder import BaseVisionTower, ProcessorWrapper
from .ijepa.vision_transformer import vit_giant, vit_huge


class IJepaVisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(IJepaVisionTower, self).__init__(vision_tower, args, delay_load)

        # extract image resolution from model name
        if self.vision_tower_name.startswith("ijepa"):
            self._image_size = 224

        if not self.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        self.vision_model = "ijepa"

        if self.vision_tower_name.lower() == "ijepa-vit-h-14":
            self.vision_tower = vit_huge()
            local_path = "/mnt/disks/storage/vision_ckpts/ijepa/IN22K-vit.h.14-900e.pth.tar"
            url = "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar"
        elif self.vision_tower_name.lower() == "ijepa-vit-g-16":
            self.vision_tower = vit_giant()
            local_path = "/mnt/disks/storage/vision_ckpts/ijepa/IN22K-vit.g.16-600e.pth.tar"
            url = "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar"
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        if os.path.exists(local_path):
            logger.info(f"Loading `{self.vision_tower_name}` from local path: {local_path}")
            ckpt = torch.load(local_path, map_location=torch.device('cpu'))
        else:
            logger.info(f"Downloading `{self.vision_tower_name}` from url: {url}")
            ckpt = torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu'))
        pretrained_dict = ckpt['encoder']

        # Load the pre-trained weights into the vision tower model
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}

        self.vision_tower.load_state_dict(pretrained_dict)
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size
        self._patch_size = self.vision_tower.patch_embed.patch_size

        preprocess = transforms.Compose([
            transforms.Resize(256),            # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),        # Crop the center 224x224 pixels
            transforms.ToTensor(),             # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor
                                 std=[0.229, 0.224, 0.225])
        ])

        self.image_processor = ProcessorWrapper(preprocess, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        logger.info(f"Loaded IJepa model: {self.vision_tower_name} with hidden size: {self._hidden_size}, image size: {self._image_size}, patch size: {self._patch_size}\nSetting requires_grad: {self.unfreeze_mm_vision_tower}")
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
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
