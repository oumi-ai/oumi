import timm
import torch
from torchvision import transforms
from transformers import DPTForDepthEstimation, DPTImageProcessor

from .base_encoder import BaseVisionTower


class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378, image_mean=[0.48145466, 0.4578275, 0.40821073]):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        self.to_tensor = transforms.Compose([
            # Add any additional transformations here (e.g., Resize, Normalize)
            transforms.ToTensor(),  # This converts the PIL Image to a PyTorch Tensor
        ])
        # print(transform)
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = self._transforms(images=image, return_tensors="pt")

        # Convert the NumPy array to a PyTorch tensor
        # output['pixel_values'] = [torch.from_numpy(output['pixel_values'][0])]

        return output


class MiDaSVisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(MiDaSVisionTower, self).__init__(vision_tower, args, delay_load)

        # extract midas model info here
        self.vision_model = "midas"
        self._model_name = ""

        if self.vision_tower_name.lower() == "hybrid-midas":
            # TODO: why does this immediately NaN???
            self._model_name = "Intel/dpt-hybrid-midas"
            self._hidden_size = 768
            self._image_size = 384
            self._patch_size = 16
        elif self.vision_tower_name.lower() == "large-midas":
            self._model_name = "Intel/dpt-large"
            self._hidden_size = 1024
            self._image_size = 384
            self._patch_size = 16
        elif self.vision_tower_name.lower() == "large-beit-midas-512":
            self._model_name = "Intel/dpt-beit-large-512"
            self._hidden_size = 1024
            self._image_size = 512
            self._patch_size = 16
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        if not self.delay_load:
            # print("Use this?")
            self.load_model()

    def load_model(self, device_map=None):
        transforms = DPTImageProcessor.from_pretrained(self._model_name)
        self.vision_tower = DPTForDepthEstimation.from_pretrained(self._model_name)
        """ValueError: DPTForDepthEstimation does not support `device_map='auto'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."""
        self.vision_tower._no_split_modules = ["DPTViTLayer"]

        # print(self.vision_tower)
        self.vision_tower.output_tokens = True

        # get model specific transforms (normalization, resize)
        # data_config = timm.data.resolve_model_data_config(self.vision_tower)

        self.image_processor = ProcessorWrapper(transforms, height=self._image_size, width=self._image_size, image_mean=[0.5, 0.5, 0.5])

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
            forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True
            )
            image_features = forward_outs["hidden_states"][-1]
            return self._feature_select(image_features)

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2
