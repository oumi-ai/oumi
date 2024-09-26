import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

# from ezcolorlog import root_logger as logger
from oumi.utils.logging import logger

from .base_encoder import BaseVisionTower
from .load import load_vision_model


def revert_preprocessing(images):
    def convert_to_pil(tensor):
        return to_pil_image(tensor)

    reversed_images = []
    for image in images:
        reversed_image = convert_to_pil(image)
        reversed_images.append(reversed_image)

    return reversed_images


class HybridVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        model_names = self.vision_tower_name.replace("hybridmodel-", "")
        self.model_names = model_names.split("-&&&-")
        logger.warning(
            f"Creating a Hybrid Vision Tower with models: {self.model_names}"
        )

        if not self.delay_load:
            self.load_model()

    def load_model(self):
        self.vision_model = "hybrid"
        for i, model_name in enumerate(self.model_names, start=1):
            setattr(
                self, f"vision_tower_{i}", load_vision_model(model_name, args=self.args)
            )

        self._hidden_size = sum(
            [
                getattr(self, f"vision_tower_{i}").hidden_size
                for i in range(1, len(self.model_names) + 1)
            ]
        )
        self._image_size = 384
        self._patch_size = 16

        self.image_processor = []
        for i in range(1, len(self.model_names) + 1):
            vision_tower = getattr(self, f"vision_tower_{i}")
            self.image_processor.append(vision_tower.image_processor)

        for i in range(1, len(self.model_names) + 1):
            getattr(self, f"vision_tower_{i}").requires_grad_(
                self.unfreeze_mm_vision_tower
            )

        self.is_loaded = True

    def forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            output_images_features = []
            for i in range(1, len(self.model_names) + 1):
                vision_tower = getattr(self, f"vision_tower_{i}")
                processed_images = [image[i - 1] for image in images]
                if len(processed_images) > 1:
                    batch_tensor = torch.stack(processed_images)
                else:
                    batch_tensor = processed_images[0]

                # print("batch tensor", batch_tensor.shape)

                image_features = vision_tower._forward(
                    batch_tensor.to(device=self.device, dtype=self.dtype)
                )
                # print(image_features.shape)
                b, num_tokens, dim = image_features.shape
                if num_tokens != self.image_token_len:
                    target_h = target_w = int(np.sqrt(self.image_token_len))
                    h = w = int(np.sqrt(num_tokens))
                    image_features = image_features.view(b, h, w, dim)
                    image_features = image_features.permute(0, 3, 1, 2).contiguous()
                    image_features = F.interpolate(
                        image_features.to(torch.float32),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).to(image_features.dtype)
                    image_features = (
                        image_features.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                    )

                output_images_features.append(image_features)

            output_tensor = torch.cat(output_images_features, dim=-1)
            # print("output", output_tensor.shape)

            return output_tensor

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
        if hasattr(self.vision_tower_1, "dtype"):
            return self.vision_tower_1.dtype
        else:
            params = list(self.vision_tower_1.parameters())
            return (
                params[0].dtype if len(params) > 0 else torch.float32
            )  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, "device"):
            return self.vision_tower_1.device
        else:
            params = list(self.vision_tower_1.parameters())
            return (
                params[0].device if len(params) > 0 else torch.device("cpu")
            )  # Default to CPU if no parameters
