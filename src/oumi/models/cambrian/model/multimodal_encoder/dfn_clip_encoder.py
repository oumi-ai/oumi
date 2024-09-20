import torch
from open_clip import create_model_from_pretrained

from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower


class DfnClipVisionTower(ClipVisionTower):
    def load_model(self, device_map=None):
        if self.vision_tower_name == "apple/DFN5B-CLIP-ViT-H-14-378":
            clip_model, processor = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
            # self._hidden_size = 1280
            # self.image_processor = ProcessorWrapper(processor)
        elif self.vision_tower_name == "apple/DFN2B-CLIP-ViT-L-14":
            clip_model, processor = create_model_from_pretrained('hf-hub:apple/DFN2B-CLIP-ViT-L-14')
            # self._hidden_size = 1024
            # self.image_processor = ProcessorWrapper(processor, height=224, width=224)

        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        self.vision_tower = clip_model.visual
        self.vision_tower.output_tokens = True

        self._hidden_size = clip_model.visual.ln_post.normalized_shape[0]
        self._image_size = clip_model.visual.image_size[0]
        self._patch_size = clip_model.visual.patch_size[0]
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            _, image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            return image_features
