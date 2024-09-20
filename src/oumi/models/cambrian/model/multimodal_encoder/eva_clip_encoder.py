import torch
from open_clip import create_model_from_pretrained
from timm.models.eva import Eva

from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower


class EvaClipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()

    def load_model(self, device_map=None):
        if self.vision_tower_name in (
            "eva/CLIP-ViT-L-336",
            "timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
        ):
            self.vision_model = "evaclip"
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k')
            self.image_processor = ProcessorWrapper(processor, height=336, width=336)
            self._patch_size = 14
        elif self.vision_tower_name in (
            "eva/CLIP-ViT-L-224",
            "timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
        ):
            self.vision_model = "evaclip"
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k')
            self.image_processor = ProcessorWrapper(processor, height=224, width=224)
            self._patch_size = 14
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        self.vision_tower: Eva = clip_model.visual.trunk
        self.vision_tower.output_tokens = True
        self._hidden_size = 1024

        self._image_size = self.vision_tower.pretrained_cfg["input_size"][-1]

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self._feature_select(image_forward_outs).to(images.dtype)

            return image_features
