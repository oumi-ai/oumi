import copy

from oumi.utils.logging import logger

from .clip_convnext_encoder import CLIPConvNextTower
from .clip_encoder import ClipVisionTower
from .dino_encoder import DinoVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(
            f"Vision Tower is not specified in the config: {vision_tower_cfg}"
        )

    # CLIP-based Vision Towers
    if "openai/clip" in vision_tower.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower}")
        return ClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "clip-convnext" in vision_tower.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower}")
        return CLIPConvNextTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower.lower():
        logger.info(f"Loading **DINO Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Supervised Vision Towers

    # Other Vision Towers

    raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    vision_tower_aux_name_list = getattr(
        vision_tower_cfg,
        "mm_vision_tower_aux_list",
        getattr(vision_tower_cfg, "vision_tower_aux_list", None),
    )
    vision_tower_aux_token_len_list = getattr(
        vision_tower_cfg,
        "mm_vision_tower_aux_token_len_list",
        getattr(vision_tower_cfg, "vision_tower_aux_token_len_list", None),
    )
    vision_tower_aux_list = []
    for vision_tower_aux_name, vision_tower_aux_token_len in zip(
        vision_tower_aux_name_list, vision_tower_aux_token_len_list
    ):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_aux_name += f"-interp{vision_tower_aux_token_len}"

        # CLIP-based Vision Towers
        if "openai/clip" in vision_tower_aux_name.lower():
            logger.info(
                f"Loading **OpenAI CLIP** Vision Tower: {vision_tower_aux_name}"
            )
            vision_tower_aux_list.append(
                ClipVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )
        elif "siglip" in vision_tower_aux_name.lower():
            logger.info(
                f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_aux_name}"
            )
            vision_tower_aux_list.append(
                SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )
        elif "clip-convnext" in vision_tower_aux_name.lower():
            logger.info(
                f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_aux_name}"
            )
            vision_tower_aux_list.append(
                CLIPConvNextTower(vision_tower_aux_name, args=config, **kwargs)
            )
        # SSL-based Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower():
            logger.info(f"Loading **DINO Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(
                DinoVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )
        # Supervised Vision Towers
        # Other Vision Towers
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower_aux_name}")
    return vision_tower_aux_list
