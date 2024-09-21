import copy
import os

from oumi.utils.logging import logger

from .clip_convnext_encoder import CLIPConvNextTower
from .clip_encoder import ClipVisionTower
from .dfn_clip_encoder import DfnClipVisionTower
from .diffusion_encoder import DiffusionVisionTower
from .dino_encoder import DinoVisionTower
from .eva_clip_encoder import EvaClipVisionTower
from .hybrid_encoder import HybridVisionTower
from .ijepa_encoder import IJepaVisionTower
from .mae_encoder import MAEVisionTower
from .maws_encoder import MawsVisionTower
from .midas_encoder import MiDaSVisionTower
from .moco_encoder import MoCoVisionTower
from .sam_encoder import SAMVisionTower
from .siglip_encoder import SiglipVisionTower
from .supervised_vit_encoder import SupervisedViT_VisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(f'Vision Tower is not specified in the config: {vision_tower_cfg}')
    if "maws" in vision_tower.lower():
        logger.info(f"Loading **MAWS** Vision Tower: {vision_tower}")
        return MawsVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # CLIP-based Vision Towers
    if "openai/clip" in vision_tower.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower}")
        return ClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "apple/dfn" in vision_tower.lower():
        logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower}")
        return DfnClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "eva/clip" in vision_tower.lower():
        logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower}")
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "clip-convnext" in vision_tower.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower}")
        return CLIPConvNextTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower.lower():
        logger.info(f"Loading **DINO Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "mae" in vision_tower.lower():
        logger.info(f"Loading **MAE** Vision Tower: {vision_tower}")
        return MAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "moco" in vision_tower.lower():
        logger.info(f"Loading **MoCo** Vision Tower: {vision_tower}")
        return MoCoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "ijepa" in vision_tower.lower():
        logger.info(f"Loading **IJepa** Vision Tower: {vision_tower}")
        return IJepaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Supervised Vision Towers
    if "supervised-vit" in vision_tower.lower():
        logger.info(f"Loading **Supervised** Vision Tower: {vision_tower}")
        return SupervisedViT_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Other Vision Towers
    if "hybridmodel" in vision_tower.lower():
        logger.info(f"Loading **Hybrid** Vision Tower: {vision_tower}")
        return HybridVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "diffusion" in vision_tower.lower():
        logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower}")
        return DiffusionVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "midas" in vision_tower.lower():
        logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower}")
        return MiDaSVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "sam" in vision_tower.lower():
        logger.info(f"Loading **SAM Vision Tower: {vision_tower}")
        return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')



def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    vision_tower_aux_name_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_list', getattr(vision_tower_cfg, 'vision_tower_aux_list', None))
    vision_tower_aux_token_len_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_token_len_list', getattr(vision_tower_cfg, 'vision_tower_aux_token_len_list', None))
    vision_tower_aux_list = []
    for vision_tower_aux_name, vision_tower_aux_token_len in zip(vision_tower_aux_name_list, vision_tower_aux_token_len_list):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_aux_name += "-interp{}".format(vision_tower_aux_token_len)
        if "maws" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MAWS** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(MawsVisionTower(vision_tower_aux_name, args=config, **kwargs))

        # CLIP-based Vision Towers
        elif "openai/clip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(ClipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "apple/dfn" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(DfnClipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "siglip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "eva/clip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(EvaClipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "clip-convnext" in vision_tower_aux_name.lower():
            logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(CLIPConvNextTower(vision_tower_aux_name, args=config, **kwargs))

        # SSL-based Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower():
            logger.info(f"Loading **DINO Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(DinoVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "mae" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MAE** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(MAEVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "moco" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MoCo** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(MoCoVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "ijepa" in vision_tower_aux_name.lower():
            logger.info(f"Loading **IJepa** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(IJepaVisionTower(vision_tower_aux_name, args=config, **kwargs))

        # Supervised Vision Towers
        elif "supervised-vit" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Supervised** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(SupervisedViT_VisionTower(vision_tower_aux_name, args=config, **kwargs))

        # Other Vision Towers
        elif "hybridmodel" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Hybrid** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(HybridVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "diffusion" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(DiffusionVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "midas" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(MiDaSVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "sam" in vision_tower_aux_name.lower():
            logger.info(f"Loading **SAM Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(SAMVisionTower(vision_tower_aux_name, args=config, **kwargs))
        else:
            raise ValueError(f'Unknown vision tower: {vision_tower_aux_name}')
    return vision_tower_aux_list
