from ezcolorlog import root_logger as logger

from .clip_convnext_encoder import CLIPConvNextTower
from .clip_encoder import ClipVisionTower
from .dfn_clip_encoder import DfnClipVisionTower
from .diffusion_encoder import DiffusionVisionTower
from .dino_encoder import DinoVisionTower
from .eva_clip_encoder import EvaClipVisionTower
from .ijepa_encoder import IJepaVisionTower
from .mae_encoder import MAEVisionTower
from .maws_encoder import MawsVisionTower
from .midas_encoder import MiDaSVisionTower
from .moco_encoder import MoCoVisionTower
from .sam_encoder import SAMVisionTower
from .siglip_encoder import SiglipVisionTower
from .supervised_vit_encoder import SupervisedViT_VisionTower


def load_vision_model(vision_tower_name: str, args, **kwargs):
    """
    Load a vision tower model based on the model name

    Args:
        vision_tower_name (str): The name of the vision tower model.
        args (argparse.Namespace): The arguments parsed from the command line.
        kwargs: Additional keyword arguments.
    """

    if vision_tower_name.lower().startswith("hybridmodel"):
        raise ValueError(
            "HybridModels must be loaded using the `multimodal_encoder.builderbuild_vision_tower()` function."
        )

    # CLIP-based Vision Towers
    if "openai/clip" in vision_tower_name.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower_name}")
        return ClipVisionTower(vision_tower_name, args=args, **kwargs)
    if "apple/dfn" in vision_tower_name.lower():
        logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower_name}")
        return DfnClipVisionTower(vision_tower_name, args=args, **kwargs)
    if "siglip" in vision_tower_name.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_name}")
        return SiglipVisionTower(vision_tower_name, args=args, **kwargs)
    if "eva/clip" in vision_tower_name.lower():
        logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower_name}")
        return EvaClipVisionTower(vision_tower_name, args=args, **kwargs)
    if "clip-convnext" in vision_tower_name.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_name}")
        return CLIPConvNextTower(vision_tower_name, args=args, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower_name.lower():
        logger.info(f"Loading **DINOv2** Vision Tower: {vision_tower_name}")
        return DinoVisionTower(vision_tower_name, args=args, **kwargs)
    if "maws/" in vision_tower_name.lower():
        logger.info(f"Loading **MAWS** Vision Tower: {vision_tower_name}")
        return MawsVisionTower(vision_tower_name, args=args, **kwargs)
    if "mae" in vision_tower_name.lower():
        logger.info(f"Loading **MAE** Vision Tower: {vision_tower_name}")
        return MAEVisionTower(vision_tower_name, args=args, **kwargs)
    if "moco" in vision_tower_name.lower():
        logger.info(f"Loading **MoCo** Vision Tower: {vision_tower_name}")
        return MoCoVisionTower(vision_tower_name, args=args, **kwargs)
    if "ijepa" in vision_tower_name.lower():
        logger.info(f"Loading **IJepa** Vision Tower: {vision_tower_name}")
        return IJepaVisionTower(vision_tower_name, args=args, **kwargs)

    # Supervised Vision Towers
    if "supervised-vit" in vision_tower_name.lower():
        logger.info(f"Loading **Supervised** Vision Tower: {vision_tower_name}")
        return SupervisedViT_VisionTower(vision_tower_name, args=args, **kwargs)

    # Other Vision Towers
    if "diffusion" in vision_tower_name.lower():
        logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower_name}")
        return DiffusionVisionTower(vision_tower_name, args=args, **kwargs)
    if "midas" in vision_tower_name.lower():
        logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower_name}")
        return MiDaSVisionTower(vision_tower_name, args=args, **kwargs)
    if "sam" in vision_tower_name.lower():
        logger.info(f"Loading **SAM Vision Tower: {vision_tower_name}")
        return SAMVisionTower(vision_tower_name, args=args, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower_name}")
