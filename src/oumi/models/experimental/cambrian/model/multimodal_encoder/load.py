# from ezcolorlog import root_logger as logger
from oumi.utils.logging import logger

from .clip_convnext_encoder import CLIPConvNextTower
from .clip_encoder import ClipVisionTower
from .dino_encoder import DinoVisionTower
from .siglip_encoder import SiglipVisionTower


def load_vision_model(vision_tower_name: str, args, **kwargs):
    """Load a vision tower model based on the model name

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
    if "siglip" in vision_tower_name.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_name}")
        return SiglipVisionTower(vision_tower_name, args=args, **kwargs)
    if "clip-convnext" in vision_tower_name.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_name}")
        return CLIPConvNextTower(vision_tower_name, args=args, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower_name.lower():
        logger.info(f"Loading **DINOv2** Vision Tower: {vision_tower_name}")
        return DinoVisionTower(vision_tower_name, args=args, **kwargs)

    # Supervised Vision Towers

    # Other Vision Towers

    raise ValueError(f"Unknown vision tower: {vision_tower_name}")
