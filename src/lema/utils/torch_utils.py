import torch

from lema.logging import logger
from lema.utils.debugging_utils import get_nvidia_gpu_memory_utilization


def device_cleanup() -> None:
    """Empty's cuda cache, good to do before and after training for cleanup."""
    if torch.cuda.is_available():
        logger.debug("Cleaning up GPU memory.")
        logger.debug(
            "GPU memory occupied before cleanup: "
            f"{get_nvidia_gpu_memory_utilization()} MB"
        )

        torch.cuda.empty_cache()

        logger.debug(f"Memory after cleanup: {get_nvidia_gpu_memory_utilization()} MB")


def limit_per_process_memory(percent: float = 0.95) -> None:
    """Limits process memory by a certain percentage.

    On Windows and WSL, there's a pool of 'shared gpu memory'.
    This pool is using the RAM (slow) on one's machine rather than actual
    VRAM (fast). Setting this value ensures your machine never uses the slow
    memory and OOMs instead. Note that this may not be needed on Linux machines
    since this is an OS-level feature.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(percent)
