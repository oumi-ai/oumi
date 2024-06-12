from typing import Optional

import torch

from lema.logging import logger
from lema.utils.debugging_utils import get_nvidia_gpu_memory_utilization


def device_cleanup() -> None:
    """Empties cuda cache, good to do before and after training for cleanup."""
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


def log_versioning_info() -> None:
    """Logs misc versioning information."""
    logger.info(f"Torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        logger.info("CUDA is not available!")
        return

    def _format_cudnn_version(v: Optional[int]) -> str:
        if v is None:
            return ""
        return ".".join(map(str, (v // 1000, v // 100 % 10, v % 100)))

    # For AMD GPUs, these functions return ROCm, MlOpen versions respectively.
    logger.info(
        f"CUDA version: {torch.version.cuda} "
        f"CuDNN version: {_format_cudnn_version(torch.backends.cudnn.version())}"
    )


def log_devices_info() -> None:
    """Logs high-level info about all available accelerator devices."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available!")
        return

    num_devices = torch.cuda.device_count()
    logger.info(f"CUDA devices: {num_devices}")

    def _mem_to_gb(x):
        return round(float(x) / 1024**3, 2)

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_allocated = torch.cuda.memory_allocated(i)
        mem_reserved = torch.cuda.memory_reserved(i)
        capability = torch.cuda.get_device_capability(i)
        logger.info(
            f"device({i})='{device_name}' "
            f"Capability: {capability} "
            f"Memory: [Total: {_mem_to_gb(mem_total)}GB "
            f"Free: {_mem_to_gb(mem_free)}GB "
            f"Allocated: {_mem_to_gb(mem_allocated)}GB "
            f"Cached: {_mem_to_gb(mem_reserved)}GB]"
        )
