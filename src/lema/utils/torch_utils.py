import torch
from lema.utils.debugging_utils import print_nvidia_gpu_memory_utilization


def device_cleanup(verbose=False) -> None:
    """Empty's cuda cache, good to do before and after training for cleanup."""
    if torch.cuda.is_available():
        if verbose:
            print("Outputting max memory usage before cleanup")
            print_nvidia_gpu_memory_utilization()
        torch.cuda.empty_cache()

        if verbose:
            print("Memory after cleanup:")
            print_nvidia_gpu_memory_utilization()


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
