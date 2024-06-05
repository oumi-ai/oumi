try:
    # The library is only useful for NVIDIA GPUs, and
    # may not be installed for other vendors e.g., AMD
    import pynvml
except ModuleNotFoundError:
    pynvml = None

# TODO: Add support for `amdsmi.amdsmi_init()`` for AMD GPUs


def get_nvidia_gpu_memory_utilization() -> float:
    """Returns amount of memory being used on an Nvidia GPU in MB.

    TODO: Extend to support multiple GPUs.
    """
    if pynvml is None:
        return 0.0

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return float(info.used) // 1024**2


def print_nvidia_gpu_memory_utilization() -> None:
    """Prints amount of memory being used on an Nvidia GPU."""
    print(f"GPU memory occupied: {get_nvidia_gpu_memory_utilization()} MB.")
