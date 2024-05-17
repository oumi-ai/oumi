import pynvml


def print_nvidia_gpu_memory_utilization():
    """Print amount of memory being used on an Nvidia GPU."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
