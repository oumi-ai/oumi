from typing import Optional

from lema.utils.logging import logger

try:
    # The library is only useful for NVIDIA GPUs, and
    # may not be installed for other vendors e.g., AMD
    import pynvml
except ModuleNotFoundError:
    pynvml = None

# TODO: Add support for `amdsmi.amdsmi_init()`` for AMD GPUs


def _initialize_pynvml() -> bool:
    """Attempts to initialize pynvml library. Returns True on success."""
    if pynvml is None:
        return False

    if not hasattr(_initialize_pynvml, "pynvml_initialized"):
        _initialize_pynvml.pynvml_initialized = False

    if not _initialize_pynvml.pynvml_initialized:
        try:
            pynvml.nvmlInit()
            _initialize_pynvml.pynvml_initialized = True
        except Exception:
            logger.exception("Failed to initialize pynvml library.")

    return _initialize_pynvml.pynvml_initialized


def _initialize_pynvml_and_get_pynvml_device_count() -> Optional[int]:
    """Attempts to initialize pynvml library.

    Returns device count on success, or None otherwise.
    """
    if pynvml is None or not _initialize_pynvml():
        return None
    return int(pynvml.nvmlDeviceGetCount())


def get_nvidia_gpu_memory_utilization(device_index: int = 0) -> float:
    """Returns amount of memory being used on an Nvidia GPU in MiB."""
    if pynvml is None:
        return 0.0

    device_count = _initialize_pynvml_and_get_pynvml_device_count()
    if device_count is None or device_count <= 0:
        return 0.0
    elif device_index < 0 or device_index >= device_count:
        raise ValueError(
            f"Device index ({device_index}) must be "
            f"within the [0, {device_count}) range."
        )

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.used) // 1024**2
    except Exception as _:
        return 0.0


def log_nvidia_gpu_memory_utilization() -> None:
    """Prints amount of memory being used on an Nvidia GPU."""
    logger.info(f"GPU memory occupied: {get_nvidia_gpu_memory_utilization()} MiB.")


# nvmlDeviceGetTemperature
