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
    global pynvml
    if pynvml is None:
        return False

    try:
        pynvml.nvmlInit()
    except Exception:
        logger.error(
            "Failed to initialize pynvml library. All pynvml calls will be disabled."
        )
        pynvml = None

    return pynvml is not None


def _initialize_pynvml_and_get_pynvml_device_count() -> Optional[int]:
    """Attempts to initialize pynvml library.

    Returns device count on success, or None otherwise.
    """
    global pynvml
    # The call to `pynvml is None` is technically redundant but exists here
    # to make pyright happy.
    if pynvml is None or not _initialize_pynvml():
        return None
    return int(pynvml.nvmlDeviceGetCount())


def get_nvidia_gpu_memory_utilization(device_index: int = 0) -> float:
    """Returns amount of memory being used on an Nvidia GPU in MiB."""
    global pynvml
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
    except Exception:
        logger.exception(f"Failed to get GPU memory info for device: {device_index}")
        return 0.0


def log_nvidia_gpu_memory_utilization(
    device_index: int = 0, log_prefix: str = ""
) -> None:
    """Prints amount of memory being used on an Nvidia GPU."""
    memory_mib = get_nvidia_gpu_memory_utilization(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU memory occupied: {memory_mib} MiB.")


def get_nvidia_gpu_temperature(device_index: int = 0) -> float:
    """Returns the current temperature readings for the device, in degrees C."""
    global pynvml
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
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        return float(temperature)
    except Exception:
        logger.exception(f"Failed to get GPU temperature for device: {device_index}")
        return 0.0


def log_nvidia_gpu_temperature(device_index: int = 0, log_prefix: str = "") -> None:
    """Prints amount of memory being used on an Nvidia GPU."""
    temperature = get_nvidia_gpu_temperature(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU temperature: {temperature} C.")
