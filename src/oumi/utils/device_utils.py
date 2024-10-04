from typing import NamedTuple, Optional, Sequence

from oumi.utils.logging import logger

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


class NVidiaGpuRuntimeInfo(NamedTuple):
    device_index: int
    """Zero-based device index."""

    device_count: int
    """Total number of GPU devices on this node."""

    used_memory_mb: Optional[float] = None
    """Used GPU memory in MB."""

    temperature: Optional[int] = None
    """GPU temperature in Celcius."""

    fan_speeds: Optional[Sequence[int]] = None
    """An array of GPU fan speeds.

    The array's length is equal to the number of fans per GPU (can be multiple).
    Speed values are in [0, 100] range.
    """

    power_usage_watts: Optional[float] = None
    """GPU power usage in Watts."""


def _get_nvidia_gpu_runtime_info_impl(
    device_index: int = 0,
    *,
    memory: bool = False,
    temperature: bool = False,
    fan_speed: bool = False,
    power_usage: bool = False,
) -> Optional[NVidiaGpuRuntimeInfo]:
    global pynvml
    if pynvml is None:
        return None

    device_count = _initialize_pynvml_and_get_pynvml_device_count()
    if device_count is None or device_count <= 0:
        return None
    elif device_index < 0 or device_index >= device_count:
        raise ValueError(
            f"Device index ({device_index}) must be "
            f"within the [0, {device_count}) range."
        )

    try:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:
        logger.exception(f"Failed to get GPU handle for device: {device_index}")
        return None

    used_memory_mb_value: Optional[float] = None
    if memory:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            used_memory_mb_value = float(info.used) // 1024**2
        except Exception:
            logger.exception(
                f"Failed to get GPU memory info for device: {device_index}"
            )
            return None

    temperature_value: Optional[int] = None
    if temperature:
        try:
            temperature_value = pynvml.nvmlDeviceGetTemperature(
                gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            logger.exception(
                f"Failed to get GPU temperature for device: {device_index}"
            )
            return None

    fan_speeds_value: Optional[Sequence[int]] = None
    if fan_speed:
        try:
            fan_count = pynvml.nvmlDeviceGetNumFans(gpu_handle)
            fan_speeds_value = [0] * fan_count
            for i in range(fan_count):
                speed = pynvml.nvmlDeviceGetFanSpeed_v2(gpu_handle, i)
                fan_speeds_value[i] = speed
            # Make it immutable.
            fan_speeds_value = tuple(fan_speeds_value)
        except Exception:
            logger.exception(f"Failed to get GPU fan speeds for device: {device_index}")
            return None

    power_usage_watts_value: Optional[float] = None
    if power_usage:
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
            power_usage_watts_value = float(milliwatts) * 1e-3
        except Exception:
            logger.exception(
                f"Failed to get GPU power usage for device: {device_index}"
            )
            return None

    return NVidiaGpuRuntimeInfo(
        device_index=device_index,
        device_count=device_count,
        used_memory_mb=used_memory_mb_value,
        temperature=temperature_value,
        fan_speeds=fan_speeds_value,
        power_usage_watts=power_usage_watts_value,
    )


def get_nvidia_gpu_runtime_info(
    device_index: int = 0,
    *,
    memory: bool = False,
    temperature: bool = False,
    fan_speed: bool = False,
    power_usage: bool = False,
) -> Optional[NVidiaGpuRuntimeInfo]:
    """Returns runtime stats for Nvidia GPU."""
    return _get_nvidia_gpu_runtime_info_impl(
        device_index=device_index,
        memory=True,
        temperature=True,
        fan_speed=True,
        power_usage=True,
    )


def get_nvidia_gpu_memory_utilization(device_index: int = 0) -> float:
    """Returns amount of memory being used on an Nvidia GPU in MiB."""
    info: Optional[NVidiaGpuRuntimeInfo] = _get_nvidia_gpu_runtime_info_impl(
        device_index=device_index, memory=True
    )
    return (
        info.used_memory_mb
        if (info is not None and info.used_memory_mb is not None)
        else 0.0
    )


def log_nvidia_gpu_memory_utilization(
    device_index: int = 0, log_prefix: str = ""
) -> None:
    """Prints amount of memory being used on an Nvidia GPU."""
    memory_mib = get_nvidia_gpu_memory_utilization(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU memory occupied: {memory_mib} MiB.")


def get_nvidia_gpu_temperature(device_index: int = 0) -> int:
    """Returns the current temperature readings for the device, in degrees C."""
    info: Optional[NVidiaGpuRuntimeInfo] = _get_nvidia_gpu_runtime_info_impl(
        device_index=device_index,
        temperature=True,
    )
    return (
        info.temperature if (info is not None and info.temperature is not None) else 0
    )


def log_nvidia_gpu_temperature(device_index: int = 0, log_prefix: str = "") -> None:
    """Prints the current temperature readings for the device, in degrees C."""
    temperature = get_nvidia_gpu_temperature(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU temperature: {temperature} C.")


def get_nvidia_gpu_fan_speeds(device_index: int = 0) -> Sequence[int]:
    """Returns the current fan speeds for NVIDIA GPU device."""
    info: Optional[NVidiaGpuRuntimeInfo] = _get_nvidia_gpu_runtime_info_impl(
        device_index=device_index, fan_speed=True
    )
    return (
        info.fan_speeds
        if (info is not None and info.fan_speeds is not None)
        else tuple()
    )


def log_nvidia_gpu_fan_speeds(device_index: int = 0, log_prefix: str = "") -> None:
    """Prints the current NVIDIA GPU fan speeds."""
    fan_speeds = get_nvidia_gpu_fan_speeds(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU fan speeds: {fan_speeds}.")


def get_nvidia_gpu_power_usage(device_index: int = 0) -> float:
    """Returns the current power usage for NVIDIA GPU device."""
    info: Optional[NVidiaGpuRuntimeInfo] = _get_nvidia_gpu_runtime_info_impl(
        device_index=device_index, power_usage=True
    )
    return (
        info.power_usage_watts
        if (info is not None and info.power_usage_watts is not None)
        else 0.0
    )


def log_nvidia_gpu_power_usage(device_index: int = 0, log_prefix: str = "") -> None:
    """Prints the current NVIDIA GPU power usage."""
    power_usage = get_nvidia_gpu_power_usage(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU power usage: {power_usage:1}W.")
