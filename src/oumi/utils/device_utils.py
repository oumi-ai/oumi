# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from pprint import pformat
from typing import NamedTuple, Optional

from oumi.utils.logging import logger

try:
    # The library is only useful for NVIDIA GPUs, and
    # may not be installed for other vendors e.g., AMD
    import pynvml  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    pynvml = None

# TODO: OPE-562 - Add support for `amdsmi.amdsmi_init()`` for AMD GPUs


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
    """Contains misc NVIDIA GPU measurements and stats retrieved by `pynvml`.

    The majority of fields are optional. You can control whether they are
    populated by setting boolean query parameters of
    `_get_nvidia_gpu_runtime_info_impl(, ...)` such as `memory`, `temperature`,
    `fan_speed`, etc.
    """

    device_index: int
    """Zero-based device index."""

    device_count: int
    """Total number of GPU devices on this node."""

    used_memory_mb: Optional[float] = None
    """Used GPU memory in MB."""

    temperature: Optional[int] = None
    """GPU temperature in Celcius."""

    fan_speed: Optional[int] = None
    """GPU fan speed in [0,100] range."""

    fan_speeds: Optional[Sequence[int]] = None
    """An array of GPU fan speeds.

    The array's length is equal to the number of fans per GPU (can be multiple).
    Speed values are in [0, 100] range.
    """

    power_usage_watts: Optional[float] = None
    """GPU power usage in Watts."""

    power_limit_watts: Optional[float] = None
    """GPU power limit in Watts."""

    gpu_utilization: Optional[int] = None
    """GPU compute utilization. Range: [0,100]."""

    memory_utilization: Optional[int] = None
    """GPU memory utilization. Range: [0,100]."""

    performance_state: Optional[int] = None
    """See `nvmlPstates_t`. Valid values are in [0,15] range, or 32 if unknown.

    0 corresponds to Maximum Performance.
    15 corresponds to Minimum Performance.
    """

    clock_speed_graphics: Optional[int] = None
    """Graphics clock speed (`NVML_CLOCK_GRAPHICS`) in MHz."""

    clock_speed_sm: Optional[int] = None
    """SM clock speed (`NVML_CLOCK_SM`) in MHz."""

    clock_speed_memory: Optional[int] = None
    """Memory clock speed (`NVML_CLOCK_MEM`) in MHz."""


def _get_nvidia_gpu_runtime_info_impl(
    device_index: int = 0,
    *,
    memory: bool = False,
    temperature: bool = False,
    fan_speed: bool = False,
    power_usage: bool = False,
    utilization: bool = False,
    performance_state: bool = False,
    clock_speed: bool = False,
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

    fan_speed_value: Optional[int] = None
    fan_speeds_value: Optional[Sequence[int]] = None
    if fan_speed:
        try:
            fan_speed_value = pynvml.nvmlDeviceGetFanSpeed(gpu_handle)
        except Exception:
            # The `GetFanSpeed` function fails on many systems
            # Only do DEBUG-level logging to reduce noise.
            logger.debug(
                f"Failed to get GPU fan speed for device: {device_index}", exc_info=True
            )

        if fan_speed_value is not None:
            fan_speeds_value = tuple([fan_speed_value])
            if hasattr(pynvml, "nvmlDeviceGetNumFans"):
                try:
                    fan_count = pynvml.nvmlDeviceGetNumFans(gpu_handle)
                    value = [0] * fan_count
                    for i in range(fan_count):
                        speed = pynvml.nvmlDeviceGetFanSpeed_v2(gpu_handle, i)
                        value[i] = speed
                    # Make it immutable.
                    fan_speeds_value = tuple(value)
                except Exception:
                    fan_speeds_value = tuple([fan_speed_value])

    power_usage_watts_value: Optional[float] = None
    power_limit_watts_value: Optional[float] = None
    if power_usage:
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
            power_usage_watts_value = float(milliwatts) * 1e-3

            milliwatts = pynvml.nvmlDeviceGetPowerManagementLimit(gpu_handle)
            power_limit_watts_value = float(milliwatts) * 1e-3
        except Exception:
            logger.exception(
                f"Failed to get GPU power usage for device: {device_index}"
            )
            return None

    gpu_utilization_value: Optional[float] = None
    memory_utilization_value: Optional[float] = None
    if utilization:
        try:
            result = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            gpu_utilization_value = int(result.gpu)
            memory_utilization_value = int(result.memory)
        except Exception:
            logger.exception(
                f"Failed to get GPU utilization for device: {device_index}"
            )
            return None

    performance_state_value: Optional[int] = None
    if performance_state:
        try:
            performance_state_value = int(
                pynvml.nvmlDeviceGetPerformanceState(gpu_handle)
            )
        except Exception:
            logger.exception(
                f"Failed to get GPU performance state for device: {device_index}"
            )
            return None

    clock_speed_graphics_value: Optional[int] = None
    clock_speed_sm_value: Optional[int] = None
    clock_speed_memory_value: Optional[int] = None
    if clock_speed:
        try:
            clock_speed_graphics_value = int(
                pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
            )
            clock_speed_sm_value = int(
                pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_SM)
            )
            clock_speed_memory_value = int(
                pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_MEM)
            )
        except Exception:
            logger.exception(
                f"Failed to get GPU clock speed for device: {device_index}"
            )
            return None

    return NVidiaGpuRuntimeInfo(
        device_index=device_index,
        device_count=device_count,
        used_memory_mb=used_memory_mb_value,
        temperature=temperature_value,
        fan_speed=fan_speed_value,
        fan_speeds=fan_speeds_value,
        power_usage_watts=power_usage_watts_value,
        power_limit_watts=power_limit_watts_value,
        gpu_utilization=gpu_utilization_value,
        memory_utilization=memory_utilization_value,
        performance_state=performance_state_value,
        clock_speed_graphics=clock_speed_graphics_value,
        clock_speed_sm=clock_speed_sm_value,
        clock_speed_memory=clock_speed_memory_value,
    )


def get_nvidia_gpu_runtime_info(
    device_index: int = 0,
) -> Optional[NVidiaGpuRuntimeInfo]:
    """Returns runtime stats for Nvidia GPU."""
    return _get_nvidia_gpu_runtime_info_impl(
        device_index=device_index,
        memory=True,
        temperature=True,
        fan_speed=True,
        power_usage=True,
        utilization=True,
        performance_state=True,
        clock_speed=True,
    )


def log_nvidia_gpu_runtime_info(device_index: int = 0, log_prefix: str = "") -> None:
    """Prints the current NVIDIA GPU runtime info."""
    info = get_nvidia_gpu_runtime_info(device_index)
    logger.info(f"{log_prefix.rstrip()} GPU runtime info: {pformat(info)}.")


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
    logger.info(f"{log_prefix.rstrip()} GPU power usage: {power_usage:.2f}W.")
