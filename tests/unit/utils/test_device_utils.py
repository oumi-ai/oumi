import numpy as np
import torch

from oumi.utils.device_utils import (
    get_nvidia_gpu_fan_speeds,
    get_nvidia_gpu_memory_utilization,
    get_nvidia_gpu_power_usage,
    get_nvidia_gpu_runtime_info,
    get_nvidia_gpu_temperature,
    log_nvidia_gpu_fan_speeds,
    log_nvidia_gpu_memory_utilization,
    log_nvidia_gpu_power_usage,
    log_nvidia_gpu_runtime_info,
    log_nvidia_gpu_temperature,
)
from tests.markers import requires_cuda_initialized, requires_cuda_not_available


@requires_cuda_initialized()
def test_nvidia_gpu_memory_utilization():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        for device_index in range(0, num_devices):
            memory_mib = get_nvidia_gpu_memory_utilization(device_index)
            assert memory_mib > 1  # Must have at least 1 MB
            assert memory_mib < 1024 * 1024  # No known GPU has 1 TB of VRAM yet.
            log_nvidia_gpu_memory_utilization(device_index)

        # Test default argument value
        assert get_nvidia_gpu_memory_utilization() == get_nvidia_gpu_memory_utilization(
            0
        )
    else:
        # Test default argument value
        assert get_nvidia_gpu_memory_utilization() == 0.0

    log_nvidia_gpu_memory_utilization()


@requires_cuda_not_available()
def test_nvidia_gpu_memory_utilization_no_cuda():
    assert get_nvidia_gpu_memory_utilization() == 0.0
    log_nvidia_gpu_memory_utilization()


@requires_cuda_initialized()
def test_nvidia_gpu_temperature():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        for device_index in range(0, num_devices):
            temperature = get_nvidia_gpu_temperature(device_index)
            assert temperature > 0 and temperature < 100
            log_nvidia_gpu_temperature(device_index)

        # Test default argument value
        temperature = get_nvidia_gpu_temperature()
        assert temperature > 0 and temperature < 100
    else:
        # Test default argument value
        assert get_nvidia_gpu_temperature() == 0.0

    log_nvidia_gpu_temperature()


@requires_cuda_not_available()
def test_nvidia_gpu_temperature_no_cuda():
    assert get_nvidia_gpu_temperature() == 0.0
    log_nvidia_gpu_temperature()


@requires_cuda_initialized()
def test_nvidia_gpu_fan_speeds():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        for device_index in range(0, num_devices):
            fan_speeds = get_nvidia_gpu_fan_speeds(device_index)
            if fan_speeds:
                assert len(fan_speeds) > 0
                fan_speeds = np.array(fan_speeds)
                assert np.all(fan_speeds >= 0)
                assert np.all(fan_speeds <= 100)
            else:
                assert fan_speeds == tuple()
            log_nvidia_gpu_fan_speeds(device_index)

        # Test default argument value
        fan_speeds = get_nvidia_gpu_fan_speeds()
        if fan_speeds:
            assert len(fan_speeds) > 0
            fan_speeds = np.array(fan_speeds)
            assert np.all(fan_speeds >= 0)
            assert np.all(fan_speeds <= 100)
        else:
            assert fan_speeds == tuple()
        log_nvidia_gpu_fan_speeds(device_index)
    else:
        # Test default argument value
        assert get_nvidia_gpu_fan_speeds() == tuple()

    log_nvidia_gpu_fan_speeds()


@requires_cuda_not_available()
def test_nvidia_gpu_fan_speeds_no_cuda():
    assert get_nvidia_gpu_fan_speeds() == tuple()
    log_nvidia_gpu_fan_speeds()


@requires_cuda_initialized()
def test_nvidia_gpu_power_usage():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        for device_index in range(0, num_devices):
            watts = get_nvidia_gpu_power_usage(device_index)
            assert watts > 0 and watts < 2000
            log_nvidia_gpu_power_usage(device_index)

        # Test default argument value
        watts = get_nvidia_gpu_power_usage()
        assert watts > 0 and watts < 2000
    else:
        # Test default argument value
        assert get_nvidia_gpu_power_usage() == 0.0

    log_nvidia_gpu_power_usage()


@requires_cuda_not_available()
def test_nvidia_gpu_power_usage_no_cuda():
    assert get_nvidia_gpu_power_usage() == 0.0
    log_nvidia_gpu_power_usage()


@requires_cuda_initialized()
def test_nvidia_gpu_runtime_info():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        for device_index in range(0, num_devices):
            info = get_nvidia_gpu_runtime_info(device_index)
            assert info is not None
            assert info.device_index == device_index
            assert info.device_count == num_devices
            assert info.used_memory_mb is not None and info.used_memory_mb > 0
            assert (
                info.temperature is not None
                and info.temperature >= 0
                and info.temperature <= 100
            )
            assert info.fan_speed is None or (
                info.fan_speed >= 0 and info.fan_speed <= 100
            )
            assert info.fan_speeds is None or len(info.fan_speeds) > 0
            assert info.power_usage_watts is not None and info.power_usage_watts >= 0
            assert info.power_limit_watts is not None and info.power_limit_watts > 0
            assert (
                info.gpu_utilization is not None
                and info.gpu_utilization >= 0
                and info.gpu_utilization <= 100
            )
            assert (
                info.memory_utilization is not None
                and info.memory_utilization >= 0
                and info.memory_utilization <= 100
            )
            assert info.performance_state is not None
            assert info.performance_state >= 0 and info.performance_state <= 32
            assert (
                info.clock_speed_graphics is not None and info.clock_speed_graphics > 0
            )
            assert info.clock_speed_sm is not None and info.clock_speed_sm > 0
            assert info.clock_speed_memory is not None and info.clock_speed_memory > 0

            log_nvidia_gpu_runtime_info(device_index)

        # Test default argument value
        info = get_nvidia_gpu_runtime_info()
        assert info is not None
        assert info.device_index == 0
        assert info.device_count == num_devices
        assert info.used_memory_mb is not None and info.used_memory_mb > 0
        assert (
            info.temperature is not None
            and info.temperature >= 0
            and info.temperature <= 100
        )
        assert info.fan_speed is None or (info.fan_speed >= 0 and info.fan_speed <= 100)
        assert info.fan_speeds is None or len(info.fan_speeds) > 0
        assert info.power_usage_watts is not None and info.power_usage_watts >= 0
        assert info.power_limit_watts is not None and info.power_limit_watts > 0
        assert (
            info.gpu_utilization is not None
            and info.gpu_utilization >= 0
            and info.gpu_utilization <= 100
        )
        assert (
            info.memory_utilization is not None
            and info.memory_utilization >= 0
            and info.memory_utilization <= 100
        )
        assert info.performance_state is not None
        assert info.performance_state >= 0 and info.performance_state <= 32
        assert info.clock_speed_graphics is not None and info.clock_speed_graphics > 0
        assert info.clock_speed_sm is not None and info.clock_speed_sm > 0
        assert info.clock_speed_memory is not None and info.clock_speed_memory > 0
    else:
        # Test default argument value
        assert get_nvidia_gpu_runtime_info() is None

    log_nvidia_gpu_runtime_info()


@requires_cuda_not_available()
def test_nvidia_gpu_runtime_info_no_cuda():
    assert get_nvidia_gpu_runtime_info() is None
    log_nvidia_gpu_runtime_info()
