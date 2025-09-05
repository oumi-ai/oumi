from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from oumi.utils.device_utils import (
    _get_nvidia_gpu_runtime_info_impl,
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


def _create_mock_pynvml_with_not_supported_error():
    """Helper function to create a mock pynvml with NVMLError_NotSupported."""
    mock_pynvml = MagicMock()

    # Mock successful initialization and device count
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 1

    # Mock successful GPU handle retrieval by default
    mock_gpu_handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_gpu_handle

    # Create a mock NVMLError_NotSupported exception
    class MockNVMLError_NotSupported(Exception):
        pass

    mock_pynvml.NVMLError_NotSupported = MockNVMLError_NotSupported

    return mock_pynvml, MockNVMLError_NotSupported


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "utilization",
            "mock_method": "nvmlDeviceGetUtilizationRates",
            "function_kwargs": {"utilization": True},
            "expected_log_message": "GPU utilization not supported for device: 0",
            "expected_none_fields": ["gpu_utilization", "memory_utilization"],
            "should_return_none": False,
        },
        {
            "name": "memory",
            "mock_method": "nvmlDeviceGetMemoryInfo",
            "function_kwargs": {"memory": True},
            "expected_log_message": (
                "pyNVML GPU memory info not supported for device: 0"
            ),
            "expected_none_fields": ["used_memory_mb"],
            "should_return_none": False,
        },
        {
            "name": "temperature",
            "mock_method": "nvmlDeviceGetTemperature",
            "function_kwargs": {"temperature": True},
            "expected_log_message": (
                "pyNVML GPU temperature not supported for device: 0"
            ),
            "expected_none_fields": ["temperature"],
            "should_return_none": False,
        },
        {
            "name": "fan_speed",
            "mock_method": "nvmlDeviceGetFanSpeed",
            "function_kwargs": {"fan_speed": True},
            "expected_log_message": (
                "pyNVML GPU fan speed not supported for device: 0"
            ),
            "expected_none_fields": ["fan_speed", "fan_speeds"],
            "should_return_none": False,
        },
        {
            "name": "power_usage",
            "mock_method": "nvmlDeviceGetPowerUsage",
            "function_kwargs": {"power_usage": True},
            "expected_log_message": (
                "pyNVML GPU power usage not supported for device: 0"
            ),
            "expected_none_fields": ["power_usage_watts", "power_limit_watts"],
            "should_return_none": False,
        },
        {
            "name": "performance_state",
            "mock_method": "nvmlDeviceGetPerformanceState",
            "function_kwargs": {"performance_state": True},
            "expected_log_message": (
                "pyNVML GPU performance state not supported for device: 0"
            ),
            "expected_none_fields": ["performance_state"],
            "should_return_none": False,
        },
        {
            "name": "clock_speed",
            "mock_method": "nvmlDeviceGetClockInfo",
            "function_kwargs": {"clock_speed": True},
            "expected_log_message": (
                "pyNVML GPU clock speed not supported for device: 0"
            ),
            "expected_none_fields": [
                "clock_speed_graphics",
                "clock_speed_sm",
                "clock_speed_memory",
            ],
            "should_return_none": False,
        },
    ],
)
def test_debug_logging_for_unsupported_gpu_operations(caplog, test_case):
    """Test that debug logging occurs when GPU operations are not supported."""
    mock_pynvml, MockNVMLError_NotSupported = (
        _create_mock_pynvml_with_not_supported_error()
    )

    # Set up the specific method to raise NotSupported error
    getattr(
        mock_pynvml, test_case["mock_method"]
    ).side_effect = MockNVMLError_NotSupported("Not supported")

    with patch("oumi.utils.device_utils.pynvml", mock_pynvml):
        with caplog.at_level("DEBUG", logger="oumi"):
            result = _get_nvidia_gpu_runtime_info_impl(
                device_index=0, **test_case["function_kwargs"]
            )

        # Verify debug message was logged
        assert any(
            test_case["expected_log_message"] in record.message
            for record in caplog.records
            if record.levelname == "DEBUG"
        ), f"Expected debug message not found: {test_case['expected_log_message']}"

        # Check return value and field values
        if test_case["should_return_none"]:
            assert result is None
        else:
            assert result is not None
            for field_name in test_case["expected_none_fields"]:
                assert getattr(result, field_name) is None, (
                    f"Expected {field_name} to be None"
                )
