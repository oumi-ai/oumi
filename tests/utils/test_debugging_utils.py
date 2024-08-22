import pytest
import torch

from lema.utils.debugging_utils import (
    get_nvidia_gpu_memory_utilization,
    get_nvidia_gpu_temperature,
    log_nvidia_gpu_memory_utilization,
    log_nvidia_gpu_temperature,
)


def is_cuda_available_and_initialized():
    return torch.cuda.is_available() and torch.cuda.is_initialized()


@pytest.mark.skipif(
    not is_cuda_available_and_initialized(),
    reason="CUDA is not available",
)
def test_nvidia_gpu_memory_utilization():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        for device_index in range(0, num_devices):
            memory_mib = get_nvidia_gpu_memory_utilization(device_index)
            assert memory_mib > 1024  # Must have at least 1 GB
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


@pytest.mark.skipif(
    is_cuda_available_and_initialized(),
    reason="CUDA is available",
)
def test_nvidia_gpu_memory_utilization_no_cuda():
    assert get_nvidia_gpu_memory_utilization() == 0.0
    log_nvidia_gpu_memory_utilization()


@pytest.mark.skipif(
    not is_cuda_available_and_initialized(),
    reason="CUDA is not available",
)
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


@pytest.mark.skipif(
    is_cuda_available_and_initialized(),
    reason="CUDA is available",
)
def test_nvidia_gpu_temperature_no_cuda():
    assert get_nvidia_gpu_temperature() == 0.0
    log_nvidia_gpu_temperature()
