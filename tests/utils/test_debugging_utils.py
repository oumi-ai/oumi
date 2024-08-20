import pytest
import torch

from lema.utils.debugging_utils import (
    get_nvidia_gpu_memory_utilization,
    get_nvidia_gpu_temperature,
)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_initialized()),
    reason="CUDA not available",
)
def test_get_nvidia_gpu_memory_utilization():
    num_devices = torch.cuda.device_count()
    for device_index in range(0, num_devices):
        memory_mib = get_nvidia_gpu_memory_utilization(device_index)
        assert memory_mib > 1024  # Must have at least 1 GB
        assert memory_mib < 1024 * 1024  # No known GPU has 1 TB of VRAM yet.


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_initialized()),
    reason="CUDA not available",
)
def test_get_nvidia_gpu_temperature():
    num_devices = torch.cuda.device_count()
    for device_index in range(0, num_devices):
        temperature = get_nvidia_gpu_temperature(device_index)
        assert temperature > 0 and temperature < 100
