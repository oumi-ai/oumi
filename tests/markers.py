import pytest
import torch


def requires_gpus(count: int = 1) -> pytest.MarkDecorator:
    """Decorator to skip a test if the required number of GPUs is not available.

    Args:
        count (int): The number of GPUs required for the test. Defaults to 1.

    Returns:
        pytest.MarkDecorator: A decorator that skips the test if the required
            number of GPUs is not available.
    """

    message = ""

    cuda_unavailable = not torch.cuda.is_available()
    low_gpu_count = torch.cuda.device_count() < count

    if cuda_unavailable:
        message = "CUDA not available"

    elif low_gpu_count:
        message = (
            f"Not enough GPUs to run the test: requires '{count}',"
            f" got '{torch.cuda.device_count()}'"
        )

    return pytest.mark.skipif(cuda_unavailable or low_gpu_count, reason=message)


def requires_cuda_initialized() -> pytest.MarkDecorator:
    if not torch.cuda.is_available():
        return pytest.mark.skipif(
            not torch.cuda.is_initialized(), reason="CUDA is not available"
        )

    if not torch.cuda.is_initialized():
        torch.cuda.init()

    return pytest.mark.skipif(
        not torch.cuda.is_initialized(), reason="CUDA is not initialized"
    )
