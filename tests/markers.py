import pytest
import torch

from oumi.utils.hf_utils import find_hf_token


def requires_gpus(count: int = 1, min_gb: float = 0.0) -> pytest.MarkDecorator:
    """Decorator to skip a test if the required number of GPUs is not available.

    Args:
        count (int): The number of GPUs required for the test. Defaults to 1.
        min_gb: Min required GPU VRAM in GB-s. Has no effect if zero or negative.

    Returns:
        pytest.MarkDecorator: A decorator that skips the test if the required
            number of GPUs is not available.
    """

    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA is not available")

    gpu_count = torch.cuda.device_count()

    error_message = ""
    if gpu_count < count:
        error_message = (
            f"Not enough GPUs to run the test: requires '{count}',"
            f" got '{torch.cuda.device_count()}'"
        )
    elif min_gb > 0.0:
        eps = 1e-2  # relative tolerance
        for device_idx in range(gpu_count):
            _, total_memory = torch.cuda.mem_get_info(device_idx)
            total_memory_gb = float(total_memory) / float(1024 * 1024 * 1024)
            if total_memory_gb < min_gb * (1 - eps):
                device_name = torch.cuda.get_device_name(device_idx)
                error_message = (
                    "Not enough GPU memory to run the test: "
                    f"requires {min_gb:.3f}GB, got {total_memory_gb:.3f}GB. "
                    f"GPU: {device_name}"
                ) + (f" ({device_idx + 1} of {gpu_count})" if gpu_count > 1 else "")

    return pytest.mark.skipif(len(error_message) > 0, reason=error_message)


def requires_cuda_initialized() -> pytest.MarkDecorator:
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA is not available")

    if not torch.cuda.is_initialized():
        torch.cuda.init()

    return pytest.mark.skipif(
        not torch.cuda.is_initialized(), reason="CUDA is not initialized"
    )


def requires_cuda_not_available() -> pytest.MarkDecorator:
    return pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")


def requires_hf_token() -> pytest.MarkDecorator:
    return pytest.mark.skipif(not find_hf_token(), reason="HF token is not available")


def requires_pdf_support():
    from importlib.util import find_spec

    return pytest.mark.skipif(
        not find_spec("pdf2image"), reason="PDF support is not available"
    )
