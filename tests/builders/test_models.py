from unittest.mock import patch

import pytest
import torch

from lema.builders.models import _patch_model_for_liger_kernel
from lema.core.configs import ModelParams


@pytest.fixture
def mock_liger_kernel():
    with patch("lema.builders.models.liger_kernel.transformers") as mock:
        yield mock


@pytest.mark.parametrize(
    "model_name, expected_function",
    [
        ("meta-llama/Llama-2-7b-hf", "apply_liger_kernel_to_llama"),
        ("meta-llama/Meta-Llama-3.1-70B", "apply_liger_kernel_to_llama"),
        ("mistralai/Mistral-7B-v0.1", "apply_liger_kernel_to_mistral"),
        ("google/gemma-7b", "apply_liger_kernel_to_gemma"),
        ("mistralai/Mixtral-8x7B-v0.1", "apply_liger_kernel_to_mixtral"),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Liger Kernel is not supported on CPU"
)
def test_patch_model_for_liger_kernel(mock_liger_kernel, model_name, expected_function):
    model_params = ModelParams(model_name=model_name)
    _patch_model_for_liger_kernel(model_params.model_name)
    getattr(mock_liger_kernel, expected_function).assert_called_once()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Liger Kernel is not supported on CPU"
)
def test_patch_model_for_liger_kernel_unsupported():
    model_params = ModelParams(model_name="gpt2")
    with pytest.raises(ValueError, match="Unsupported model: gpt2"):
        _patch_model_for_liger_kernel(model_params.model_name)


def test_patch_model_for_liger_kernel_import_error():
    with patch("lema.builders.models.liger_kernel", None):
        model_params = ModelParams(model_name="llama-7b")
        with pytest.raises(ImportError, match="Liger Kernel not installed"):
            _patch_model_for_liger_kernel(model_params.model_name)
