from unittest.mock import patch

import pytest

from lema.builders.models import _patch_model_for_liger_kernel
from lema.core.configs import ModelParams


@pytest.fixture
def mock_liger_kernel():
    with patch("lema.builders.models.liger_kernel.transformers") as mock:
        yield mock


@pytest.mark.parametrize(
    "model_name, expected_function",
    [
        ("llama-7b", "apply_liger_kernel_to_llama"),
        ("Qwen2-7B-Chat", "apply_liger_kernel_to_qwen2"),
        ("phi-3", "apply_liger_kernel_to_phi3"),
        ("Mistral-7B-v0.1", "apply_liger_kernel_to_mistral"),
        ("gemma-7b", "apply_liger_kernel_to_gemma"),
        ("mixtral-8x7b", "apply_liger_kernel_to_mixtral"),
    ],
)
def test_patch_model_for_liger_kernel(mock_liger_kernel, model_name, expected_function):
    model_params = ModelParams(model_name=model_name)
    _patch_model_for_liger_kernel(model_params.model_name)
    getattr(mock_liger_kernel, expected_function).assert_called_once()


def test_patch_model_for_liger_kernel_unsupported():
    model_params = ModelParams(model_name="gpt2")
    with pytest.raises(ValueError, match="Unsupported model: gpt2"):
        _patch_model_for_liger_kernel(model_params.model_name)


def test_patch_model_for_liger_kernel_import_error():
    with patch("lema.builders.models.liger_kernel", None):
        model_params = ModelParams(model_name="llama-7b")
        with pytest.raises(ImportError, match="Liger Kernel not installed"):
            _patch_model_for_liger_kernel(model_params.model_name)
