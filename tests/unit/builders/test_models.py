from unittest.mock import patch

import pytest
import torch

from oumi.builders.models import (
    _patch_model_for_liger_kernel,
    build_chat_template,
    is_image_text_llm,
)
from oumi.core.configs import ModelParams


@pytest.fixture
def mock_liger_kernel():
    with patch("oumi.builders.models.liger_kernel.transformers") as mock:
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
    with patch("oumi.builders.models.liger_kernel", None):
        model_params = ModelParams(model_name="llama-7b")
        with pytest.raises(ImportError, match="Liger Kernel not installed"):
            _patch_model_for_liger_kernel(model_params.model_name)


@pytest.mark.parametrize(
    "template_name",
    ["zephyr"],
)
def test_build_chat_template_existing_templates(template_name):
    template = build_chat_template(template_name)

    assert template is not None
    assert len(template) > 0


def test_build_chat_template_nonexistent_template():
    with pytest.raises(FileNotFoundError) as exc_info:
        build_chat_template("nonexistent_template")

    assert "Chat template file not found" in str(exc_info.value)


def test_build_chat_template_removes_indentation_and_newlines():
    template_content = """
        {{ bos_token }}
        {% for message in messages %}
            {% if message['role'] == 'user' %}
                User: {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
                Assistant: {{ message['content'] }}
            {% endif %}
            {{ eos_token }}
        {% endfor %}
    """
    expected = (
        "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}{% endif %}{{ eos_token }}{% endfor %}"
    )

    with patch("oumi.builders.models.get_oumi_root_directory"), patch(
        "oumi.builders.models.load_file"
    ) as mock_load_file:
        mock_load_file.return_value = template_content

        result = build_chat_template("test_template")

        assert result == expected
        mock_load_file.assert_called_once()


@pytest.mark.parametrize(
    "model_name, expected_result",
    [
        ("openai-community/gpt2", False),
        ("llava-hf/llava-1.5-7b-hf", True),
        ("Salesforce/blip2-opt-2.7b", True),
    ],
)
def test_is_image_text_llm(model_name, expected_result):
    assert is_image_text_llm(ModelParams(model_name=model_name)) == expected_result
