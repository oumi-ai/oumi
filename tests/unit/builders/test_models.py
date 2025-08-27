from typing import Optional
from unittest.mock import Mock, patch

import pytest
from torch import nn

from oumi.builders.models import (
    _get_model_type,
    _patch_model_for_liger_kernel,
    build_chat_template,
    build_huggingface_model,
    build_tokenizer,
    is_image_text_llm,
)
from oumi.core.configs import ModelParams
from oumi.core.configs.internal.supported_models import find_model_hf_config
from oumi.utils.logging import logger


@pytest.fixture
def mock_liger_kernel():
    with patch("oumi.builders.models.liger_kernel") as mock:
        yield mock


def create_mock_model(model_type):
    model = Mock(spec=nn.Module)
    model.config = Mock()
    model.config.model_type = model_type
    return model


@pytest.mark.parametrize(
    "model_type",
    [
        "llama",
        "mixtral",
        "mistral",
        "gemma",
    ],
)
def test_patch_model_for_liger_kernel(mock_liger_kernel, model_type):
    model = create_mock_model(model_type)

    _patch_model_for_liger_kernel(model)

    mock_liger_kernel.transformers._apply_liger_kernel.assert_called_once_with(
        model_type
    )


def test_patch_model_for_liger_kernel_no_config(mock_liger_kernel):
    model = Mock(spec=nn.Module)
    with pytest.raises(ValueError, match=f"Could not find model type for: {model}"):
        _patch_model_for_liger_kernel(model)


def test_patch_model_for_liger_kernel_import_error():
    with patch("oumi.builders.models.liger_kernel", None):
        model = create_mock_model("llama")
        with pytest.raises(ImportError, match="Liger Kernel not installed"):
            _patch_model_for_liger_kernel(model)


def test_get_model_type():
    # Test with valid model
    model = create_mock_model("llama")
    assert _get_model_type(model) == "llama"

    # Test with no config
    model = Mock(spec=nn.Module)
    assert _get_model_type(model) is None

    # Test with config but no model_type
    model = Mock(spec=nn.Module)
    model.config = Mock()
    model.config.model_type = None
    assert _get_model_type(model) is None


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

    with (
        patch("oumi.builders.models.get_oumi_root_directory"),
        patch("oumi.builders.models.load_file") as mock_load_file,
    ):
        mock_load_file.return_value = template_content

        result = build_chat_template("test_template")

        assert result == expected
        mock_load_file.assert_called_once()


@pytest.mark.parametrize(
    "model_name, trust_remote_code, expected_result",
    [
        ("MlpEncoder", False, False),  # Custom model
        ("CnnClassifier", False, False),  # Custom model
        ("openai-community/gpt2", False, False),
        ("HuggingFaceTB/SmolLM2-135M-Instruct", False, False),
        ("llava-hf/llava-1.5-7b-hf", False, True),
        ("Salesforce/blip2-opt-2.7b", False, True),
        ("microsoft/Phi-3-vision-128k-instruct", True, True),
        ("HuggingFaceTB/SmolVLM-Instruct", False, True),
    ],
)
def test_is_image_text_llm(
    model_name: str, trust_remote_code: bool, expected_result: bool
):
    assert (
        is_image_text_llm(
            ModelParams(model_name=model_name, trust_remote_code=trust_remote_code)
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "model_name, trust_remote_code, template_name, expected_padding_side",
    [
        ("openai-community/gpt2", False, "gpt2", "right"),
        ("HuggingFaceTB/SmolLM2-135M-Instruct", False, None, "right"),
        ("llava-hf/llava-1.5-7b-hf", False, "llava", "left"),
        ("microsoft/Phi-3-vision-128k-instruct", True, "phi3-instruct", "right"),
        ("Qwen/Qwen2-VL-2B-Instruct", True, "qwen2-vl-instruct", "left"),
        # These models require allowlisting:
        # ("meta-llama/Llama-3.2-3B-Instruct", False, None, "right"),
    ],
)
def test_default_chat_template_in_build_tokenizer(
    model_name: str,
    trust_remote_code: bool,
    template_name: Optional[str],
    expected_padding_side: str,
):
    tokenizer = build_tokenizer(
        ModelParams(model_name=model_name, trust_remote_code=trust_remote_code)
    )

    debug_tag = f"template_name: {template_name} model_name: {model_name}"
    if template_name:
        expected = build_chat_template(template_name=template_name)
        if tokenizer.chat_template != expected:
            logger.info(
                f"Tokenizer chat template:\n\n{tokenizer.chat_template}\n\n"
                f"Expected chat template:\n\n{expected}\n\n"
            )
        assert tokenizer.chat_template == expected, debug_tag
    else:
        # Using the model's built-in config.
        assert tokenizer.chat_template is not None, (
            f"Unspecified built-in template: {debug_tag}"
        )
        assert len(tokenizer.chat_template) > 0, f"Empty built-in template: {debug_tag}"

    # Also check padding side here.
    assert hasattr(tokenizer, "padding_side")
    assert tokenizer.padding_side == expected_padding_side


def test_find_model_hf_config_with_custom_kwargs():
    mock_config = Mock()
    mock_config.model_type = "test_model"

    custom_kwargs = {
        "max_position_embeddings": 2048,
    }

    with patch(
        "oumi.core.configs.internal.supported_models.transformers.AutoConfig.from_pretrained"
    ) as mock_from_pretrained:
        mock_from_pretrained.return_value = (mock_config, {})

        result = find_model_hf_config(
            model_name="test-model",
            trust_remote_code=False,
            revision="main",
            **custom_kwargs,
        )

        # Verify the result
        assert result == mock_config

        # Verify AutoConfig.from_pretrained was called with custom kwargs
        mock_from_pretrained.assert_called_once_with(
            "test-model",
            trust_remote_code=False,
            return_unused_kwargs=True,
            revision="main",
            **custom_kwargs,
        )


def test_find_model_hf_config_logs_unused_kwargs():
    """Test that find_model_hf_config logs a warning for unused kwargs."""
    mock_config = Mock()
    mock_config.model_type = "test_model"
    unused_kwargs = {"unsupported_param": "value"}

    with (
        patch(
            "oumi.core.configs.internal.supported_models.transformers.AutoConfig.from_pretrained"
        ) as mock_from_pretrained,
        patch("oumi.core.configs.internal.supported_models.logger") as mock_logger,
    ):
        mock_from_pretrained.return_value = (mock_config, unused_kwargs)

        find_model_hf_config(
            model_name="test-model",
            trust_remote_code=False,
            unsupported_param="value",
        )

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            f"Unused kwargs found in 'test-model' config: {unused_kwargs}."
        )


def test_build_huggingface_model_passes_model_kwargs_to_find_model_hf_config():
    """Test that build_huggingface_model passes model_kwargs to find_model_hf_config."""
    model_kwargs = {
        "max_position_embeddings": 2048,
    }

    model_params = ModelParams(
        model_name="test-model", trust_remote_code=False, model_kwargs=model_kwargs
    )

    mock_config = Mock()
    mock_config.model_type = "llama"
    mock_config.use_cache = True

    mock_model = Mock()
    mock_model.config = mock_config

    with (
        patch("oumi.builders.models.find_model_hf_config") as mock_find_config,
        patch("oumi.builders.models._get_transformers_model_class") as mock_get_class,
        patch("oumi.builders.models.get_device_rank_info") as mock_device_info,
        patch("oumi.builders.models.is_using_accelerate_fsdp", return_value=False),
    ):
        mock_find_config.return_value = mock_config
        mock_model_class = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_get_class.return_value = mock_model_class
        mock_device_info.return_value = Mock(world_size=1, local_rank=0)

        result = build_huggingface_model(model_params)

        # Verify find_model_hf_config was called with model_kwargs
        mock_find_config.assert_called_once_with(
            "test-model", trust_remote_code=False, revision=None, **model_kwargs
        )

        # Verify the model was built successfully
        assert result == mock_model
