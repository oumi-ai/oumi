import pytest

from oumi.core.configs.internal.supported_models import (
    find_internal_model_config,
    find_internal_model_config_using_model_name,
    find_model_hf_config,
    find_model_type_using_model_name,
)
from oumi.core.configs.params.model_params import ModelParams


@pytest.mark.parametrize(
    "model_name, trust_remote_code",
    [
        ("llava-hf/llava-1.5-7b-hf", False),
        ("microsoft/Phi-3-vision-128k-instruct", True),
        ("Qwen/Qwen2-VL-2B-Instruct", True),
        ("Salesforce/blip2-opt-2.7b", False),
        # Access is restricted (gated repo):
        # ("meta-llama/Llama-3.2-11B-Vision-Instruct", False),
    ],
)
def test_common_vlm_models(model_name: str, trust_remote_code):
    debug_tag = f"model_name: {model_name} trust_remote_code:{trust_remote_code}"
    assert (
        find_model_hf_config(model_name, trust_remote_code=trust_remote_code)
        is not None
    ), debug_tag

    assert (
        find_internal_model_config_using_model_name(
            model_name, trust_remote_code=trust_remote_code
        )
        is not None
    ), debug_tag

    assert (
        find_internal_model_config(
            ModelParams(model_name=model_name, trust_remote_code=trust_remote_code)
        )
        is not None
    ), debug_tag


#
# find_model_type_using_model_name
#


@pytest.mark.parametrize(
    "model_name,expected",
    [
        pytest.param("google/gemma-4-E2B-it", "gemma4", id="gemma-4"),
        pytest.param("zai-org/GLM-4.5", "glm4_moe", id="glm-4.5"),
        pytest.param("Qwen/Qwen3-0.6B", "qwen3", id="qwen3"),
    ],
)
def test_find_model_type_reports_text_model_architectures(model_name, expected):
    """Check that HF models have resolvable model type and config"""
    assert find_model_type_using_model_name(model_name, True) == expected
    assert find_internal_model_config_using_model_name(model_name, True) is None


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("MlpEncoder", id="custom-oumi-model"),
        pytest.param("CnnClassifier", id="custom-oumi-model-2"),
    ],
)
def test_find_model_type_returns_none_for_custom_models(model_name):
    """Oumi's own models have no HuggingFace config to read."""
    assert find_model_type_using_model_name(model_name, False) is None


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("does-not-exist/nope-123", id="unresolvable"),
        pytest.param("", id="empty"),
    ],
)
def test_find_model_type_propagates_unreadable_configs(model_name):
    """Same contract as the other lookups in this module: an unreadable config is an
    error, not a None. Callers that reach this point have already loaded the config.
    """
    with pytest.raises(OSError):
        find_model_type_using_model_name(model_name, False)
