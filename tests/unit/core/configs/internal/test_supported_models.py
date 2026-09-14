from unittest import mock

import pytest

from oumi.core.configs.internal.supported_models import (
    find_internal_model_config,
    find_internal_model_config_using_model_name,
    find_model_hf_config,
    find_model_type_using_model_name,
    get_all_models_map,
    is_dual_mode_model_type,
    is_vision_language_model_type,
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


class _FakeConfig:
    """Stand-in HF config whose class identity drives the mapping lookups."""


def _patch_mappings(causal_cls, vlm_cls):
    """Patch the two transformers auto-mappings to return the given classes."""
    causal_map = mock.MagicMock()
    causal_map._model_mapping.get.return_value = causal_cls
    vlm_map = mock.MagicMock()
    vlm_map._model_mapping.get.return_value = vlm_cls
    return mock.patch.multiple(
        "oumi.core.configs.internal.supported_models",
        MODEL_FOR_CAUSAL_LM_MAPPING=causal_map,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING=vlm_map,
    )


def test_dual_mode_true_when_causal_class_distinct():
    # qwen3_5: Qwen3_5ForCausalLM != Qwen3_5ForConditionalGeneration
    with _patch_mappings(causal_cls=object, vlm_cls=type("VLM", (), {})):
        assert is_dual_mode_model_type(_FakeConfig()) is True  # pyright: ignore[reportArgumentType]


def test_dual_mode_false_when_same_class():
    # gemma3: both mappings resolve to Gemma3ForConditionalGeneration
    same = type("SameCls", (), {})
    with _patch_mappings(causal_cls=same, vlm_cls=same):
        assert is_dual_mode_model_type(_FakeConfig()) is False  # pyright: ignore[reportArgumentType]


def test_dual_mode_false_when_no_causal_mapping():
    # qwen3_vl: no AutoModelForCausalLM entry
    with _patch_mappings(causal_cls=None, vlm_cls=type("VLM", (), {})):
        assert is_dual_mode_model_type(_FakeConfig()) is False  # pyright: ignore[reportArgumentType]


def test_dual_mode_false_when_no_vlm_mapping():
    # plain text model: no ImageTextToText entry
    with _patch_mappings(causal_cls=type("Causal", (), {}), vlm_cls=None):
        assert is_dual_mode_model_type(_FakeConfig()) is False  # pyright: ignore[reportArgumentType]


def test_vision_language_true_when_vlm_mapping_exists():
    # Any model with an ImageTextToText class (vision-only or dual-mode).
    with _patch_mappings(causal_cls=None, vlm_cls=type("VLM", (), {})):
        assert is_vision_language_model_type(_FakeConfig()) is True  # pyright: ignore[reportArgumentType]


def test_vision_language_false_when_no_vlm_mapping():
    # Plain text model: no ImageTextToText class.
    with _patch_mappings(causal_cls=type("Causal", (), {}), vlm_cls=None):
        assert is_vision_language_model_type(_FakeConfig()) is False  # pyright: ignore[reportArgumentType]


def test_qwen3_5_registered_as_vlm():
    models = get_all_models_map()
    for mt in ("qwen3_5", "qwen3_5_moe"):
        assert mt in models, f"{mt} missing from registry"
        assert models[mt].config.visual_config is not None, (
            f"{mt} should carry a visual_config (VLM by default)"
        )


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
