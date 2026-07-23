from unittest import mock

import pytest

from oumi.core.configs.internal.supported_models import (
    find_internal_model_config,
    find_internal_model_config_using_model_name,
    find_model_hf_config,
    is_dual_mode_model_type,
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
