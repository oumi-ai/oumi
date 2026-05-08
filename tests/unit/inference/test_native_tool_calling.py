import pytest

from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.inference.native_tool_calling import (
    NATIVE_TOOL_CALLING_ENGINES,
    supports_native_tool_calling,
)

_SUPPORTED = [
    InferenceEngineType.OPENAI,
    InferenceEngineType.ANTHROPIC,
    InferenceEngineType.VLLM,
    InferenceEngineType.REMOTE,
    InferenceEngineType.REMOTE_VLLM,
    InferenceEngineType.SGLANG,
    InferenceEngineType.DEEPSEEK,
    InferenceEngineType.FIREWORKS,
    InferenceEngineType.OPENROUTER,
    InferenceEngineType.CEREBRAS,
    InferenceEngineType.PARASAIL,
    InferenceEngineType.SAMBANOVA,
    InferenceEngineType.TOGETHER,
    InferenceEngineType.GOOGLE_VERTEX,
    InferenceEngineType.HUGGING_FACE_ROUTER,
]

_UNSUPPORTED = [
    InferenceEngineType.NATIVE,
    InferenceEngineType.LLAMACPP,
    InferenceEngineType.BEDROCK,
    InferenceEngineType.GOOGLE_GEMINI,
]


@pytest.mark.parametrize("engine_type", _SUPPORTED)
def test_supported_engines_return_true(engine_type):
    assert supports_native_tool_calling(engine_type) is True


@pytest.mark.parametrize("engine_type", _UNSUPPORTED)
def test_unsupported_engines_return_false(engine_type):
    assert supports_native_tool_calling(engine_type) is False


def test_none_returns_false():
    assert supports_native_tool_calling(None) is False


def test_allowlist_disjoint_from_unsupported():
    unsupported_set = frozenset(_UNSUPPORTED)
    assert NATIVE_TOOL_CALLING_ENGINES.isdisjoint(unsupported_set)


def test_supported_plus_unsupported_equals_full_enum():
    all_engine_types = frozenset(InferenceEngineType)
    classified = NATIVE_TOOL_CALLING_ENGINES | frozenset(_UNSUPPORTED)
    assert classified == all_engine_types, (
        f"Unclassified engine types: {all_engine_types - classified}. "
        "Update NATIVE_TOOL_CALLING_ENGINES or the unsupported list."
    )
