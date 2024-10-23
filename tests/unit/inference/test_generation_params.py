import contextlib
from importlib.util import find_spec
from typing import List
from unittest.mock import patch

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
    VLLMInferenceEngine,
)

vllm_import_failed = find_spec("vllm") is None
llama_cpp_import_failed = find_spec("llama_cpp") is None


# Mock model params for testing
MODEL_PARAMS = ModelParams(model_name="gpt2", tokenizer_pad_token="<|endoftext|>")

# Sample conversation for testing
SAMPLE_CONVERSATION = Conversation(
    messages=[
        Message(role=Role.USER, content="Hello, how are you?"),
    ]
)


@pytest.fixture
def sample_conversations() -> List[Conversation]:
    return [SAMPLE_CONVERSATION]


def _should_skip_engine(engine_class) -> bool:
    return (engine_class == VLLMInferenceEngine and vllm_import_failed) or (
        engine_class == LlamaCppInferenceEngine and llama_cpp_import_failed
    )


@pytest.mark.parametrize(
    "engine_class",
    [
        RemoteInferenceEngine,
        AnthropicInferenceEngine,
        LlamaCppInferenceEngine,
        NativeTextInferenceEngine,
        VLLMInferenceEngine,
    ],
)
def test_generation_params(engine_class, sample_conversations):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    # We need to mock the Llama.from_pretrained call for LlamaCppInferenceEngine
    # otherwise it will try to load a non-existent model
    mock_ctx = (
        patch("llama_cpp.Llama.from_pretrained")
        if engine_class == LlamaCppInferenceEngine
        else contextlib.nullcontext()
    )

    with patch.object(
        engine_class, "_infer", return_value=sample_conversations
    ) as mock_infer, mock_ctx:
        engine = engine_class(MODEL_PARAMS)

        generation_params = GenerationParams(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_strings=["END"],
            stop_token_ids=[128001, 128008, 128009],
            logit_bias={1: 1.0, 2: -1.0},
            min_p=0.05,
            remote_params=RemoteParams(api_url="<placeholder>"),
        )
        inference_config = InferenceConfig(
            model=MODEL_PARAMS, generation=generation_params
        )

        result = engine.infer_online(sample_conversations, inference_config)

        # Check that the result is as expected
        assert result == sample_conversations

        # Check that _infer was called with the correct parameters
        mock_infer.assert_called_once()
        called_params = mock_infer.call_args[0][1].generation
        assert called_params.max_new_tokens == 100
        assert called_params.temperature == 0.7
        assert called_params.top_p == 0.9
        assert called_params.frequency_penalty == 0.1
        assert called_params.presence_penalty == 0.1
        assert called_params.stop_strings == ["END"]
        assert called_params.stop_token_ids == [128001, 128008, 128009]
        assert called_params.logit_bias == {1: 1.0, 2: -1.0}
        assert called_params.min_p == 0.05


@pytest.mark.parametrize(
    "engine_class",
    [
        RemoteInferenceEngine,
        AnthropicInferenceEngine,
        LlamaCppInferenceEngine,
        NativeTextInferenceEngine,
        VLLMInferenceEngine,
    ],
)
def test_generation_params_defaults(engine_class, sample_conversations):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    # We need to mock the Llama.from_pretrained call for LlamaCppInferenceEngine
    # otherwise it will try to load a non-existent model
    mock_ctx = (
        patch("llama_cpp.Llama.from_pretrained")
        if engine_class == LlamaCppInferenceEngine
        else contextlib.nullcontext()
    )

    with patch.object(
        engine_class, "_infer", return_value=sample_conversations
    ) as mock_infer, mock_ctx:
        engine = engine_class(MODEL_PARAMS)

        generation_params = GenerationParams(
            remote_params=RemoteParams(api_url="<placeholder>")
        )
        inference_config = InferenceConfig(
            model=MODEL_PARAMS, generation=generation_params
        )

        result = engine.infer_online(sample_conversations, inference_config)

        assert result == sample_conversations

        mock_infer.assert_called_once()
        called_params = mock_infer.call_args[0][1].generation
        assert called_params.max_new_tokens == 256
        assert called_params.temperature == 1.0
        assert called_params.top_p == 1.0
        assert called_params.frequency_penalty == 0.0
        assert called_params.presence_penalty == 0.0
        assert called_params.stop_strings is None
        assert called_params.logit_bias == {}
        assert called_params.min_p == 0.0


def test_generation_params_validation():
    with pytest.raises(ValueError, match="Temperature must be non-negative."):
        GenerationParams(temperature=-0.1)

    with pytest.raises(ValueError, match="top_p must be between 0 and 1."):
        GenerationParams(top_p=1.1)

    with pytest.raises(
        ValueError, match="Logit bias for token 1 must be between -100 and 100."
    ):
        GenerationParams(logit_bias={1: 101})

    with pytest.raises(ValueError, match="min_p must be between 0 and 1."):
        GenerationParams(min_p=1.1)
