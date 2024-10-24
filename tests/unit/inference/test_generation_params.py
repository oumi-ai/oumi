import contextlib
import inspect
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
    GoogleVertexInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
    VLLMInferenceEngine,
)

vllm_import_failed = find_spec("vllm") is None
llama_cpp_import_failed = find_spec("llama_cpp") is None

SUPPORTED_INFERENCE_ENGINES = [
    RemoteInferenceEngine,
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    VLLMInferenceEngine,
    GoogleVertexInferenceEngine,
]

# Mock model params for testing
MODEL_PARAMS = ModelParams(model_name="gpt2", tokenizer_pad_token="<|endoftext|>")

# Sample conversation for testing
SAMPLE_CONVERSATION = Conversation(
    messages=[
        Message(role=Role.USER, content="Hello, how are you?"),
    ]
)


@pytest.fixture
def sample_conversation():
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
        ]
    )


@pytest.fixture
def model_params():
    return ModelParams(model_name="gpt2", tokenizer_pad_token="<|endoftext|>")


@pytest.fixture
def generation_params_fields():
    """Get all field names from GenerationParams."""
    return set(inspect.signature(GenerationParams).parameters.keys())


@pytest.fixture
def sample_conversations() -> List[Conversation]:
    return [SAMPLE_CONVERSATION]


def _should_skip_engine(engine_class) -> bool:
    return (engine_class == VLLMInferenceEngine and vllm_import_failed) or (
        engine_class == LlamaCppInferenceEngine and llama_cpp_import_failed
    )


def _mock_engine(engine_class):
    """Mock the engine to avoid loading non-existent models."""
    if engine_class == VLLMInferenceEngine:
        mock_ctx = patch("vllm.LLM")
    elif engine_class == LlamaCppInferenceEngine:
        mock_ctx = patch("llama_cpp.Llama.from_pretrained")
    elif issubclass(engine_class, RemoteInferenceEngine):
        mock_ctx = patch("aiohttp.ClientSession")
    else:
        mock_ctx = contextlib.nullcontext()

    return mock_ctx


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


@pytest.mark.parametrize(
    "engine_class",
    SUPPORTED_INFERENCE_ENGINES,
)
def test_generation_params_used_in_inference(engine_class, sample_conversations):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

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
    SUPPORTED_INFERENCE_ENGINES,
)
def test_generation_params_defaults_used_in_inference(
    engine_class, sample_conversations
):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

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


@pytest.mark.parametrize(
    "engine_class",
    SUPPORTED_INFERENCE_ENGINES,
)
def test_supported_params_exist_in_config(
    engine_class, model_params, generation_params_fields
):
    mock_ctx = _mock_engine(engine_class)

    with mock_ctx:
        engine = engine_class(model_params)

        supported_params = engine.get_supported_params()

        # Additional check that all expected params exist in GenerationParams
        invalid_params = supported_params - generation_params_fields

        assert not invalid_params, (
            f"Test expects support for parameters that don't exist in "
            f"GenerationParams: {invalid_params}"
        )


@pytest.mark.parametrize(
    "engine_class,unsupported_param,value",
    [
        (AnthropicInferenceEngine, "min_p", 0.1),
        (AnthropicInferenceEngine, "frequency_penalty", 0.5),
        (VLLMInferenceEngine, "logit_bias", {1: 1.0}),
        (LlamaCppInferenceEngine, "remote_params", RemoteParams(api_url="test")),
    ],
)
def test_unsupported_params_warning(
    engine_class, unsupported_param, value, model_params, sample_conversation, caplog
):
    mock_ctx = _mock_engine(engine_class)

    with mock_ctx, patch.object(
        engine_class, "_infer", return_value=[sample_conversation]
    ):
        engine = engine_class(model_params)

        # Create generation params with the unsupported parameter
        params_dict = {
            "max_new_tokens": 100,  # Add a supported param
            unsupported_param: value,
        }
        if issubclass(engine_class, RemoteInferenceEngine):
            params_dict["remote_params"] = RemoteParams(api_url="test")
        generation_params = GenerationParams(**params_dict)
        inference_config = InferenceConfig(
            model=model_params, generation=generation_params
        )

        # Call infer which should trigger the warning
        engine.infer([sample_conversation], inference_config)

        # Check that warning was logged
        assert any(
            record.levelname == "WARNING"
            and f"{engine_class.__name__} does not support {unsupported_param}"
            in record.message
            for record in caplog.records
        )


@pytest.mark.parametrize(
    "engine_class,param,default_value",
    [
        (AnthropicInferenceEngine, "min_p", 0.0),
        (AnthropicInferenceEngine, "frequency_penalty", 0.0),
        (VLLMInferenceEngine, "logit_bias", {}),
        (LlamaCppInferenceEngine, "remote_params", None),
    ],
)
def test_no_warning_for_default_values(
    engine_class, param, default_value, model_params, sample_conversation, caplog
):
    mock_ctx = _mock_engine(engine_class)

    with mock_ctx, patch.object(
        engine_class, "_infer", return_value=[sample_conversation]
    ):
        engine = engine_class(model_params)

        params_dict = {
            "max_new_tokens": 100,  # Add a supported param
            param: default_value,
        }
        if issubclass(engine_class, RemoteInferenceEngine):
            params_dict["remote_params"] = RemoteParams(api_url="test")
        generation_params = GenerationParams(**params_dict)
        inference_config = InferenceConfig(
            model=model_params, generation=generation_params
        )

        engine.infer([sample_conversation], inference_config)

        # Check that no warning was logged for this parameter
        assert not any(
            record.levelname == "WARNING"
            and f"{engine_class.__name__} does not support {param}" in record.message
            for record in caplog.records
        )
