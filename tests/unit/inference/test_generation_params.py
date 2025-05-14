import contextlib
import inspect
from importlib.util import find_spec
from unittest import mock
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
    RemoteVLLMInferenceEngine,
    SambanovaInferenceEngine,
    SGLangInferenceEngine,
    VLLMInferenceEngine,
)

vllm_import_failed = find_spec("vllm") is None
llama_cpp_import_failed = find_spec("llama_cpp") is None

SUPPORTED_INFERENCE_ENGINES = [
    RemoteInferenceEngine,
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    SambanovaInferenceEngine,
    SGLangInferenceEngine,
    VLLMInferenceEngine,
    RemoteVLLMInferenceEngine,
    GoogleVertexInferenceEngine,
]


@pytest.fixture
def sample_conversation():
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
        ]
    )


@pytest.fixture
def model_params():
    return ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_pad_token="<|endoftext|>",
        tokenizer_name="gpt2",
        load_pretrained_weights=False,
    )


@pytest.fixture
def generation_params_fields():
    """Get all field names from GenerationParams."""
    return set(inspect.signature(GenerationParams).parameters.keys())


def _should_skip_engine(engine_class) -> bool:
    return (engine_class == VLLMInferenceEngine and vllm_import_failed) or (
        engine_class == LlamaCppInferenceEngine and llama_cpp_import_failed
    )


def _mock_engine(engine_class):
    """Mock the engine to avoid loading non-existent models."""

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.eos_token = "<eos>"
    mock_model = mock.MagicMock()
    mock_model.generate = mock.MagicMock()  # Add generate attribute

    if engine_class == VLLMInferenceEngine:
        mock_llm = mock.MagicMock()
        mock_ctx = patch.multiple(
            "oumi.inference.vllm_inference_engine",
            vllm=mock.MagicMock(LLM=mock.MagicMock(return_value=mock_llm)),
            build_tokenizer=mock.MagicMock(return_value=mock_tokenizer),
        )
    elif engine_class == LlamaCppInferenceEngine:
        mock_ctx = patch("llama_cpp.Llama.from_pretrained")
    elif engine_class == SGLangInferenceEngine:
        mock_ctx = patch.multiple(
            "oumi.inference.sglang_inference_engine",
            build_tokenizer=mock.MagicMock(return_value=mock_tokenizer),
            build_processor=mock.MagicMock(return_value=None),
            is_image_text_llm=mock.MagicMock(return_value=False),
        )
    elif engine_class == NativeTextInferenceEngine:
        mock_ctx = patch.multiple(
            "oumi.inference.native_text_inference_engine",
            build_model=mock.MagicMock(return_value=mock_model),
            build_tokenizer=mock.MagicMock(return_value=mock_tokenizer),
            build_processor=mock.MagicMock(return_value=None),
            is_image_text_llm=mock.MagicMock(return_value=False),
        )
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
def test_generation_params_used_in_inference(
    engine_class, sample_conversation, model_params
):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

    with (
        patch.object(
            engine_class, "_infer", return_value=[sample_conversation]
        ) as mock_infer,
        mock_ctx,
    ):
        remote_params = RemoteParams(api_url="<placeholder>")
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=model_params, remote_params=remote_params
            )
        else:
            engine = engine_class(model_params)

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
        )
        inference_config = InferenceConfig(
            model=model_params,
            generation=generation_params,
            remote_params=remote_params,
        )

        result = engine.infer_online([sample_conversation], inference_config)

        # Check that the result is as expected
        assert result == [sample_conversation]

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
    engine_class, sample_conversation, model_params
):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

    with (
        patch.object(
            engine_class, "_infer", return_value=[sample_conversation]
        ) as mock_infer,
        mock_ctx,
    ):
        remote_params = RemoteParams(api_url="<placeholder>")
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=model_params, remote_params=remote_params
            )
        else:
            engine = engine_class(model_params)

        generation_params = GenerationParams()
        inference_config = InferenceConfig(
            model=model_params,
            generation=generation_params,
            remote_params=remote_params,
        )

        result = engine.infer_online([sample_conversation], inference_config)

        assert result == [sample_conversation]

        mock_infer.assert_called_once()
        called_params = mock_infer.call_args[0][1].generation
        assert called_params.max_new_tokens == 1024
        assert called_params.temperature == 0.0
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
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

    with mock_ctx:
        remote_params = RemoteParams(api_url="<placeholder>")
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=model_params, remote_params=remote_params
            )
        else:
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
        (GoogleVertexInferenceEngine, "frequency_penalty", 0.5),
        (GoogleVertexInferenceEngine, "presence_penalty", 0.5),
        (VLLMInferenceEngine, "logit_bias", {1: 1.0}),
        (LlamaCppInferenceEngine, "num_beams", 8),
    ],
)
def test_unsupported_params_warning(
    engine_class, unsupported_param, value, model_params, sample_conversation, caplog
):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

    with (
        mock_ctx,
        patch.object(engine_class, "_infer", return_value=[sample_conversation]),
    ):
        remote_params = RemoteParams(api_url="test")
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=model_params, remote_params=remote_params
            )
        else:
            engine = engine_class(model_params)

        # Create generation params with the unsupported parameter
        params_dict = {
            "max_new_tokens": 100,  # Add a supported param
            unsupported_param: value,
        }

        generation_params = GenerationParams(**params_dict)
        inference_config = InferenceConfig(
            model=model_params, generation=generation_params
        )
        if issubclass(engine_class, RemoteInferenceEngine):
            inference_config.remote_params = remote_params

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
        (LlamaCppInferenceEngine, "num_beams", 1),
    ],
)
def test_no_warning_for_default_values(
    engine_class, param, default_value, model_params, sample_conversation, caplog
):
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

    with (
        mock_ctx,
        patch.object(engine_class, "_infer", return_value=[sample_conversation]),
    ):
        remote_params = RemoteParams(api_url="test")
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=model_params, remote_params=remote_params
            )
        else:
            engine = engine_class(model_params)

        params_dict = {
            "max_new_tokens": 100,  # Add a supported param
            param: default_value,
        }

        generation_params = GenerationParams(**params_dict)
        inference_config = InferenceConfig(
            model=model_params, generation=generation_params
        )
        if issubclass(engine_class, RemoteInferenceEngine):
            inference_config.remote_params = remote_params

        engine.infer([sample_conversation], inference_config)

        # Check that no warning was logged for this parameter
        assert not any(
            record.levelname == "WARNING"
            and f"{engine_class.__name__} does not support {param}" in record.message
            for record in caplog.records
        )


@pytest.mark.parametrize(
    "engine_class",
    SUPPORTED_INFERENCE_ENGINES,
)
def test_supported_params_are_accessed(engine_class, model_params, sample_conversation):
    """Test that all supported parameters are actually accessed during inference."""
    if _should_skip_engine(engine_class):
        pytest.skip(f"{engine_class.__name__} is not available")

    mock_ctx = _mock_engine(engine_class)

    class AccessTrackingGenerationParams(GenerationParams):
        """A version of GenerationParams that tracks which parameters are accessed."""

        _accessed_params: set[str]

        def __init__(self, **kwargs):
            self._accessed_params: set[str] = set()
            super().__init__(**kwargs)

        def __getattribute__(self, name):
            # No need to track access to private attributes or methods
            if not name.startswith("_"):
                # Use object.__getattribute__ to avoid infinite recursion
                accessed_params = object.__getattribute__(self, "_accessed_params")
                accessed_params.add(name)
            return object.__getattribute__(self, name)

        @property
        def accessed_params(self):
            return self._accessed_params

        def clear(self):
            self._accessed_params.clear()

    with mock_ctx, mock.patch.object(engine_class, "_check_unsupported_params"):
        remote_params = RemoteParams(api_url="test")
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=model_params, remote_params=remote_params
            )
        else:
            engine = engine_class(model_params)

        # Create config with tracking
        tracked_params = AccessTrackingGenerationParams()
        tracked_params.clear()

        inference_config = InferenceConfig(
            model=model_params, generation=tracked_params
        )

        if issubclass(engine_class, RemoteInferenceEngine):
            inference_config.remote_params = remote_params

            # To avoid running inference, we just call the method that converts
            # the conversation to the API input. This should access most of the
            # parameters.
            engine._convert_conversation_to_api_input(
                sample_conversation, tracked_params, model_params
            )
        elif engine_class == LlamaCppInferenceEngine:
            with patch.object(engine, "_llm") as mock_llm:
                mock_llm.create_chat_completion.return_value = {
                    "choices": [{"message": {"content": "test"}}]
                }

                engine.infer([sample_conversation], inference_config)
        elif engine_class == NativeTextInferenceEngine:
            inference_config.generation.exclude_prompt_from_response = False
            engine.infer([sample_conversation], inference_config)
        else:
            engine.infer([sample_conversation], inference_config)

        # Get params that were supported but never accessed
        unused_params = engine.get_supported_params() - tracked_params.accessed_params

        assert not unused_params, (
            f"{engine_class.__name__} claims to support these parameters "
            f"but never accessed them: {unused_params}"
        )

        # Get params that were accessed but not marked as supported
        unregistered_params = (
            tracked_params.accessed_params - engine.get_supported_params()
        )
        unregistered_params.remove("accessed_params")  # Test param, ignore

        assert not unregistered_params, (
            f"{engine_class.__name__} accessed 'unsupported' parameters: "
            f"{unregistered_params}"
        )
