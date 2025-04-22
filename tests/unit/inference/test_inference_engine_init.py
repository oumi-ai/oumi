import contextlib
from unittest import mock
from unittest.mock import patch

import pytest

from oumi.builders.inference_engines import ENGINE_MAP, build_inference_engine
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
    RemoteParams,
)
from oumi.inference import (
    AnthropicInferenceEngine,
    DeepSeekInferenceEngine,
    GoogleGeminiInferenceEngine,
    GoogleVertexInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    OpenAIInferenceEngine,
    ParasailInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SambanovaInferenceEngine,
    SGLangInferenceEngine,
    TogetherInferenceEngine,
    VLLMInferenceEngine,
)

# Check if optional dependencies are available
try:
    import vllm  # noqa: F401 # pyright: ignore[reportMissingImports]

    vllm_import_failed = False
except ImportError:
    vllm_import_failed = True

try:
    import llama_cpp  # noqa: F401 # pyright: ignore[reportMissingImports]

    llama_cpp_import_failed = False
except ImportError:
    llama_cpp_import_failed = True

# Group engines by whether they require remote params
LOCAL_ENGINES = [
    NativeTextInferenceEngine,
    VLLMInferenceEngine,
    LlamaCppInferenceEngine,
]

REMOTE_ENGINES = [
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SGLangInferenceEngine,
]

REMOTE_API_ENGINES = [
    AnthropicInferenceEngine,
    DeepSeekInferenceEngine,
    GoogleGeminiInferenceEngine,
    GoogleVertexInferenceEngine,
    OpenAIInferenceEngine,
    ParasailInferenceEngine,
    SambanovaInferenceEngine,
    TogetherInferenceEngine,
]


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


@pytest.mark.parametrize("engine_class", LOCAL_ENGINES)
def test_local_engine_init_with_model_params(engine_class):
    """Test that local engines can be initialized with just model params."""
    model_params = ModelParams(model_name="test-model")
    mock_engine_class = _mock_engine(engine_class)
    with mock_engine_class:
        engine = engine_class(model_params=model_params)
    assert engine._model_params.model_name == "test-model"


@pytest.mark.parametrize("engine_class", REMOTE_ENGINES + REMOTE_API_ENGINES)
def test_remote_engine_init_with_model_and_remote_params(engine_class):
    """Test that remote engines can be initialized with both model and remote params."""
    model_params = ModelParams(model_name="test-model")
    remote_params = RemoteParams(api_url="http://test.com", api_key="test-key")
    mock_engine_class = _mock_engine(engine_class)
    with mock_engine_class:
        engine = engine_class(model_params=model_params, remote_params=remote_params)
    assert engine._model_params.model_name == "test-model"
    assert engine._remote_params.api_url == "http://test.com"
    assert engine._remote_params.api_key == "test-key"


@pytest.mark.parametrize(
    "engine_class", LOCAL_ENGINES + REMOTE_ENGINES + REMOTE_API_ENGINES
)
def test_engine_init_missing_model_params_fails(engine_class):
    """Test that all engines fail when initialized without model params."""
    remote_params = RemoteParams(api_url="http://test.com", api_key="test-key")
    with pytest.raises(TypeError):
        if engine_class in LOCAL_ENGINES:
            engine_class()  # Should fail - missing model_params
        else:
            engine_class(
                remote_params=remote_params
            )  # Should fail - missing model_params


@pytest.mark.parametrize(
    "engine_class", LOCAL_ENGINES + REMOTE_ENGINES + REMOTE_API_ENGINES
)
def test_engine_init_with_invalid_params_fails(engine_class):
    """Test that all engines fail when initialized with invalid params."""
    with pytest.raises(TypeError):
        engine_class(invalid_param="test")  # Should fail - invalid param


@pytest.mark.parametrize("engine_class", LOCAL_ENGINES)
def test_local_engine_config_overrides_constructor_params(engine_class):
    """Test that InferenceConfig params override constructor params."""
    # Initialize with one set of params
    init_model_params = ModelParams(
        model_name="init-model",
        model_max_length=128,
        torch_dtype_str="float32",
    )
    mock_engine_class = _mock_engine(engine_class)
    with mock_engine_class:
        engine = engine_class(model_params=init_model_params)
    assert engine._model_params.model_name == "init-model"

    # Create config with different params
    config_model_params = ModelParams(
        model_name="config-model",
        model_max_length=256,
        torch_dtype_str="float16",
    )
    config = InferenceConfig(
        model=config_model_params,
        generation=GenerationParams(max_new_tokens=100),
    )

    # Mock _infer to avoid actual inference
    with patch.object(engine, "_infer") as mock_infer:
        engine.infer([], config)

        # Verify the config params were used
        call_args = mock_infer.call_args[0]
        assert len(call_args) >= 2  # Should have at least input and config args
        passed_config = call_args[1]
        assert passed_config.model.model_name == "config-model"
        assert passed_config.model.model_max_length == 256
        assert passed_config.model.torch_dtype_str == "float16"


@pytest.mark.parametrize("engine_class", REMOTE_ENGINES + REMOTE_API_ENGINES)
def test_remote_engine_config_overrides_constructor_params(engine_class):
    """Test that InferenceConfig params override constructor params."""
    # Initialize with one set of params
    init_model_params = ModelParams(
        model_name="init-model",
        model_max_length=128,
    )
    init_remote_params = RemoteParams(
        api_url="http://init.com",
        api_key="init-key",
        num_workers=1,
        max_retries=3,
    )
    mock_engine_class = _mock_engine(engine_class)
    with mock_engine_class:
        engine = engine_class(
            model_params=init_model_params,
            remote_params=init_remote_params,
        )
    assert engine._model_params.model_name == "init-model"
    assert engine._remote_params.api_url == "http://init.com"

    # Create config with different params
    config_model_params = ModelParams(
        model_name="config-model",
        model_max_length=256,
    )
    config_remote_params = RemoteParams(
        api_url="http://config.com",
        api_key="config-key",
        num_workers=2,
        max_retries=5,
    )
    config = InferenceConfig(
        model=config_model_params,
        remote_params=config_remote_params,
        generation=GenerationParams(max_new_tokens=100),
    )

    # Mock _infer to avoid actual inference
    with patch.object(engine, "_infer") as mock_infer:
        engine.infer([], config)

        # Verify the config params were used
        call_args = mock_infer.call_args[0]
        assert len(call_args) >= 2  # Should have at least input and config args
        passed_config = call_args[1]
        assert passed_config.model.model_name == "config-model"
        assert passed_config.model.model_max_length == 256
        assert passed_config.remote_params.api_url == "http://config.com"
        assert passed_config.remote_params.api_key == "config-key"
        assert passed_config.remote_params.num_workers == 2
        assert passed_config.remote_params.max_retries == 5


@pytest.mark.parametrize(
    "engine_class", LOCAL_ENGINES + REMOTE_ENGINES + REMOTE_API_ENGINES
)
def test_engine_config_partial_override(engine_class):
    """Test that InferenceConfig partially overrides constructor params."""
    # Initialize with full params
    init_model_params = ModelParams(
        model_name="init-model",
        model_max_length=128,
        torch_dtype_str="float32",
    )
    init_remote_params = (
        RemoteParams(
            api_url="http://init.com",
            api_key="init-key",
            num_workers=1,
            max_retries=3,
        )
        if engine_class not in LOCAL_ENGINES
        else None
    )

    mock_engine_class = _mock_engine(engine_class)
    with mock_engine_class:
        if issubclass(engine_class, RemoteInferenceEngine):
            engine = engine_class(
                model_params=init_model_params,
                remote_params=init_remote_params,
            )
        else:
            engine = engine_class(model_params=init_model_params)

    # Create config with only some params changed
    config_model_params = ModelParams(
        model_name="config-model",  # Changed
        model_max_length=128,  # Same as init
        torch_dtype_str="float32",  # Same as init
    )
    config_remote_params = (
        RemoteParams(
            api_url="http://config.com",  # Changed
            api_key="init-key",  # Same as init
            num_workers=1,  # Same as init
            max_retries=3,  # Same as init
        )
        if engine_class not in LOCAL_ENGINES
        else None
    )
    config = InferenceConfig(
        model=config_model_params,
        remote_params=config_remote_params,
        generation=GenerationParams(max_new_tokens=100),
    )

    # Mock _infer to avoid actual inference
    with patch.object(engine, "_infer") as mock_infer:
        engine.infer([], config)

        # Verify only changed params were overridden
        call_args = mock_infer.call_args[0]
        passed_config = call_args[1]
        assert passed_config.model.model_name == "config-model"  # Changed
        assert passed_config.model.model_max_length == 128  # Same
        assert passed_config.model.torch_dtype_str == "float32"  # Same

        if engine_class not in LOCAL_ENGINES:
            assert passed_config.remote_params.api_url == "http://config.com"  # Changed
            assert passed_config.remote_params.api_key == "init-key"  # Same
            assert passed_config.remote_params.num_workers == 1  # Same
            assert passed_config.remote_params.max_retries == 3  # Same


def test_all_inference_engine_types_in_engine_map():
    """Test that all InferenceEngineType values are present in ENGINE_MAP."""
    for engine_type in InferenceEngineType:
        assert engine_type in ENGINE_MAP, (
            f"Missing engine type {engine_type} in ENGINE_MAP"
        )


def test_build_all_inference_engines():
    """Test that all inference engines can be built using the builder."""
    model_params = ModelParams(model_name="test-model")
    remote_params = RemoteParams(api_url="http://test.com", api_key="test-key")

    for engine_type in InferenceEngineType:
        engine_class = ENGINE_MAP[engine_type]
        mock_ctx = _mock_engine(engine_class)

        with mock_ctx:
            # Build with appropriate params based on engine type
            if issubclass(engine_class, RemoteInferenceEngine):
                engine = build_inference_engine(
                    engine_type=engine_type,
                    model_params=model_params,
                    remote_params=remote_params,
                )
            else:
                engine = build_inference_engine(
                    engine_type=engine_type,
                    model_params=model_params,
                )

            # Verify the engine is of the correct type
            assert isinstance(engine, engine_class)
