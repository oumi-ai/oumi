from importlib.util import find_spec
from unittest.mock import patch

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

litellm_import_failed = find_spec("litellm") is None


@pytest.fixture
def mock_litellm():
    """Mock litellm to avoid actual API calls."""
    with patch("oumi.inference.litellm_inference_engine.litellm") as mock:
        yield mock


@pytest.fixture
def litellm_engine(mock_litellm):
    """Create LiteLLMInferenceEngine with mocked litellm."""
    model_params = ModelParams(model_name="anthropic/claude-sonnet-4-5")
    remote_params = RemoteParams(api_key="test_api_key")
    return LiteLLMInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_initialization(mock_litellm):
    model_params = ModelParams(
        model_name="anthropic/claude-sonnet-4-5",
        model_max_length=2048,
    )
    remote_params = RemoteParams(api_key="test-key")
    engine = LiteLLMInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )

    assert engine._model_params.model_name == "anthropic/claude-sonnet-4-5"
    assert engine._remote_params.api_key == "test-key"


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_convert_conversation_to_api_input(litellm_engine):
    conversation = Conversation(
        messages=[
            Message(content="System message", role=Role.SYSTEM),
            Message(content="User message", role=Role.USER),
            Message(content="Assistant message", role=Role.ASSISTANT),
        ]
    )
    generation_params = GenerationParams(max_new_tokens=100, top_p=1.0, temperature=0.5)

    result = litellm_engine._convert_conversation_to_api_input(
        conversation, generation_params, litellm_engine._model_params
    )

    assert result["model"] == "anthropic/claude-sonnet-4-5"
    assert result["max_completion_tokens"] == 100
    assert result["temperature"] == 0.5
    assert result["top_p"] == 1.0
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_infer_online(litellm_engine):
    with patch.object(litellm_engine, "_infer") as mock_infer:
        mock_infer.return_value = [
            Conversation(
                conversation_id="1",
                messages=[Message(content="Response", role=Role.ASSISTANT)],
            )
        ]

        input_conversations = [
            Conversation(
                conversation_id="1",
                messages=[Message(content="Hello", role=Role.USER)],
            )
        ]
        inference_config = InferenceConfig(
            generation=GenerationParams(max_new_tokens=50),
        )

        result = litellm_engine.infer(input_conversations, inference_config)

        mock_infer.assert_called_once_with(input_conversations, inference_config)
        assert result == mock_infer.return_value


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_get_supported_params(litellm_engine):
    params = litellm_engine.get_supported_params()
    assert "max_new_tokens" in params
    assert "temperature" in params
    assert "top_p" in params
    assert "stop_strings" in params
    assert "seed" in params
    assert "frequency_penalty" in params
    assert "presence_penalty" in params


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_batch_methods_not_implemented(litellm_engine):
    with pytest.raises(NotImplementedError):
        litellm_engine.infer_batch([])

    with pytest.raises(NotImplementedError):
        litellm_engine.get_batch_status("batch-id")

    with pytest.raises(NotImplementedError):
        litellm_engine.list_batches()

    with pytest.raises(NotImplementedError):
        litellm_engine.get_batch_results("batch-id", [])


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_base_url_is_none(litellm_engine):
    assert litellm_engine.base_url is None


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_api_key_env_varname_is_none(litellm_engine):
    assert litellm_engine.api_key_env_varname is None


@pytest.mark.skipif(litellm_import_failed, reason="litellm not available")
def test_default_remote_params(litellm_engine):
    params = litellm_engine._default_remote_params()
    assert isinstance(params, RemoteParams)
