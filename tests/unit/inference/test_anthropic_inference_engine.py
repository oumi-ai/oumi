from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.anthropic_inference_engine import AnthropicInferenceEngine


@pytest.fixture
def anthropic_engine():
    return AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_convert_conversation_to_api_input(anthropic_engine):
    conversation = Conversation(
        messages=[
            Message(content="System message", role=Role.SYSTEM),
            Message(content="User message", role=Role.USER),
            Message(content="Assistant message", role=Role.ASSISTANT),
        ]
    )
    generation_params = GenerationParams(max_new_tokens=100)

    result = anthropic_engine._convert_conversation_to_api_input(
        conversation, generation_params, anthropic_engine._model_params
    )

    assert result["model"] == "claude-3"
    assert result["system"] == "System message"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == "User message"
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["content"] == "Assistant message"
    assert result["messages"][1]["role"] == "assistant"
    assert result["max_tokens"] == 100


def test_convert_api_output_to_conversation(anthropic_engine):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {"content": [{"text": "Assistant response"}]}

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].content == "User message"
    assert result.messages[1].content == "Assistant response"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.metadata == {"key": "value"}
    assert result.conversation_id == "test_id"


def test_get_request_headers(anthropic_engine):
    remote_params = RemoteParams(api_key="test_api_key", api_url="<placeholder>")

    with patch.object(
        AnthropicInferenceEngine,
        "_get_api_key",
        return_value="test_api_key",
    ):
        result = anthropic_engine._get_request_headers(remote_params)

    assert result["Content-Type"] == "application/json"
    assert result["anthropic-version"] == AnthropicInferenceEngine.anthropic_version
    assert result["X-API-Key"] == "test_api_key"


def test_remote_params_defaults():
    anthropic_engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
    )
    assert anthropic_engine._remote_params.num_workers == 5
    assert anthropic_engine._remote_params.politeness_policy == 60.0
