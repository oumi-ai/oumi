from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.sambanova_inference_engine import SambanovaInferenceEngine


@pytest.fixture
def sambanova_engine():
    return SambanovaInferenceEngine(
        model_params=ModelParams(model_name="Meta-Llama-3.1-8B-Instruct"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_convert_conversation_to_api_input(sambanova_engine):
    """Test conversion of conversation to SambaNova API input format."""
    conversation = Conversation(
        messages=[
            Message(content="System message", role=Role.SYSTEM),
            Message(content="User message", role=Role.USER),
            Message(content="Assistant message", role=Role.ASSISTANT),
        ]
    )
    generation_params = GenerationParams(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop_strings=["stop"],
    )

    result = sambanova_engine._convert_conversation_to_api_input(
        conversation, generation_params, sambanova_engine._model_params
    )

    # Verify the API input format
    assert result["model"] == "Meta-Llama-3.1-8B-Instruct"
    assert len(result["messages"]) == 3
    assert result["messages"][0]["content"] == "System message"
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["content"] == "User message"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["content"] == "Assistant message"
    assert result["messages"][2]["role"] == "assistant"
    assert result["max_tokens"] == 100
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
    assert result["stop"] == ["stop"]
    assert result["stream"] is False


def test_convert_api_output_to_conversation(sambanova_engine):
    """Test conversion of SambaNova API output to conversation."""
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {
        "choices": [{"message": {"content": "Assistant response", "role": "assistant"}}]
    }

    result = sambanova_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].content == "User message"
    assert result.messages[1].content == "Assistant response"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.metadata == {"key": "value"}
    assert result.conversation_id == "test_id"


def test_convert_api_output_to_conversation_error_handling(sambanova_engine):
    """Test error handling in API output conversion."""
    original_conversation = Conversation(
        messages=[Message(content="User message", role=Role.USER)]
    )

    # Test empty choices
    with pytest.raises(RuntimeError, match="No choices found in API response"):
        sambanova_engine._convert_api_output_to_conversation(
            {"choices": []}, original_conversation
        )

    # Test missing message
    with pytest.raises(RuntimeError, match="No message found in API response"):
        sambanova_engine._convert_api_output_to_conversation(
            {"choices": [{}]}, original_conversation
        )


def test_get_request_headers(sambanova_engine):
    """Test generation of request headers."""
    remote_params = RemoteParams(api_key="test_api_key", api_url="<placeholder>")

    with patch.object(
        SambanovaInferenceEngine,
        "_get_api_key",
        return_value="test_api_key",
    ):
        result = sambanova_engine._get_request_headers(remote_params)

    assert result["Content-Type"] == "application/json"
    assert result["Authorization"] == "Bearer test_api_key"


def test_get_supported_params(sambanova_engine):
    """Test supported generation parameters."""
    supported_params = sambanova_engine.get_supported_params()

    assert supported_params == {
        "max_new_tokens",
        "stop_strings",
        "temperature",
        "top_p",
    }
