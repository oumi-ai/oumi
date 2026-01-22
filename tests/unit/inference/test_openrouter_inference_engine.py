import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.openrouter_inference_engine import OpenRouterInferenceEngine


@pytest.fixture
def openrouter_engine():
    return OpenRouterInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_openrouter_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="anthropic/claude-3-opus")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = OpenRouterInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "anthropic/claude-3-opus"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_openrouter_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="openai/gpt-4")
    engine = OpenRouterInferenceEngine(model_params)
    assert engine._model_params.model_name == "openai/gpt-4"
    assert (
        engine._remote_params.api_url
        == "https://openrouter.ai/api/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "OPENROUTER_API_KEY"


def test_openrouter_convert_conversation_to_api_input(openrouter_engine):
    """Test conversion of conversation to API input format."""
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
        ]
    )
    generation_params = GenerationParams(
        max_new_tokens=100,
        temperature=0.7,
    )
    model_params = ModelParams(model_name="openai/gpt-4")

    api_input = openrouter_engine._convert_conversation_to_api_input(
        conversation, generation_params, model_params
    )

    assert api_input["model"] == "openai/gpt-4"
    assert len(api_input["messages"]) == 2
    assert api_input["messages"][0]["role"] == "system"
    assert api_input["messages"][1]["role"] == "user"
    assert api_input["max_completion_tokens"] == 100
    assert api_input["temperature"] == 0.7


def test_openrouter_convert_api_output_to_conversation(openrouter_engine):
    """Test conversion of API output to conversation."""
    original_conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello!"),
        ]
    )
    api_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openai/gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
        },
    }

    result = openrouter_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].role == Role.USER
    assert result.messages[0].content == "Hello!"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.messages[1].content == "Hi there! How can I help you today?"


def test_openrouter_supported_params(openrouter_engine):
    """Test that supported params include expected generation parameters."""
    supported = openrouter_engine.get_supported_params()

    assert "max_new_tokens" in supported
    assert "temperature" in supported
    assert "top_p" in supported
    assert "stop_strings" in supported
    assert "frequency_penalty" in supported
    assert "presence_penalty" in supported
