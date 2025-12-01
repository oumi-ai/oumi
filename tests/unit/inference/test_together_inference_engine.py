import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.together_inference_engine import TogetherInferenceEngine


@pytest.fixture
def together_engine():
    return TogetherInferenceEngine(
        model_params=ModelParams(model_name="together-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_together_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="together-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = TogetherInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "together-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_together_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="together-model")
    engine = TogetherInferenceEngine(model_params)
    assert engine._model_params.model_name == "together-model"
    assert (
        engine._remote_params.api_url == "https://api.together.xyz/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "TOGETHER_API_KEY"


def test_together_convert_conversation_to_api_input():
    """Test conversion of conversation to API input."""
    # test p[assing in remote_params.api_kwargs.
    remote_params = RemoteParams(api_kwargs={"reasoning": {"enabled": True}})
    model_params = ModelParams(model_name="together-model")
    generation_params = GenerationParams(max_new_tokens=100)
    engine = TogetherInferenceEngine(
        model_params=model_params, remote_params=remote_params
    )
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
        ]
    )
    api_input = engine._convert_conversation_to_api_input(
        conversation, generation_params, model_params
    )
    assert "reasoning" in api_input
    assert api_input["reasoning"]["enabled"] is True


def test_together_convert_api_output_to_conversation():
    """Test conversion of API output to conversation."""
    remote_params = RemoteParams(api_kwargs={"reasoning": {"enabled": True}})
    model_params = ModelParams(model_name="together-model")
    engine = TogetherInferenceEngine(
        model_params=model_params, remote_params=remote_params
    )

    response = {
        "choices": [
            {
                "message": {
                    "content": "Hello, how are you?",
                    "reasoning": "I am thinking about the answer...",
                    "role": "assistant",
                }
            }
        ]
    }
    original_conversation = Conversation(
        messages=[Message(role=Role.USER, content="Hello, how are you?")]
    )
    conversation = engine._convert_api_output_to_conversation(
        response, original_conversation
    )
    assert (
        conversation.messages[-1].content
        == "<think>I am thinking about the answer...</think> Hello, how are you?"
    )
