import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.bedrock_inference_engine import BedrockInferenceEngine


@pytest.fixture
def bedrock_engine():
    return BedrockInferenceEngine(
        model_params=ModelParams(model_name="claude-3"),
        remote_params=RemoteParams(api_key="test_api_key"),
    )


def test_convert_conversation_to_api_input(bedrock_engine):
    conversation = Conversation(
        messages=[
            Message(content="System message", role=Role.SYSTEM),
            Message(content="User message", role=Role.USER),
            Message(content="Assistant message", role=Role.ASSISTANT),
        ]
    )
    generation_params = GenerationParams(max_new_tokens=100)

    result = bedrock_engine._convert_conversation_to_api_input(
        conversation, generation_params, bedrock_engine._model_params
    )

    print(result)

    assert result["inferenceConfig"]["maxTokens"] == 100
    assert result["inferenceConfig"]["temperature"] == 0.0
    assert result["inferenceConfig"]["topP"] == 1.0
    assert result["messages"][0]["content"][0]["text"] == "User message"
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["content"][0]["text"] == "Assistant message"
    assert result["messages"][1]["role"] == "assistant"
    assert result["system"][0]["text"] == "System message"


def test_convert_api_output_to_conversation(bedrock_engine):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {
        "output": {"message": {"content": [{"text": "Assistant response"}]}}
    }

    result = bedrock_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].content == "User message"
    assert result.messages[1].content == "Assistant response"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.metadata == {"key": "value"}
    assert result.conversation_id == "test_id"
