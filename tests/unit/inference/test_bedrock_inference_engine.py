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
from oumi.inference.bedrock_inference_engine import BedrockInferenceEngine

boto3_import_failed = find_spec("boto3") is None


@pytest.fixture
def mock_boto3():
    """Mock boto3 client to avoid actual AWS calls."""
    with patch("oumi.inference.bedrock_inference_engine.boto3") as mock:
        yield mock


@pytest.fixture
def bedrock_engine(mock_boto3):
    """Create BedrockInferenceEngine with mocked boto3."""
    model_params = ModelParams(model_name="claude-3")
    remote_params = RemoteParams(api_key="test_api_key")
    return BedrockInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )


@pytest.mark.skipif(boto3_import_failed, reason="boto3 not available")
def test_initialization(mock_boto3):
    model_params = ModelParams(
        model_name="claude-3",
        model_max_length=2048,
    )
    remote_params = RemoteParams(api_key="test-key")
    engine = BedrockInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )

    assert engine._model_params.model_name == "claude-3"
    assert engine._remote_params.api_key == "test-key"


@pytest.mark.skipif(boto3_import_failed, reason="boto3 not available")
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


@pytest.mark.skipif(boto3_import_failed, reason="boto3 not available")
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


@pytest.mark.skipif(boto3_import_failed, reason="boto3 not available")
def test_infer_online(bedrock_engine):
    with patch.object(bedrock_engine, "_infer") as mock_infer:
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

        result = bedrock_engine.infer(input_conversations, inference_config)

        mock_infer.assert_called_once_with(input_conversations, inference_config)
        assert result == mock_infer.return_value
