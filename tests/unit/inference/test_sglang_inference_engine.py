from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference.sglang_inference_engine import SGLangInferenceEngine


@pytest.fixture
def sglang_vision_language_engine():
    return SGLangInferenceEngine(
        model_params=ModelParams(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype_str="bfloat16",
            model_max_length=1024,
            chat_template="llama3-instruct",
            trust_remote_code=True,
        )
    )


def test_convert_conversation_to_api_input(sglang_vision_language_engine):
    conversation = Conversation(
        messages=[
            Message(content="System message", role=Role.SYSTEM),
            Message(content="User message", role=Role.USER),
            Message(content="Assistant message", role=Role.ASSISTANT),
        ]
    )
    generation_params = GenerationParams(max_new_tokens=100)

    result = sglang_vision_language_engine._convert_conversation_to_api_input(
        conversation, generation_params
    )

    assert result["model"] == "claude-3"
    assert result["system"] == "System message"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == "User message"
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["content"] == "Assistant message"
    assert result["messages"][1]["role"] == "assistant"
    assert result["max_tokens"] == 100


def test_convert_api_output_to_conversation(sglang_vision_language_engine):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {"text": "Assistant response"}

    result = sglang_vision_language_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].content == "User message"
    assert result.messages[1].content == "Assistant response"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.messages[1].type == Type.TEXT
    assert result.metadata == {"key": "value"}
    assert result.conversation_id == "test_id"


def test_get_request_headers(sglang_vision_language_engine):
    remote_params = RemoteParams(api_key="test_api_key", api_url="<placeholder>")

    with patch.object(
        SGLangInferenceEngine,
        "_get_api_key",
        return_value="test_api_key",
    ):
        result = sglang_vision_language_engine._get_request_headers(remote_params)

    assert result["Content-Type"] == "application/json"
