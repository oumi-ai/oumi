import json
from unittest.mock import AsyncMock, patch

import pydantic
import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.gemini_inference_engine import GoogleGeminiInferenceEngine


@pytest.fixture
def gemini_engine():
    model_params = ModelParams(model_name="gemini-model")
    return GoogleGeminiInferenceEngine(
        model_params,
        remote_params=RemoteParams(
            api_url="https://example.com/api",
            api_key="dummy_api_key",
            num_workers=1,
            max_retries=3,
            connection_timeout=30,
            politeness_policy=0.1,
        ),
    )


@pytest.fixture
def generation_params():
    return GenerationParams(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def inference_config(generation_params):
    return InferenceConfig(
        generation=generation_params,
        remote_params=RemoteParams(
            api_url="https://example.com/api",
            api_key="dummy_api_key",
            num_workers=1,
            max_retries=3,
            connection_timeout=30,
            politeness_policy=0.1,
        ),
    )


def test_gemini_convert_conversation(gemini_engine, generation_params):
    """Test basic conversation conversion without special features."""
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ]
    )

    api_input = gemini_engine._convert_conversation_to_api_input(
        conversation, generation_params, gemini_engine._model_params
    )

    assert api_input["model"] == "gemini-model"
    assert len(api_input["messages"]) == 3
    assert api_input["max_completion_tokens"] == 100
    assert api_input["temperature"] == 0.7
    assert api_input["top_p"] == 0.9
    assert api_input["n"] == 1


def test_gemini_convert_conversation_with_guided_decoding(
    gemini_engine, generation_params
):
    """Test conversation conversion with JSON schema guided decoding."""

    class TestSchema(pydantic.BaseModel):
        name: str
        age: int

    generation_params.guided_decoding = GuidedDecodingParams(json=TestSchema)
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    api_input = gemini_engine._convert_conversation_to_api_input(
        conversation, generation_params, gemini_engine._model_params
    )

    assert "response_format" in api_input
    assert api_input["response_format"]["type"] == "json_schema"
    assert api_input["response_format"]["json_schema"]["name"] == "TestSchema"
    assert "properties" in api_input["response_format"]["json_schema"]["schema"]


@pytest.mark.parametrize(
    "json_schema",
    [
        {"type": "object", "properties": {"name": {"type": "string"}}},  # dict
        '{"type": "object", "properties": {"name": {"type": "string"}}}',  # str
    ],
)
def test_gemini_convert_conversation_with_json_schema_variations(
    gemini_engine, generation_params, json_schema
):
    """Test conversation conversion with different JSON schema formats."""
    generation_params.guided_decoding = GuidedDecodingParams(json=json_schema)
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    api_input = gemini_engine._convert_conversation_to_api_input(
        conversation, generation_params, gemini_engine._model_params
    )

    assert "response_format" in api_input
    assert api_input["response_format"]["type"] == "json_schema"
    assert api_input["response_format"]["json_schema"]["name"] == "Response"


def test_gemini_convert_conversation_invalid_schema(gemini_engine, generation_params):
    """Test that invalid schema types raise appropriate errors."""
    generation_params.guided_decoding = GuidedDecodingParams(json=123)  # Invalid type
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        gemini_engine._convert_conversation_to_api_input(
            conversation, generation_params, gemini_engine._model_params
        )

    assert "unsupported JSON schema type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_infer_online(gemini_engine, inference_config):
    """Test online inference with Gemini."""
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    with patch.object(gemini_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gemini_engine.infer_online([conversation], inference_config)

    assert len(results) == 1
    assert results[0] == conversation


def test_gemini_infer_from_file(gemini_engine, inference_config, tmp_path):
    """Test file-based inference with Gemini."""
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        json.dump(conversation.to_dict(), f)
        f.write("\n")

    with patch.object(gemini_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gemini_engine.infer_from_file(str(input_file), inference_config)

    assert len(results) == 1
    assert results[0] == conversation


def test_gemini_batch_prediction_disabled(gemini_engine, inference_config):
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    with pytest.raises(NotImplementedError):
        gemini_engine.infer_batch([conversation], inference_config)


def test_remote_params_defaults():
    gemini_engine = GoogleGeminiInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
    )
    assert gemini_engine._remote_params.num_workers == 2
    assert gemini_engine._remote_params.politeness_policy == 60.0
