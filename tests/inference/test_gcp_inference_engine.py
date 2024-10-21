import json
from unittest.mock import AsyncMock, patch

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference.gcp_inference_engine import GoogleVertexInferenceEngine


@pytest.fixture
def gcp_engine():
    model_params = ModelParams(model_name="gcp-model")
    return GoogleVertexInferenceEngine(model_params)


@pytest.fixture
def remote_params():
    return RemoteParams(
        api_url="https://example.com/api",
        api_key="path/to/service_account.json",
        num_workers=1,
        max_retries=3,
        connection_timeout=30,
        politeness_policy=0.1,
    )


@pytest.fixture
def generation_params(remote_params):
    return GenerationParams(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        remote_params=remote_params,
    )


@pytest.fixture
def inference_config(generation_params):
    return InferenceConfig(
        generation=generation_params,
    )


@pytest.fixture
def conversation():
    return Conversation(
        messages=[
            Message(content="Hello", role=Role.USER, type=Type.TEXT),
            Message(content="Hi there!", role=Role.ASSISTANT, type=Type.TEXT),
            Message(content="How are you?", role=Role.USER, type=Type.TEXT),
        ]
    )


def test_get_api_key(gcp_engine, remote_params):
    with patch("google.oauth2.service_account.Credentials") as mock_credentials:
        mock_credentials.from_service_account_file.return_value.token = "fake_token"
        token = gcp_engine._get_api_key(remote_params)
        assert token == "fake_token"
        mock_credentials.from_service_account_file.assert_called_once_with(
            filename="path/to/service_account.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )


def test_get_request_headers(gcp_engine, remote_params):
    with patch.object(gcp_engine, "_get_api_key", return_value="fake_token"):
        headers = gcp_engine._get_request_headers(remote_params)
        assert headers == {
            "Authorization": "Bearer fake_token",
            "Content-Type": "application/json",
        }


def test_convert_conversation_to_api_input(gcp_engine, conversation, inference_config):
    api_input = gcp_engine._convert_conversation_to_api_input(
        conversation, inference_config.generation
    )
    assert api_input["model"] == "gcp-model"
    assert len(api_input["messages"]) == 3
    assert api_input["max_completion_tokens"] == 100
    assert api_input["temperature"] == 0.7
    assert api_input["top_p"] == 0.9


def test_infer_online(gcp_engine, conversation, inference_config):
    with patch.object(gcp_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gcp_engine.infer_online([conversation], inference_config)

    assert len(results) == 1
    assert results[0] == conversation


def test_infer_from_file(gcp_engine, conversation, inference_config, tmp_path):
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        json.dump(conversation.to_dict(), f)
        f.write("\n")

    with patch.object(gcp_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gcp_engine.infer_from_file(str(input_file), inference_config)

    assert len(results) == 1
    assert results[0] == conversation
