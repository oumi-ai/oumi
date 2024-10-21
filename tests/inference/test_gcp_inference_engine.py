import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientSession

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference.gcp_inference_engine import GCPInferenceEngine


@pytest.fixture
def gcp_engine():
    model_params = ModelParams(model_name="gcp-model")
    return GCPInferenceEngine(model_params)


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


def test_convert_conversation_to_api_input(gcp_engine, conversation, generation_params):
    api_input = gcp_engine._convert_conversation_to_api_input(
        conversation, generation_params
    )
    assert api_input["model"] == "gcp-model"
    assert len(api_input["messages"]) == 3
    assert api_input["max_completion_tokens"] == 100
    assert api_input["temperature"] == 0.7
    assert api_input["top_p"] == 0.9


@pytest.mark.asyncio
async def test_query_api(gcp_engine, conversation, generation_params, remote_params):
    mock_response = {
        "predictions": [
            {"candidates": [{"content": "I'm doing well, thank you for asking!"}]}
        ]
    }

    mock_session = MagicMock(spec=ClientSession)
    mock_post = mock_session.return_value.__aenter__.return_value.post
    mock_post.return_value.__aenter__.return_value.json = AsyncMock(
        return_value=mock_response
    )
    mock_post.return_value.__aenter__.return_value.status = 200

    mock_semaphore = AsyncMock()

    result = await gcp_engine._query_api(
        conversation, generation_params, remote_params, mock_semaphore, mock_session
    )

    assert isinstance(result, Conversation)
    assert len(result.messages) == 4
    assert result.messages[-1].content == "I'm doing well, thank you for asking!"
    assert result.messages[-1].role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_infer(gcp_engine, conversation, generation_params, remote_params):
    mock_response = {
        "predictions": [
            {"candidates": [{"content": "I'm doing well, thank you for asking!"}]}
        ]
    }

    with patch("aiohttp.ClientSession") as mock_session:
        mock_post = mock_session.return_value.__aenter__.return_value.post
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value=mock_response
        )
        mock_post.return_value.__aenter__.return_value.status = 200

        results = await gcp_engine._infer(
            [conversation], generation_params, remote_params
        )

    assert len(results) == 1
    assert isinstance(results[0], Conversation)
    assert len(results[0].messages) == 4
    assert results[0].messages[-1].content == "I'm doing well, thank you for asking!"
    assert results[0].messages[-1].role == Role.ASSISTANT


def test_infer_online(gcp_engine, conversation, generation_params):
    with patch.object(gcp_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gcp_engine.infer_online([conversation], generation_params)

    assert len(results) == 1
    assert results[0] == conversation


def test_infer_from_file(gcp_engine, conversation, generation_params, tmp_path):
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        json.dump(conversation.to_dict(), f)
        f.write("\n")

    with patch.object(gcp_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gcp_engine.infer_from_file(str(input_file), generation_params)

    assert len(results) == 1
    assert results[0] == conversation
