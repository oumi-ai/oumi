import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.anthropic_inference_engine import AnthropicInferenceEngine
from oumi.inference.remote_inference_engine import BatchInfo, BatchStatus


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


def test_convert_api_output_to_conversation_with_usage(anthropic_engine):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {
        "content": [{"text": "Assistant response"}],
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert result.metadata["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }
    assert result.metadata["key"] == "value"
    assert result.conversation_id == "test_id"


def test_convert_api_output_to_conversation_no_usage(anthropic_engine):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
    )
    api_response = {"content": [{"text": "Assistant response"}]}

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert "usage" not in result.metadata
    assert result.metadata["key"] == "value"


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


def _make_conversation(text: str) -> Conversation:
    return Conversation(messages=[Message(content=text, role=Role.USER)])


def _make_batch_info(
    status: BatchStatus,
    results_url: str = "https://example.com/results.jsonl",
    completed_requests: int = 0,
    failed_requests: int = 0,
) -> BatchInfo:
    return BatchInfo(
        id="batch-123",
        status=status,
        total_requests=completed_requests + failed_requests,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        metadata={"results_url": results_url} if results_url else None,
    )


def _mock_session_response(results_content: str):
    """Create a mock for _create_session that returns the given JSONL content."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value=results_content)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=(mock_session, {}))
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


@pytest.mark.asyncio
async def test_batch_results_partial_failed_status_retrieves_partial_results(
    anthropic_engine,
):
    """FAILED batches should log a warning and return partial results."""
    conversations = [_make_conversation("Q0"), _make_conversation("Q1")]

    batch_info = _make_batch_info(
        BatchStatus.FAILED, completed_requests=1, failed_requests=1
    )

    results_jsonl = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "request-0",
                    "result": {
                        "type": "succeeded",
                        "message": {"content": [{"text": "A0"}]},
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "request-1",
                    "result": {
                        "type": "errored",
                        "error": {"type": "server_error", "message": "overloaded"},
                    },
                }
            ),
        ]
    )

    with (
        patch.object(
            anthropic_engine,
            "_get_anthropic_batch_status",
            new_callable=AsyncMock,
            return_value=batch_info,
        ),
        patch.object(
            anthropic_engine,
            "_create_session",
            return_value=_mock_session_response(results_jsonl),
        ),
    ):
        result = await anthropic_engine._get_anthropic_batch_results_partial(
            "batch-123", conversations
        )

    assert len(result.successful) == 1
    assert result.successful[0][0] == 0
    assert result.failed_indices == [1]
    assert "server_error" in result.error_messages[1]
