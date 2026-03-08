import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.fireworks_inference_engine import FireworksInferenceEngine


@pytest.fixture
def fireworks_engine():
    return FireworksInferenceEngine(
        model_params=ModelParams(model_name="fireworks-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_fireworks_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="fireworks-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = FireworksInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "fireworks-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_fireworks_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="fireworks-model")
    engine = FireworksInferenceEngine(model_params)
    assert engine._model_params.model_name == "fireworks-model"
    assert (
        engine._remote_params.api_url
        == "https://api.fireworks.ai/inference/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "FIREWORKS_API_KEY"


def _make_fireworks_conversation(content: str) -> Conversation:
    return Conversation(messages=[Message(content=content, role=Role.USER)])


@pytest.mark.asyncio
async def test_fireworks_batch_partial_detects_missing_indices(fireworks_engine):
    """Indices absent from both results and error files appear in failed_indices."""
    # 3 conversations: index 0 succeeds, index 1 errors, index 2 is MISSING
    conversations = [
        _make_fireworks_conversation("Q0"),
        _make_fireworks_conversation("Q1"),
        _make_fireworks_conversation("Q2"),
    ]

    results_jsonl = json.dumps(
        {
            "custom_id": "request-0",
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "A0"}}],
            },
        }
    )
    errors_jsonl = json.dumps(
        {
            "custom_id": "request-1",
            "error": {"message": "rate limited"},
        }
    )

    mock_batch_info = MagicMock()
    mock_batch_info.is_terminal = True
    mock_batch_info.status = MagicMock()
    mock_batch_info.status.value = "completed"
    mock_batch_info.metadata = {"output_dataset_id": "accounts/x/datasets/ds-123"}

    with (
        patch.object(
            fireworks_engine,
            "_get_fireworks_batch_status",
            new_callable=AsyncMock,
            return_value=mock_batch_info,
        ),
        patch.object(
            fireworks_engine,
            "_extract_resource_id",
            return_value="ds-123",
        ),
        patch.object(
            fireworks_engine,
            "_get_fireworks_dataset_urls",
            new_callable=AsyncMock,
            return_value={
                "results.jsonl": "https://example.com/results.jsonl",
                "errors.jsonl": "https://example.com/errors.jsonl",
            },
        ),
        patch.object(
            fireworks_engine,
            "_download_fireworks_file",
            new_callable=AsyncMock,
            side_effect=[results_jsonl, errors_jsonl],
        ),
        patch.object(
            fireworks_engine,
            "_convert_api_output_to_conversation",
            return_value=Conversation(
                messages=[
                    Message(content="Q0", role=Role.USER),
                    Message(content="A0", role=Role.ASSISTANT),
                ]
            ),
        ),
    ):
        batch_result = await fireworks_engine._get_fireworks_batch_results_partial(
            "batch-abc", conversations
        )

    # Index 0 succeeded
    assert len(batch_result.successful) == 1
    assert batch_result.successful[0][0] == 0

    # Indices 1 and 2 failed (1 = error file, 2 = missing)
    assert batch_result.failed_indices == [1, 2]

    # Index 2 has the missing-from-output error message
    assert 2 in batch_result.error_messages
    assert "missing" in batch_result.error_messages[2].lower()

    # Index 1 has the rate limited error
    assert "rate limited" in batch_result.error_messages[1]
