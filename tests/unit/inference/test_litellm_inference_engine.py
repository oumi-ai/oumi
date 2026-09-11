import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

# These tests patch the module-level ``litellm`` symbol, so they run
# deterministically without the optional ``litellm`` package or any network.


@pytest.fixture
def mock_litellm():
    """Mock litellm to avoid actual API calls."""
    with patch("oumi.inference.litellm_inference_engine.litellm") as mock:
        yield mock


@pytest.fixture
def litellm_engine(mock_litellm):
    """Create LiteLLMInferenceEngine with mocked litellm."""
    model_params = ModelParams(model_name="anthropic/claude-sonnet-4-5")
    remote_params = RemoteParams(api_key="test_api_key")
    return LiteLLMInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )


def _openai_response(content="4", prompt_tokens=3, completion_tokens=2):
    """A minimal OpenAI-compatible response dict as litellm returns."""
    return {
        "id": "cmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _semaphore(engine):
    return PoliteAdaptiveSemaphore(
        capacity=1,
        politeness_policy=engine._remote_params.politeness_policy,
    )


def test_initialization(mock_litellm):
    model_params = ModelParams(
        model_name="anthropic/claude-sonnet-4-5",
        model_max_length=2048,
    )
    remote_params = RemoteParams(api_key="test-key")
    engine = LiteLLMInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )

    assert engine._model_params.model_name == "anthropic/claude-sonnet-4-5"
    assert engine._remote_params.api_key == "test-key"


def test_convert_conversation_to_api_input(litellm_engine):
    conversation = Conversation(
        messages=[
            Message(content="System message", role=Role.SYSTEM),
            Message(content="User message", role=Role.USER),
            Message(content="Assistant message", role=Role.ASSISTANT),
        ]
    )
    generation_params = GenerationParams(max_new_tokens=100, top_p=1.0, temperature=0.5)

    result = litellm_engine._convert_conversation_to_api_input(
        conversation, generation_params, litellm_engine._model_params
    )

    assert result["model"] == "anthropic/claude-sonnet-4-5"
    assert result["max_completion_tokens"] == 100
    assert result["temperature"] == 0.5
    assert result["top_p"] == 1.0
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"
    # Keys with a None value are dropped so providers rejecting nulls are safe.
    assert all(value is not None for value in result.values())


def test_guided_decoding_forwarded(litellm_engine):
    conversation = Conversation(
        messages=[Message(content="Give me JSON", role=Role.USER)]
    )
    generation_params = GenerationParams(
        max_new_tokens=50,
        guided_decoding=GuidedDecodingParams(
            json={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            }
        ),
    )

    result = litellm_engine._convert_conversation_to_api_input(
        conversation, generation_params, litellm_engine._model_params
    )

    # LiteLLM is OpenAI-compatible, so guided decoding is forwarded as
    # response_format rather than dropped.
    assert result["response_format"]["type"] == "json_schema"


def test_remote_params_forwarded_to_completion(litellm_engine):
    api_input = {"model": "anthropic/claude-sonnet-4-5", "messages": []}
    remote_params = RemoteParams(
        api_key="sk-proxy",
        api_url="https://proxy.internal:4000",
        connection_timeout=42.0,
    )

    litellm_engine._apply_remote_params_to_api_input(api_input, remote_params)

    assert api_input["api_key"] == "sk-proxy"
    assert api_input["api_base"] == "https://proxy.internal:4000"
    assert api_input["timeout"] == 42.0


def test_query_api_calls_completion_and_converts_response(litellm_engine, mock_litellm):
    mock_litellm.completion.return_value.model_dump.return_value = _openai_response("4")
    conversation = Conversation(messages=[Message(content="2+2?", role=Role.USER)])

    with patch.object(litellm_engine, "_save_conversation_to_scratch") as mock_save:
        result = asyncio.run(
            litellm_engine._query_api(
                conversation, _semaphore(litellm_engine), None, None
            )
        )

    assert mock_litellm.completion.call_count == 1
    # drop_params lets LiteLLM span providers with differing param support.
    assert mock_litellm.completion.call_args.kwargs.get("drop_params") is True
    assert result.messages[-1].role == Role.ASSISTANT
    assert result.messages[-1].content == "4"
    # Progress is persisted unconditionally to preserve resume/checkpoint.
    assert mock_save.call_count == 1


def test_query_api_retries_then_succeeds(litellm_engine, mock_litellm):
    good = MagicMock()
    good.model_dump.return_value = _openai_response("ok")
    # A non-RuntimeError is retriable; first attempt fails, second succeeds.
    mock_litellm.completion.side_effect = [ValueError("transient"), good]
    litellm_engine._remote_params.max_retries = 2
    litellm_engine._remote_params.retry_backoff_base = 0.0
    conversation = Conversation(messages=[Message(content="hi", role=Role.USER)])

    with patch.object(litellm_engine, "_save_conversation_to_scratch"):
        result = asyncio.run(
            litellm_engine._query_api(
                conversation, _semaphore(litellm_engine), None, None
            )
        )

    assert mock_litellm.completion.call_count == 2
    assert result.messages[-1].content == "ok"


def test_query_api_paces_and_records_with_rate_limiter(litellm_engine, mock_litellm):
    good = MagicMock()
    good.model_dump.return_value = _openai_response(
        "ok", prompt_tokens=3, completion_tokens=2
    )
    mock_litellm.completion.side_effect = [ValueError("transient"), good]

    limiter = MagicMock()
    limiter.wait_if_needed = AsyncMock()
    limiter.record_usage = AsyncMock()
    litellm_engine._rate_limiter = limiter
    litellm_engine._remote_params.max_retries = 1
    litellm_engine._remote_params.retry_backoff_base = 0.0
    conversation = Conversation(messages=[Message(content="hi", role=Role.USER)])

    with patch.object(litellm_engine, "_save_conversation_to_scratch"):
        asyncio.run(
            litellm_engine._query_api(
                conversation, _semaphore(litellm_engine), None, None
            )
        )

    # Rate limiter is consulted before every attempt (both the failed and the
    # successful one), mirroring RemoteInferenceEngine.
    assert limiter.wait_if_needed.await_count == 2
    limiter.record_usage.assert_awaited_once_with(input_tokens=3, output_tokens=2)


def test_infer_online(litellm_engine):
    with patch.object(litellm_engine, "_infer") as mock_infer:
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

        result = litellm_engine.infer(input_conversations, inference_config)

        mock_infer.assert_called_once_with(input_conversations, inference_config)
        assert result == mock_infer.return_value


def test_get_supported_params(litellm_engine):
    params = litellm_engine.get_supported_params()
    assert "max_new_tokens" in params
    assert "temperature" in params
    assert "top_p" in params
    assert "stop_strings" in params
    assert "seed" in params
    assert "frequency_penalty" in params
    assert "presence_penalty" in params
    # Option B: LiteLLM forwards OpenAI-compatible tool and structured-output args.
    assert "guided_decoding" in params
    assert "tool_choice" in params


def test_batch_methods_not_implemented(litellm_engine):
    with pytest.raises(NotImplementedError):
        litellm_engine.infer_batch([])

    with pytest.raises(NotImplementedError):
        litellm_engine.get_batch_status("batch-id")

    with pytest.raises(NotImplementedError):
        litellm_engine.list_batches()

    with pytest.raises(NotImplementedError):
        litellm_engine.get_batch_results("batch-id", [])


def test_base_url_is_none(litellm_engine):
    assert litellm_engine.base_url is None


def test_api_key_env_varname_is_none(litellm_engine):
    assert litellm_engine.api_key_env_varname is None


def test_default_remote_params(litellm_engine):
    params = litellm_engine._default_remote_params()
    assert isinstance(params, RemoteParams)
