import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    FinishReason,
    Message,
    Role,
    Type,
)
from oumi.core.types.tool_call import ToolCall, ToolDefinition
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
    assert result["cache_control"] == {"type": "ephemeral"}


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


def test_convert_api_output_empty_content(anthropic_engine):
    original_conversation = Conversation(
        messages=[Message(content="User message", role=Role.USER)],
    )
    api_response = {
        "content": [],
        "stop_reason": "end_turn",
        "type": "message",
        "model": "claude-opus-4-6",
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }

    with pytest.raises(RuntimeError, match="Anthropic API returned empty content"):
        anthropic_engine._convert_api_output_to_conversation(
            api_response, original_conversation
        )


def test_convert_api_output_missing_content_key(anthropic_engine):
    original_conversation = Conversation(
        messages=[Message(content="User message", role=Role.USER)],
    )
    api_response = {
        "stop_reason": "end_turn",
        "type": "message",
    }

    with pytest.raises(RuntimeError, match="Anthropic API returned empty content"):
        anthropic_engine._convert_api_output_to_conversation(
            api_response, original_conversation
        )


@pytest.mark.parametrize(
    "api_usage,expected_usage",
    [
        # Basic usage
        (
            {"input_tokens": 12, "output_tokens": 8},
            {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        ),
        # With cache read tokens
        (
            {"input_tokens": 12, "output_tokens": 8, "cache_read_input_tokens": 5},
            {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "cached_tokens": 5,
            },
        ),
        # With cache read + creation tokens
        (
            {
                "input_tokens": 12,
                "output_tokens": 8,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 3,
            },
            {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "cached_tokens": 5,
                "cache_creation_tokens": 3,
            },
        ),
    ],
)
def test_convert_api_output_to_conversation_with_usage(
    anthropic_engine, api_usage, expected_usage
):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {
        "content": [{"text": "Assistant response"}],
        "usage": api_usage,
    }

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert result.metadata["usage"] == expected_usage
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


@pytest.mark.asyncio
async def test_batch_results_nested_error_structure(anthropic_engine):
    """Anthropic may nest error detail under error.error; extract inner message."""
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
                        "error": {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "max_tokens: 8096 > 8000",
                            },
                        },
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
    assert result.failed_indices == [1]
    assert "invalid_request_error" in result.error_messages[1]
    assert "max_tokens: 8096 > 8000" in result.error_messages[1]


# FinishReason extraction tests
class TestAnthropicExtractFinishReason:
    """Tests for AnthropicInferenceEngine._extract_finish_reason_from_response."""

    def test_extract_finish_reason_end_turn(self, anthropic_engine):
        response = {"stop_reason": "end_turn", "content": [{"text": "Hello"}]}
        result = AnthropicInferenceEngine._extract_finish_reason_from_response(response)
        assert result == FinishReason.STOP

    def test_extract_finish_reason_max_tokens(self, anthropic_engine):
        response = {"stop_reason": "max_tokens", "content": [{"text": "Truncated"}]}
        result = AnthropicInferenceEngine._extract_finish_reason_from_response(response)
        assert result == FinishReason.LENGTH

    def test_extract_finish_reason_stop_sequence(self, anthropic_engine):
        response = {"stop_reason": "stop_sequence", "content": [{"text": "Stopped"}]}
        result = AnthropicInferenceEngine._extract_finish_reason_from_response(response)
        assert result == FinishReason.STOP

    def test_extract_finish_reason_tool_use(self, anthropic_engine):
        response = {"stop_reason": "tool_use", "content": [{"text": "Function call"}]}
        result = AnthropicInferenceEngine._extract_finish_reason_from_response(response)
        assert result == FinishReason.TOOL_CALLS

    def test_extract_finish_reason_unknown(self, anthropic_engine):
        response = {"stop_reason": "some_new_reason", "content": [{"text": "Response"}]}
        result = AnthropicInferenceEngine._extract_finish_reason_from_response(response)
        assert result == FinishReason.UNKNOWN

    def test_extract_finish_reason_none(self, anthropic_engine):
        response = {"content": [{"text": "Response"}]}
        result = AnthropicInferenceEngine._extract_finish_reason_from_response(response)
        assert result is None


def test_convert_api_output_to_conversation_with_finish_reason(anthropic_engine):
    """Test that finish_reason is extracted from Anthropic response."""
    original_conversation = Conversation(
        messages=[Message(content="User message", role=Role.USER)],
        metadata={"key": "value"},
    )
    api_response = {
        "content": [{"text": "Assistant response"}],
        "stop_reason": "end_turn",
    }

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert result.metadata.get("finish_reason") == "stop"
    assert result.metadata["key"] == "value"


def test_convert_api_output_to_conversation_with_max_tokens_finish_reason(
    anthropic_engine,
):
    """Test that max_tokens finish_reason is mapped to 'length'."""
    original_conversation = Conversation(
        messages=[Message(content="User message", role=Role.USER)],
        metadata={},
    )
    api_response = {
        "content": [{"text": "Truncated response"}],
        "stop_reason": "max_tokens",
    }

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert result.metadata.get("finish_reason") == "length"


def test_convert_api_output_to_conversation_with_usage_and_finish_reason(
    anthropic_engine,
):
    """Test that both usage and finish_reason are extracted."""
    original_conversation = Conversation(
        messages=[Message(content="User message", role=Role.USER)],
        metadata={},
    )
    api_response = {
        "content": [{"text": "Response"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert result.metadata.get("finish_reason") == "stop"
    assert result.metadata["usage"]["prompt_tokens"] == 10
    assert result.metadata["usage"]["completion_tokens"] == 5


#
# Tool-calling tests
#
_WEATHER_TOOL = ToolDefinition.model_validate(
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
)

_CALENDAR_TOOL = ToolDefinition.model_validate(
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    }
)


def _build_body(engine, conversation, **gen_overrides):
    generation_params = GenerationParams(max_new_tokens=100, **gen_overrides)
    return engine._convert_conversation_to_api_input(
        conversation, generation_params, engine._model_params
    )


def test_openai_tools_translated_to_anthropic_schema(anthropic_engine):
    """``parameters`` becomes ``input_schema``; the ``function`` wrapper drops."""
    conversation = Conversation(
        tools=[_WEATHER_TOOL, _CALENDAR_TOOL],
        messages=[Message(role=Role.USER, content="hi")],
    )
    body = _build_body(anthropic_engine, conversation)
    assert "tools" in body
    assert body["tools"][0]["name"] == "get_weather"
    assert body["tools"][0]["description"] == "Get current weather for a city."
    assert body["tools"][0]["input_schema"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert "function" not in body["tools"][0]
    assert "parameters" not in body["tools"][0]


def test_last_tool_has_cache_control_marker(anthropic_engine):
    """Cache marker stamps only the last tool entry; nothing on the others."""
    conversation = Conversation(
        tools=[_WEATHER_TOOL, _CALENDAR_TOOL],
        messages=[Message(role=Role.USER, content="hi")],
    )
    body = _build_body(anthropic_engine, conversation)
    assert "cache_control" not in body["tools"][0]
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_assistant_tool_calls_emit_tool_use_blocks(anthropic_engine):
    """Assistant ``tool_calls`` become ``tool_use`` blocks with parsed input."""
    tool_call_dict = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
    }
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="weather?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall.model_validate(tool_call_dict)],
            ),
            Message(role=Role.TOOL, content="22C", tool_call_id="call_abc"),
        ],
    )
    body = _build_body(anthropic_engine, conversation)
    assistant_msg = body["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == [
        {
            "type": "tool_use",
            "id": "call_abc",
            "name": "get_weather",
            # Anthropic expects parsed input, not the JSON-string OpenAI sends.
            "input": {"city": "Tokyo"},
        }
    ]


def test_tool_role_string_emits_user_tool_result(anthropic_engine):
    """Role.TOOL with a string body emits a user/tool_result with string content."""
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="weather?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Tokyo"}',
                            },
                        }
                    )
                ],
            ),
            Message(role=Role.TOOL, content="22C, sunny", tool_call_id="call_abc"),
        ],
    )
    body = _build_body(anthropic_engine, conversation)
    tool_msg = body["messages"][2]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "call_abc",
            "content": "22C, sunny",
        }
    ]


def test_tool_role_multimodal_emits_blocks_in_tool_result(anthropic_engine):
    """Multimodal Role.TOOL content emits a list of text/image blocks."""
    png_bytes = b"\x89PNG\r\n\x1a\nfakeimg"
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="map?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{}",
                            },
                        }
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                tool_call_id="call_xyz",
                content=[
                    ContentItem(type=Type.TEXT, content="See attached:"),
                    ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes),
                ],
            ),
        ],
    )
    body = _build_body(anthropic_engine, conversation)
    tool_msg = body["messages"][2]
    assert tool_msg["role"] == "user"
    blocks = tool_msg["content"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "See attached:"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(png_bytes).decode("ascii"),
    }


@pytest.mark.parametrize(
    "magic_bytes,expected_media_type",
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
def test_image_media_type_sniffed_from_magic_bytes(
    anthropic_engine, magic_bytes, expected_media_type
):
    """media_type reflects the actual image format, not a hardcoded image/png."""
    image_bytes = magic_bytes + b"...rest of payload..."
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="map?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                tool_call_id="call_1",
                content=[ContentItem(type=Type.IMAGE_BINARY, binary=image_bytes)],
            ),
        ],
    )
    body = _build_body(anthropic_engine, conversation)
    image_block = body["messages"][2]["content"][0]["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == expected_media_type
    assert image_block["source"]["data"] == base64.b64encode(image_bytes).decode(
        "ascii"
    )


def test_unrecognized_image_bytes_fall_back_to_png(anthropic_engine, caplog):
    """Unknown formats default to image/png and emit a warning."""
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="map?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                tool_call_id="call_1",
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"unknown-format-bytes")
                ],
            ),
        ],
    )
    with caplog.at_level("WARNING"):
        body = _build_body(anthropic_engine, conversation)
    image_block = body["messages"][2]["content"][0]["content"][0]
    assert image_block["source"]["media_type"] == "image/png"
    assert any("Unrecognized image format" in rec.message for rec in caplog.records)


def test_response_tool_use_blocks_populate_tool_calls(anthropic_engine):
    """tool_use blocks in the response become OpenAI-format Message.tool_calls."""
    api_response = {
        "content": [
            {"type": "text", "text": "Looking up weather."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            },
        ],
        "stop_reason": "tool_use",
    }
    original = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[Message(role=Role.USER, content="weather?")],
    )
    result = anthropic_engine._convert_api_output_to_conversation(
        api_response, original
    )
    assistant = result.messages[-1]
    assert assistant.role == Role.ASSISTANT
    assert assistant.content == "Looking up weather."
    assert assistant.tool_calls is not None
    assert [tc.model_dump(mode="json") for tc in assistant.tool_calls] == [
        {
            "id": "toolu_01",
            "type": "function",
            "function": {
                "name": "get_weather",
                # OpenAI keeps arguments as a JSON-encoded string.
                "arguments": json.dumps({"city": "Tokyo"}),
            },
        }
    ]
    assert result.metadata["finish_reason"] == "tool_calls"
    assert result.tools == [_WEATHER_TOOL]


def test_tool_choice_translation(anthropic_engine):
    """tool_choice translates: auto/required/by-name; 'none' drops tools."""
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[Message(role=Role.USER, content="hi")],
    )

    body_auto = _build_body(anthropic_engine, conversation, tool_choice="auto")
    assert body_auto["tool_choice"] == {"type": "auto"}

    body_required = _build_body(anthropic_engine, conversation, tool_choice="required")
    assert body_required["tool_choice"] == {"type": "any"}

    body_named = _build_body(
        anthropic_engine,
        conversation,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert body_named["tool_choice"] == {"type": "tool", "name": "get_weather"}

    body_none = _build_body(anthropic_engine, conversation, tool_choice="none")
    assert "tools" not in body_none
    assert "tool_choice" not in body_none


def test_pure_text_conversation_emits_string_content(anthropic_engine):
    """No tool fields => request body uses primitive-string content shape."""
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi!"),
            Message(role=Role.USER, content="How are you?"),
        ]
    )
    body = _build_body(anthropic_engine, conversation)
    assert "tools" not in body
    assert "tool_choice" not in body
    # Single text-only turns collapse to primitive-string content rather than
    # wrapping in a [{"type": "text", ...}] block list.
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert body["messages"][1] == {"role": "assistant", "content": "Hi!"}
    assert body["messages"][2] == {"role": "user", "content": "How are you?"}


def test_adjacent_same_role_messages_are_merged(anthropic_engine):
    """Consecutive same-role messages merge into one turn (Anthropic alternation)."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="First user line."),
            Message(role=Role.USER, content="Second user line."),
            Message(role=Role.ASSISTANT, content="Reply."),
        ]
    )
    body = _build_body(anthropic_engine, conversation)
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "First user line."},
        {"type": "text", "text": "Second user line."},
    ]
    assert body["messages"][1] == {"role": "assistant", "content": "Reply."}


def test_tool_role_merges_into_adjacent_user_turn(anthropic_engine):
    """Role.TOOL collapses to a user turn and merges with adjacent user content."""
    conversation = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="weather?"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Tokyo"}',
                            },
                        }
                    )
                ],
            ),
            Message(role=Role.TOOL, content="22C, sunny", tool_call_id="call_1"),
            Message(role=Role.USER, content="And tomorrow?"),
        ],
    )
    body = _build_body(anthropic_engine, conversation)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    merged_user_turn = body["messages"][2]["content"]
    assert merged_user_turn[0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "22C, sunny",
    }
    assert merged_user_turn[1] == {"type": "text", "text": "And tomorrow?"}
