# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for streaming inference functionality."""

import asyncio
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.streaming import StreamingChunk, StreamingChunkType
from oumi.inference import RemoteInferenceEngine

_TARGET_SERVER: Final[str] = "http://fakeurl/v1/chat/completions"


#
# Fixtures
#
@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="test-model",
        trust_remote_code=True,
    )


def _get_default_remote_params() -> RemoteParams:
    return RemoteParams(
        api_url=_TARGET_SERVER,
        api_key="test-api-key",
    )


def _create_test_conversation() -> Conversation:
    return Conversation(
        conversation_id="test-conv-1",
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(content="Hello, how are you?", role=Role.USER),
        ],
    )


#
# StreamingChunk Tests
#
class TestStreamingChunk:
    """Tests for StreamingChunk dataclass."""

    def test_content_delta_chunk(self):
        chunk = StreamingChunk(
            chunk_type=StreamingChunkType.CONTENT_DELTA,
            delta="Hello",
            accumulated_content="Hello",
            conversation_id="test-1",
        )
        assert chunk.chunk_type == StreamingChunkType.CONTENT_DELTA
        assert chunk.delta == "Hello"
        assert chunk.accumulated_content == "Hello"
        assert not chunk.is_final

    def test_finish_chunk(self):
        chunk = StreamingChunk(
            chunk_type=StreamingChunkType.FINISH,
            finish_reason="stop",
            accumulated_content="Hello World",
            conversation_id="test-1",
        )
        assert chunk.chunk_type == StreamingChunkType.FINISH
        assert chunk.finish_reason == "stop"
        assert chunk.is_final

    def test_error_chunk(self):
        chunk = StreamingChunk(
            chunk_type=StreamingChunkType.ERROR,
            error_message="Connection failed",
            conversation_id="test-1",
        )
        assert chunk.chunk_type == StreamingChunkType.ERROR
        assert chunk.error_message == "Connection failed"
        assert chunk.is_final

    def test_is_final_property(self):
        # CONTENT_DELTA is not final
        chunk1 = StreamingChunk(chunk_type=StreamingChunkType.CONTENT_DELTA)
        assert not chunk1.is_final

        # FINISH is final
        chunk2 = StreamingChunk(chunk_type=StreamingChunkType.FINISH)
        assert chunk2.is_final

        # ERROR is final
        chunk3 = StreamingChunk(chunk_type=StreamingChunkType.ERROR)
        assert chunk3.is_final


#
# RemoteInferenceEngine Streaming Tests
#
class TestRemoteInferenceEngineStreaming:
    """Tests for RemoteInferenceEngine streaming functionality."""

    def test_supports_streaming(self):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )
        assert engine.supports_streaming()

    @pytest.mark.asyncio
    async def test_infer_stream_success(self, mock_aioresponse):
        """Test successful streaming response."""
        # SSE response simulating OpenAI streaming format
        sse_response = (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" World"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        mock_aioresponse.post(
            _TARGET_SERVER,
            body=sse_response,
            status=200,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        conversation = _create_test_conversation()
        chunks = []

        async for chunk in engine.infer_stream(conversation):
            chunks.append(chunk)

        # Should have content chunks and a finish chunk
        assert len(chunks) >= 2

        # Check first content chunk
        content_chunks = [
            c for c in chunks if c.chunk_type == StreamingChunkType.CONTENT_DELTA
        ]
        assert len(content_chunks) >= 1
        assert content_chunks[0].delta == "Hello"

        # Check finish chunk
        finish_chunks = [
            c for c in chunks if c.chunk_type == StreamingChunkType.FINISH
        ]
        assert len(finish_chunks) == 1
        assert finish_chunks[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_infer_stream_api_error(self, mock_aioresponse):
        """Test streaming with API error."""
        mock_aioresponse.post(
            _TARGET_SERVER,
            status=500,
            body=b"Internal Server Error",
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        conversation = _create_test_conversation()
        chunks = []

        async for chunk in engine.infer_stream(conversation):
            chunks.append(chunk)

        # Should have exactly one error chunk
        assert len(chunks) == 1
        assert chunks[0].chunk_type == StreamingChunkType.ERROR
        assert "500" in chunks[0].error_message

    @pytest.mark.asyncio
    async def test_infer_stream_accumulated_content(self, mock_aioresponse):
        """Test that accumulated content is tracked correctly."""
        sse_response = (
            b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"B"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"C"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        )

        mock_aioresponse.post(
            _TARGET_SERVER,
            body=sse_response,
            status=200,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        conversation = _create_test_conversation()
        chunks = []

        async for chunk in engine.infer_stream(conversation):
            chunks.append(chunk)

        # Check accumulated content grows
        content_chunks = [
            c for c in chunks if c.chunk_type == StreamingChunkType.CONTENT_DELTA
        ]
        assert len(content_chunks) == 3
        assert content_chunks[0].accumulated_content == "A"
        assert content_chunks[1].accumulated_content == "AB"
        assert content_chunks[2].accumulated_content == "ABC"

    @pytest.mark.asyncio
    async def test_infer_stream_conversation_id(self, mock_aioresponse):
        """Test that conversation_id is passed through to chunks."""
        sse_response = (
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        )

        mock_aioresponse.post(
            _TARGET_SERVER,
            body=sse_response,
            status=200,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        conversation = Conversation(
            conversation_id="unique-conv-id-123",
            messages=[Message(content="Hello", role=Role.USER)],
        )

        async for chunk in engine.infer_stream(conversation):
            assert chunk.conversation_id == "unique-conv-id-123"

    @pytest.mark.asyncio
    async def test_infer_stream_not_supported_raises_error(self):
        """Test that engines without streaming support raise NotImplementedError."""
        from oumi.core.inference import BaseInferenceEngine

        class NonStreamingEngine(BaseInferenceEngine):
            def _infer_online(self, input, inference_config=None):
                return input

            def get_supported_params(self):
                return set()

        engine = NonStreamingEngine(
            model_params=_get_default_model_params(),
        )

        assert not engine.supports_streaming()

        conversation = _create_test_conversation()

        with pytest.raises(NotImplementedError) as exc_info:
            async for _ in engine.infer_stream(conversation):
                pass

        assert "does not support streaming" in str(exc_info.value)


#
# SSE Parsing Tests
#
class TestSSEParsing:
    """Tests for SSE line parsing in RemoteInferenceEngine."""

    def test_parse_sse_line_data(self):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        # Valid data line
        result = engine._parse_sse_line('data: {"choices":[{"delta":{"content":"Hi"}}]}')
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hi"

    def test_parse_sse_line_done(self):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        # Done marker
        result = engine._parse_sse_line("data: [DONE]")
        assert result is not None
        assert result.get("_done") is True

    def test_parse_sse_line_empty(self):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        # Empty lines should return None
        assert engine._parse_sse_line("") is None
        assert engine._parse_sse_line("   ") is None
        assert engine._parse_sse_line("\n") is None

    def test_parse_sse_line_comment(self):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        # SSE comments start with ":"
        assert engine._parse_sse_line(": this is a comment") is None

    def test_parse_sse_line_invalid_json(self):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=_get_default_remote_params(),
        )

        # Invalid JSON should return None
        assert engine._parse_sse_line("data: {invalid json}") is None
