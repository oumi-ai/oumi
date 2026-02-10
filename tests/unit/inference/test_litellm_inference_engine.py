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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role


@pytest.fixture
def mock_litellm():
    """Mock the litellm module."""
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        yield


@pytest.fixture
def litellm_engine(mock_litellm):
    """Create a LiteLLMInferenceEngine instance with mocked litellm."""
    _ = mock_litellm  # Ensures litellm is mocked before import
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    return LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
        generation_params=GenerationParams(max_new_tokens=100, temperature=0.5),
    )


def test_litellm_init_with_default_params(mock_litellm):
    """Test initialization with default parameters."""
    _ = mock_litellm
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    engine = LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="anthropic/claude-3-opus"),
    )
    assert engine._model_params.model_name == "anthropic/claude-3-opus"
    assert engine._api_key is None
    assert engine._api_base is None
    assert engine._num_retries == 3
    assert engine._timeout is None
    assert engine._num_workers == 10  # Default value


def test_litellm_init_with_custom_params(mock_litellm):
    """Test initialization with custom parameters."""
    _ = mock_litellm
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    engine = LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
        generation_params=GenerationParams(max_new_tokens=256, temperature=0.7),
        api_key="test-api-key",
        api_base="https://custom.api.com",
        num_retries=5,
        timeout=60.0,
        num_workers=20,
        politeness_policy=0.5,
    )
    assert engine._model_params.model_name == "openai/gpt-4"
    assert engine._generation_params.max_new_tokens == 256
    assert engine._generation_params.temperature == 0.7
    assert engine._api_key == "test-api-key"
    assert engine._api_base == "https://custom.api.com"
    assert engine._num_retries == 5
    assert engine._timeout == 60.0
    assert engine._num_workers == 20
    assert engine._politeness_policy == 0.5


def test_litellm_supported_params(litellm_engine):
    """Test that supported params are returned correctly."""
    supported = litellm_engine.get_supported_params()
    expected = {
        "frequency_penalty",
        "logit_bias",
        "max_new_tokens",
        "presence_penalty",
        "seed",
        "stop_strings",
        "temperature",
        "top_p",
    }
    assert supported == expected


def test_litellm_convert_conversation_to_messages(litellm_engine):
    """Test converting a conversation to LiteLLM message format."""
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )

    messages = litellm_engine._convert_conversation_to_messages(conversation)

    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "Hello!"}
    assert messages[2] == {"role": "assistant", "content": "Hi there!"}


def test_litellm_build_completion_kwargs_basic(litellm_engine):
    """Test building basic completion kwargs."""
    messages = [{"role": "user", "content": "Hello"}]
    generation_params = GenerationParams(max_new_tokens=100, temperature=0.5)

    kwargs = litellm_engine._build_completion_kwargs(messages, generation_params)

    assert kwargs["model"] == "openai/gpt-4"
    assert kwargs["messages"] == messages
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.5
    assert kwargs["num_retries"] == 3
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


def test_litellm_build_completion_kwargs_with_api_credentials(mock_litellm):
    """Test building completion kwargs with API credentials."""
    _ = mock_litellm
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    engine = LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
        api_key="test-key",
        api_base="https://custom.api.com",
        timeout=30.0,
    )

    messages = [{"role": "user", "content": "Hello"}]
    generation_params = GenerationParams()

    kwargs = engine._build_completion_kwargs(messages, generation_params)

    assert kwargs["api_key"] == "test-key"
    assert kwargs["api_base"] == "https://custom.api.com"
    assert kwargs["timeout"] == 30.0


def test_litellm_build_completion_kwargs_with_optional_params(litellm_engine):
    """Test building completion kwargs with optional generation params."""
    messages = [{"role": "user", "content": "Hello"}]
    generation_params = GenerationParams(
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        stop_strings=["STOP", "END"],
        seed=42,
        logit_bias={"100": 1.0, "200": -1.0},
    )

    kwargs = litellm_engine._build_completion_kwargs(messages, generation_params)

    assert kwargs["max_tokens"] == 200
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.9
    assert kwargs["frequency_penalty"] == 0.5
    assert kwargs["presence_penalty"] == 0.3
    assert kwargs["stop"] == ["STOP", "END"]
    assert kwargs["seed"] == 42
    assert kwargs["logit_bias"] == {"100": 1.0, "200": -1.0}


def test_litellm_build_completion_kwargs_excludes_default_penalties(litellm_engine):
    """Test that default penalty values are not included."""
    messages = [{"role": "user", "content": "Hello"}]
    generation_params = GenerationParams(
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    kwargs = litellm_engine._build_completion_kwargs(messages, generation_params)

    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs


def test_litellm_convert_response_to_conversation(litellm_engine):
    """Test converting LiteLLM response to conversation."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is the response."
    mock_response.usage = None

    original_conversation = Conversation(
        messages=[Message(role=Role.USER, content="Hello!")],
        conversation_id="test-id-123",
        metadata={"key": "value"},
    )

    result = litellm_engine._convert_response_to_conversation(
        mock_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].role == Role.USER
    assert result.messages[0].content == "Hello!"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.messages[1].content == "This is the response."
    assert result.conversation_id == "test-id-123"
    assert result.metadata["key"] == "value"


def test_litellm_convert_response_with_usage_metadata(litellm_engine):
    """Test converting response extracts usage metadata."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    mock_response.usage.cached_tokens = 5

    original_conversation = Conversation(
        messages=[Message(role=Role.USER, content="Hello!")]
    )

    result = litellm_engine._convert_response_to_conversation(
        mock_response, original_conversation
    )

    assert result.metadata["usage"]["prompt_tokens"] == 10
    assert result.metadata["usage"]["completion_tokens"] == 20
    assert result.metadata["usage"]["total_tokens"] == 30
    assert result.metadata["usage"]["cached_tokens"] == 5


def test_litellm_convert_response_with_none_content(litellm_engine):
    """Test converting response when content is None."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.usage = None

    original_conversation = Conversation(
        messages=[Message(role=Role.USER, content="Hello!")]
    )

    result = litellm_engine._convert_response_to_conversation(
        mock_response, original_conversation
    )

    assert result.messages[-1].content == ""


def test_litellm_infer_uses_async_completion(mock_litellm):
    """Test that infer uses acompletion for async execution."""
    _ = mock_litellm
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    engine = LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
        generation_params=GenerationParams(max_new_tokens=50),
    )

    # Mock acompletion as an async function
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! How can I help?"
    mock_response.usage = None
    engine._litellm.acompletion = AsyncMock(return_value=mock_response)

    conversations = [
        Conversation(
            messages=[Message(role=Role.USER, content="Hi!")],
            conversation_id="conv-1",
        )
    ]

    results = engine.infer(conversations)

    assert len(results) == 1
    assert len(results[0].messages) == 2
    assert results[0].messages[-1].role == Role.ASSISTANT
    assert results[0].messages[-1].content == "Hello! How can I help?"
    engine._litellm.acompletion.assert_called_once()


def test_litellm_infer_multiple_conversations_concurrent(mock_litellm):
    """Test inference with multiple conversations uses concurrent execution."""
    _ = mock_litellm
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    engine = LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
        num_workers=5,
    )

    # Mock acompletion to return different responses
    async def mock_acompletion(**kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        # Extract message content to create unique responses
        msg = kwargs["messages"][0]["content"]
        response.choices[0].message.content = f"Response to: {msg}"
        response.usage = None
        return response

    engine._litellm.acompletion = mock_acompletion

    conversations = [
        Conversation(
            messages=[Message(role=Role.USER, content=f"Question {i}")],
            conversation_id=f"conv-{i}",
        )
        for i in range(5)
    ]

    results = engine.infer(conversations)

    assert len(results) == 5
    for i, result in enumerate(results):
        assert f"Question {i}" in result.messages[-1].content


@pytest.mark.asyncio
async def test_litellm_streaming_async_inference(mock_litellm):
    """Test async streaming inference yields chunks."""
    _ = mock_litellm
    from oumi.inference.litellm_inference_engine import LiteLLMInferenceEngine

    engine = LiteLLMInferenceEngine(
        model_params=ModelParams(model_name="openai/gpt-4"),
    )

    # Create mock streaming chunks
    chunks = []
    for text in ["Hello", " ", "world", "!"]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)

    # Add a final chunk with no content
    final_chunk = MagicMock()
    final_chunk.choices = [MagicMock()]
    final_chunk.choices[0].delta.content = None
    chunks.append(final_chunk)

    # Create async iterator mock
    async def async_chunk_iter():
        for chunk in chunks:
            yield chunk

    engine._litellm.acompletion = AsyncMock(return_value=async_chunk_iter())

    conversation = Conversation(messages=[Message(role=Role.USER, content="Hi!")])

    result_chunks = []
    async for chunk in engine.infer_stream_async(conversation):
        result_chunks.append(chunk)

    assert result_chunks == ["Hello", " ", "world", "!"]
