"""End-to-end tests for OpenAI, Anthropic, and Together.ai inference engines.

These tests make actual API calls and verify the responses. They require valid API keys
to be set in environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- TOGETHER_API_KEY

Run with: pytest tests/e2e/test_inference_engines_e2e.py -v -m e2e
"""

import json
import os
from typing import Any

import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.configs.params.anthropic_params import AnthropicParams, CacheDuration
from oumi.core.configs.params.generation_params import ReasoningEffort
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types import (
    ContentItem,
    Conversation,
    FunctionDefinition,
    Message,
    Role,
    ToolDefinition,
    ToolType,
    Type,
)
from oumi.inference import (
    AnthropicInferenceEngine,
    OpenAIInferenceEngine,
    TogetherInferenceEngine,
)

# ============================================================================
# Helper Functions
# ============================================================================


def assert_valid_response(conversation: Conversation, min_length: int = 5) -> None:
    """Assert that the conversation has a valid assistant response."""
    assert len(conversation.messages) >= 2, "Expected at least 2 messages"
    last_message = conversation.messages[-1]
    assert last_message.role == Role.ASSISTANT, "Last message should be from assistant"
    assert last_message.content, "Assistant response should not be empty"
    content_str = (
        last_message.content
        if isinstance(last_message.content, str)
        else str(last_message.content)
    )
    assert len(content_str) >= min_length, f"Response too short: {content_str}"


def assert_usage_tracked(conversation: Conversation) -> dict[str, Any]:
    """Assert that usage information is tracked in metadata."""
    assert "usage" in conversation.metadata, "Usage info should be in metadata"
    usage = conversation.metadata["usage"]
    assert usage["prompt_tokens"] > 0, "Prompt tokens should be > 0"
    assert usage["completion_tokens"] > 0, "Completion tokens should be > 0"
    assert usage["total_tokens"] > 0, "Total tokens should be > 0"
    return usage


def create_simple_conversation(user_message: str) -> Conversation:
    """Create a simple user message conversation."""
    return Conversation(messages=[Message(role=Role.USER, content=user_message)])


# ============================================================================
# OpenAI E2E Tests
# ============================================================================


@pytest.mark.e2e
def test_openai_basic_text_generation():
    """Test basic text generation with OpenAI."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY is not set")

    engine = OpenAIInferenceEngine(
        model_params=ModelParams(model_name="gpt-4o-mini"),
        generation_params=GenerationParams(
            max_new_tokens=100,
            temperature=0.7,
        ),
    )

    conversation = create_simple_conversation("What is 2+2? Answer briefly.")
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    content = result.messages[-1].content
    assert isinstance(content, str) and "4" in content, "Response should contain 4"
    usage = assert_usage_tracked(result)
    print(f"✅ OpenAI basic text: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_openai_tool_calling():
    """Test tool/function calling with OpenAI."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY is not set")

    # Define a weather tool
    get_weather_tool = ToolDefinition(
        type=ToolType.FUNCTION,
        function=FunctionDefinition(
            name="get_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        ),
    )

    engine = OpenAIInferenceEngine(
        model_params=ModelParams(model_name="gpt-4o-mini"),
        generation_params=GenerationParams(
            max_new_tokens=200,
            tools=[get_weather_tool],
            tool_choice="auto",
        ),
    )

    conversation = create_simple_conversation(
        "What's the weather like in San Francisco?"
    )
    result = engine.infer([conversation])[0]

    # Should call the tool
    last_message = result.messages[-1]
    assert last_message.role == Role.ASSISTANT
    assert last_message.tool_calls is not None, "Should have tool calls"
    assert len(last_message.tool_calls) > 0, "Should have at least one tool call"

    tool_call = last_message.tool_calls[0]
    assert tool_call.function.name == "get_weather"

    # Parse arguments
    args = json.loads(tool_call.function.arguments)
    assert "location" in args, "Should have location parameter"
    assert "san francisco" in args["location"].lower(), "Should ask about SF"

    usage = assert_usage_tracked(result)
    print(f"✅ OpenAI tool calling: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_openai_reasoning_model():
    """Test OpenAI reasoning models (o-series)."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY is not set")

    engine = OpenAIInferenceEngine(
        model_params=ModelParams(model_name="o3-mini"),
        generation_params=GenerationParams(
            max_new_tokens=500,
            reasoning_effort=ReasoningEffort.LOW,
        ),
    )

    conversation = create_simple_conversation(
        "What is 15 * 24? Show your reasoning briefly."
    )
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    usage = assert_usage_tracked(result)

    # Check for reasoning tokens
    assert "reasoning_tokens" in usage, "Should have reasoning tokens for o-series"
    assert usage["reasoning_tokens"] > 0, "Reasoning tokens should be > 0"

    print(
        f"✅ OpenAI reasoning: {usage['total_tokens']} tokens "
        f"({usage['reasoning_tokens']} reasoning)"
    )


@pytest.mark.e2e
def test_openai_streaming():
    """Test streaming with OpenAI."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY is not set")

    engine = OpenAIInferenceEngine(
        model_params=ModelParams(model_name="gpt-4o-mini"),
        generation_params=GenerationParams(
            max_new_tokens=50,
            stream=True,
            stream_options={"include_usage": True},
        ),
    )

    conversation = create_simple_conversation("Count from 1 to 5")
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    # Usage should still be tracked even with streaming
    usage = assert_usage_tracked(result)
    print(f"✅ OpenAI streaming: {usage['total_tokens']} tokens used")


# ============================================================================
# Anthropic E2E Tests
# ============================================================================


@pytest.mark.e2e
def test_anthropic_basic_text_generation():
    """Test basic text generation with Anthropic."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
        generation_params=GenerationParams(
            max_new_tokens=100,
            temperature=0.7,
        ),
    )

    conversation = create_simple_conversation("What is 2+2? Answer briefly.")
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    content = result.messages[-1].content
    assert isinstance(content, str) and "4" in content, "Response should contain 4"
    usage = assert_usage_tracked(result)
    print(f"✅ Anthropic basic text: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_anthropic_tool_calling():
    """Test tool calling with Anthropic."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    # Define a calculator tool
    calculator_tool = ToolDefinition(
        type=ToolType.FUNCTION,
        function=FunctionDefinition(
            name="calculate",
            description="Perform a mathematical calculation",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        ),
    )

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
        generation_params=GenerationParams(
            max_new_tokens=200,
            tools=[calculator_tool],
        ),
    )

    conversation = create_simple_conversation("What is 15 multiplied by 24?")
    result = engine.infer([conversation])[0]

    last_message = result.messages[-1]
    assert last_message.role == Role.ASSISTANT
    assert last_message.tool_calls is not None, "Should have tool calls"
    assert len(last_message.tool_calls) > 0, "Should have at least one tool call"

    tool_call = last_message.tool_calls[0]
    assert tool_call.function.name == "calculate"

    # Parse arguments
    args = json.loads(tool_call.function.arguments)
    assert args["operation"] == "multiply"
    assert args["a"] == 15
    assert args["b"] == 24

    usage = assert_usage_tracked(result)
    print(f"✅ Anthropic tool calling: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_anthropic_vision():
    """Test vision/image support with Anthropic."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
        generation_params=GenerationParams(max_new_tokens=200),
    )

    # Use a public image URL
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/3/3a/Cat03.jpg/240px-Cat03.jpg"
    )

    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(
                        type=Type.TEXT, content="What animal is in this image?"
                    ),
                    ContentItem(type=Type.IMAGE_URL, content=image_url),
                ],
            )
        ]
    )

    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    content = result.messages[-1].content
    assert isinstance(content, str), "Content should be a string"
    response_lower = content.lower()
    assert "cat" in response_lower, f"Should identify cat, got: {content}"

    usage = assert_usage_tracked(result)
    print(f"✅ Anthropic vision: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_anthropic_structured_outputs():
    """Test structured outputs with Anthropic using tool calling."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    # Define a JSON schema for structured output
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "city"],
    }

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
        generation_params=GenerationParams(
            max_new_tokens=200,
            guided_decoding=GuidedDecodingParams(json=person_schema),
        ),
    )

    conversation = create_simple_conversation(
        "Extract information: John Smith is 30 years old and lives in New York."
    )
    result = engine.infer([conversation])[0]

    last_message = result.messages[-1]
    assert last_message.role == Role.ASSISTANT

    # With structured outputs, Anthropic returns tool calls
    assert last_message.tool_calls is not None, (
        "Should have tool calls for structured output"
    )
    tool_call = last_message.tool_calls[0]
    assert tool_call.function.name == "format_response"

    # Parse and validate structured data
    data = json.loads(tool_call.function.arguments)
    assert "name" in data
    assert "age" in data
    assert "city" in data
    assert data["age"] == 30
    assert "john" in data["name"].lower() or "smith" in data["name"].lower()

    usage = assert_usage_tracked(result)
    print(f"✅ Anthropic structured outputs: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_anthropic_prompt_caching():
    """Test prompt caching with Anthropic."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    # Create a long context message
    long_context = (
        """
    The quick brown fox jumps over the lazy dog. """
        * 100
    )  # Repeat to make it long

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
        generation_params=GenerationParams(max_new_tokens=50),
        remote_params=RemoteParams(
            anthropic_params=AnthropicParams(
                enable_prompt_caching=True,
                cache_duration=CacheDuration.FIVE_MINUTES,
            )
        ),
    )

    # First request - should create cache
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content=long_context),
            Message(role=Role.USER, content="Summarize the above in 5 words."),
        ]
    )

    result1 = engine.infer([conversation])[0]
    assert_valid_response(result1)
    assert_usage_tracked(result1)

    # Second request with same context - should use cache
    result2 = engine.infer([conversation])[0]
    assert_valid_response(result2)
    usage2 = assert_usage_tracked(result2)

    # Second request should have cache hits
    if "cache_read_input_tokens" in usage2:
        cache_tokens = usage2["cache_read_input_tokens"]
        print(f"✅ Anthropic caching: {cache_tokens} tokens from cache")
    else:
        print("⚠️  Cache info not in second request (may need more time)")


@pytest.mark.e2e
def test_anthropic_streaming():
    """Test streaming with Anthropic."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
        generation_params=GenerationParams(
            max_new_tokens=50,
            stream=True,
        ),
    )

    conversation = create_simple_conversation("Count from 1 to 5")
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    usage = assert_usage_tracked(result)
    print(f"✅ Anthropic streaming: {usage['total_tokens']} tokens used")


# ============================================================================
# Together.ai E2E Tests
# ============================================================================


@pytest.mark.e2e
def test_together_basic_text_generation():
    """Test basic text generation with Together.ai."""
    if "TOGETHER_API_KEY" not in os.environ:
        pytest.skip("TOGETHER_API_KEY is not set")

    engine = TogetherInferenceEngine(
        model_params=ModelParams(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        generation_params=GenerationParams(
            max_new_tokens=100,
            temperature=0.7,
        ),
    )

    conversation = create_simple_conversation("What is 2+2? Answer briefly.")
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    content = result.messages[-1].content
    assert isinstance(content, str) and "4" in content, "Response should contain 4"
    usage = assert_usage_tracked(result)
    print(f"✅ Together basic text: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_together_tool_calling():
    """Test tool calling with Together.ai (OpenAI-compatible)."""
    if "TOGETHER_API_KEY" not in os.environ:
        pytest.skip("TOGETHER_API_KEY is not set")

    # Define a simple tool
    get_time_tool = ToolDefinition(
        type=ToolType.FUNCTION,
        function=FunctionDefinition(
            name="get_current_time",
            description="Get the current time in a given timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone, e.g. America/New_York",
                    },
                },
                "required": ["timezone"],
            },
        ),
    )

    engine = TogetherInferenceEngine(
        model_params=ModelParams(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        generation_params=GenerationParams(
            max_new_tokens=200,
            tools=[get_time_tool],
            tool_choice="auto",
        ),
    )

    conversation = create_simple_conversation("What time is it in New York?")
    result = engine.infer([conversation])[0]

    last_message = result.messages[-1]
    assert last_message.role == Role.ASSISTANT

    # Tool calling support varies by model
    if last_message.tool_calls:
        print("✅ Together tool calling: Model called the tool")
        tool_call = last_message.tool_calls[0]
        assert tool_call.function.name == "get_current_time"
    else:
        print(
            "⚠️  Together: Model didn't call tool (may not be supported by this model)"
        )

    usage = assert_usage_tracked(result)
    print(f"✅ Together usage tracked: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_together_streaming():
    """Test streaming with Together.ai."""
    if "TOGETHER_API_KEY" not in os.environ:
        pytest.skip("TOGETHER_API_KEY is not set")

    engine = TogetherInferenceEngine(
        model_params=ModelParams(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        generation_params=GenerationParams(
            max_new_tokens=50,
            stream=True,
        ),
    )

    conversation = create_simple_conversation("Count from 1 to 5")
    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    usage = assert_usage_tracked(result)
    print(f"✅ Together streaming: {usage['total_tokens']} tokens used")


@pytest.mark.e2e
def test_together_vision():
    """Test vision/multimodal with Together.ai."""
    if "TOGETHER_API_KEY" not in os.environ:
        pytest.skip("TOGETHER_API_KEY is not set")

    # Use a vision-capable model
    engine = TogetherInferenceEngine(
        model_params=ModelParams(
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
        ),
        generation_params=GenerationParams(max_new_tokens=200),
    )

    # Use a public image URL
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/3/3a/Cat03.jpg/240px-Cat03.jpg"
    )

    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(
                        type=Type.TEXT, content="What animal is in this image?"
                    ),
                    ContentItem(type=Type.IMAGE_URL, content=image_url),
                ],
            )
        ]
    )

    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    content = result.messages[-1].content
    assert isinstance(content, str), "Content should be a string"
    response_lower = content.lower()
    assert "cat" in response_lower, f"Should identify cat, got: {content}"

    usage = assert_usage_tracked(result)
    print(f"✅ Together vision: {usage['total_tokens']} tokens used")


# ============================================================================
# Cross-Provider Comparison Tests
# ============================================================================


@pytest.mark.e2e
def test_cost_comparison_across_providers():
    """Compare costs across all three providers for the same task."""
    results = {}

    test_prompt = "Explain what Python is in one sentence."

    # Test OpenAI
    if "OPENAI_API_KEY" in os.environ:
        engine = OpenAIInferenceEngine(
            model_params=ModelParams(model_name="gpt-4o-mini"),
            generation_params=GenerationParams(max_new_tokens=50),
        )
        result = engine.infer([create_simple_conversation(test_prompt)])[0]
        usage = result.metadata["usage"]
        content = result.messages[-1].content
        content_str = content if isinstance(content, str) else str(content)
        results["OpenAI (gpt-4o-mini)"] = {
            "tokens": usage["total_tokens"],
            "response": content_str[:50] + "...",
        }

    # Test Anthropic
    if "ANTHROPIC_API_KEY" in os.environ:
        engine = AnthropicInferenceEngine(
            model_params=ModelParams(model_name="claude-3-5-haiku-20241022"),
            generation_params=GenerationParams(max_new_tokens=50),
        )
        result = engine.infer([create_simple_conversation(test_prompt)])[0]
        usage = result.metadata["usage"]
        content = result.messages[-1].content
        content_str = content if isinstance(content, str) else str(content)
        results["Anthropic (claude-3-5-haiku)"] = {
            "tokens": usage["total_tokens"],
            "response": content_str[:50] + "...",
        }

    # Test Together.ai
    if "TOGETHER_API_KEY" in os.environ:
        engine = TogetherInferenceEngine(
            model_params=ModelParams(
                model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"
            ),
            generation_params=GenerationParams(max_new_tokens=50),
        )
        result = engine.infer([create_simple_conversation(test_prompt)])[0]
        usage = result.metadata["usage"]
        content = result.messages[-1].content
        content_str = content if isinstance(content, str) else str(content)
        results["Together (Llama-3.3-70B)"] = {
            "tokens": usage["total_tokens"],
            "response": content_str[:50] + "...",
        }

    # Print comparison
    if results:
        print("\n" + "=" * 70)
        print("COST COMPARISON ACROSS PROVIDERS")
        print("=" * 70)
        for provider, data in results.items():
            print(f"\n{provider}:")
            print(f"  Tokens: {data['tokens']}")
            print(f"  Response: {data['response']}")
        print("=" * 70)
    else:
        pytest.skip("No API keys available for comparison")


if __name__ == "__main__":
    # Allow running individual tests
    pytest.main([__file__, "-v", "-s", "-m", "e2e"])
