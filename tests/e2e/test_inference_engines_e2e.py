"""End-to-end tests for OpenAI, Anthropic, and Together.ai inference engines.

These tests make actual API calls and verify the responses. They require valid API keys
to be set in environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- TOGETHER_API_KEY

Run with: pytest tests/e2e/test_inference_engines_e2e.py -v -m e2e
"""

import os

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types import ContentItem, Conversation, Message, Role, Type
from oumi.inference import AnthropicInferenceEngine

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


def create_simple_conversation(user_message: str) -> Conversation:
    """Create a simple user message conversation."""
    return Conversation(messages=[Message(role=Role.USER, content=user_message)])


# ============================================================================
# Anthropic Vision E2E Tests
# ============================================================================


@pytest.mark.e2e
def test_anthropic_vision():
    """Test vision/multimodal with Anthropic."""
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
                    ContentItem(type=Type.IMAGE_URL, content=image_url),
                    ContentItem(type=Type.TEXT, content="What animal is in this image?"),
                ],
            )
        ]
    )

    result = engine.infer([conversation])[0]

    assert_valid_response(result)
    # Should mention cat or feline
    response = result.messages[-1].content
    assert isinstance(response, str)
    assert any(
        word in response.lower() for word in ["cat", "kitten", "feline"]
    ), f"Expected mention of cat in: {response}"
    print("Anthropic vision test passed")
