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
from oumi.core.configs.params.generation_params import ReasoningEffort
from oumi.core.types import Conversation, Message, Role
from oumi.inference import OpenAIInferenceEngine

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
# OpenAI Reasoning Model E2E Tests
# ============================================================================


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
    print("OpenAI reasoning model test passed")
