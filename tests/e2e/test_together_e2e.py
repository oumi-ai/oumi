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

"""End-to-end tests for Together.ai inference engine.

These tests make actual API calls and verify the responses. They require valid API keys
to be set in environment variables:
- TOGETHER_API_KEY

Run with: pytest tests/e2e/test_together_e2e.py -v -m e2e
"""

import os

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.inference import TogetherInferenceEngine


def create_simple_conversation(user_message: str) -> Conversation:
    """Create a simple user message conversation."""
    return Conversation(messages=[Message(role=Role.USER, content=user_message)])


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

    assert len(result.messages) >= 2, "Expected at least 2 messages"
    last_message = result.messages[-1]
    assert last_message.role == Role.ASSISTANT, "Last message should be from assistant"
    assert last_message.content, "Assistant response should not be empty"
    content = last_message.content
    assert isinstance(content, str) and "4" in content, "Response should contain 4"
    print(f"Together basic text generation passed")


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

    assert len(result.messages) >= 2
    last_message = result.messages[-1]
    content = last_message.content
    assert isinstance(content, str), "Content should be a string"
    response_lower = content.lower()
    assert "cat" in response_lower, f"Should identify cat, got: {content}"
    print(f"Together vision test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "e2e"])
