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

"""Fixtures for regression tests."""

from pathlib import Path

import PIL.Image
import pytest

from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type


@pytest.fixture
def test_image_path(root_testdata_dir) -> Path:
    """Path to a test image."""
    return root_testdata_dir / "images" / "oumi_logo_light.png"


@pytest.fixture
def test_image(test_image_path) -> PIL.Image.Image:
    """Load a test PIL image."""
    return PIL.Image.open(test_image_path)


@pytest.fixture
def test_image_bytes(test_image_path) -> bytes:
    """Load test image as bytes."""
    return test_image_path.read_bytes()


@pytest.fixture
def single_image_conversation(test_image_bytes) -> Conversation:
    """Conversation with a single image."""
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=test_image_bytes),
                    ContentItem(type=Type.TEXT, content="What is in this image?"),
                ],
            )
        ]
    )


@pytest.fixture
def multi_image_conversation(test_image_bytes) -> Conversation:
    """Conversation with multiple images."""
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=test_image_bytes),
                    ContentItem(type=Type.IMAGE_BINARY, binary=test_image_bytes),
                    ContentItem(
                        type=Type.TEXT, content="Compare these two images."
                    ),
                ],
            )
        ]
    )


@pytest.fixture
def multi_turn_vision_conversation(test_image_bytes) -> Conversation:
    """Multi-turn conversation with image."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I'm doing well, thank you!"),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=test_image_bytes),
                    ContentItem(type=Type.TEXT, content="What about this image?"),
                ],
            ),
        ]
    )


@pytest.fixture
def supported_vlm_models():
    """List of VLM models to test from supported_models.py.

    Returns a list of model configurations with metadata.
    """
    return [
        {
            "name": "llava-hf/llava-1.5-7b-hf",
            "model_type": "llava",
            "tested": True,
            "supports_multi_image": False,
            "has_variable_shape": False,
        },
        {
            "name": "microsoft/Phi-3-vision-128k-instruct",
            "model_type": "phi3_v",
            "tested": True,
            "supports_multi_image": False,
            "has_variable_shape": True,
            "sanitizes_negative_labels": True,
        },
        # Note: Qwen2-VL requires trust_remote_code=True
        # Commenting out for faster baseline tests, can enable later
        # {
        #     "name": "Qwen/Qwen2-VL-7B-Instruct",
        #     "model_type": "qwen2_vl",
        #     "tested": True,
        #     "supports_multi_image": False,
        #     "has_variable_shape": True,
        # },
    ]
