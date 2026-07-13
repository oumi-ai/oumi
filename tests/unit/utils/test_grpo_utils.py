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

import pytest

from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.grpo_utils import (
    extract_prompt_images_completion_from_conversation,
)


def _example(conversation: Conversation) -> dict:
    return {"conversation_json": conversation.to_json()}


def test_single_turn_conversation():
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )
    prompt, images, completion = extract_prompt_images_completion_from_conversation(
        _example(convo)
    )
    assert prompt == [{"role": "user", "content": "What is 2+2?"}]
    assert images == []
    assert completion == "4"


def test_multi_turn_conversation_includes_history():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )
    prompt, images, completion = extract_prompt_images_completion_from_conversation(
        _example(convo)
    )
    assert prompt == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert images == []
    assert completion == "4"


def test_raises_when_last_message_not_assistant():
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
            Message(role=Role.USER, content="Still there?"),
        ]
    )
    with pytest.raises(ValueError, match="assistant"):
        extract_prompt_images_completion_from_conversation(_example(convo))


def test_raises_when_fewer_than_two_messages():
    convo = Conversation(
        messages=[
            Message(role=Role.ASSISTANT, content="Hello!"),
        ]
    )
    with pytest.raises(ValueError, match="at least 2 messages"):
        extract_prompt_images_completion_from_conversation(_example(convo))


def test_raises_when_prompt_starts_with_assistant():
    convo = Conversation(
        messages=[
            Message(role=Role.ASSISTANT, content="Unprompted response"),
            Message(role=Role.ASSISTANT, content="Ground truth"),
        ]
    )

    with pytest.raises(ValueError, match="cannot start with an assistant"):
        extract_prompt_images_completion_from_conversation(_example(convo))


def test_raises_when_prompt_is_empty():
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="  "),
            Message(role=Role.ASSISTANT, content="Ground truth"),
        ]
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        extract_prompt_images_completion_from_conversation(_example(convo))


def test_raises_when_completion_is_empty():
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="Question"),
            Message(role=Role.ASSISTANT, content="\n\t"),
        ]
    )

    with pytest.raises(ValueError, match="completion must not be empty"):
        extract_prompt_images_completion_from_conversation(_example(convo))


def test_raises_when_conversation_json_missing():
    with pytest.raises(ValueError, match="conversation_json"):
        extract_prompt_images_completion_from_conversation({"foo": "bar"})


def test_collects_images_from_prompt():
    convo = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )
    prompt, images, completion = extract_prompt_images_completion_from_conversation(
        _example(convo)
    )
    assert prompt == [{"role": "user", "content": "Describe this."}]
    assert images == [{"bytes": b"imgbytes"}]
    assert completion == "A cat."
