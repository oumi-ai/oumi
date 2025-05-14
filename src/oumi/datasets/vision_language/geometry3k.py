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

from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)

_GEOMETRY3K_INSTRUCTION = (
    r"You FIRST think about the reasoning process as an internal monologue and then "
    r"provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)


@register_dataset("hiyouga/geometry3k")
class Geometry3kDataset(VisionLanguageSftDataset):
    """Dataset class for the `hiyouga/geometry3k` dataset."""

    default_dataset = "hiyouga/geometry3k"

    def __init__(
        self,
        *,
        add_system_instruction: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the Geometry3kDataset class."""
        self._add_system_instruction = add_system_instruction
        super().__init__(**kwargs)

    def _process_text_value(self, s: str) -> str:
        # The data contains occasional `\n` at the beginning or end
        # of text values. Let's strip them.
        return s.strip() if s else ""

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = self._process_text_value(example["problem"])
        input_text = input_text.removeprefix("<image>")
        if not self._add_system_instruction:
            input_text = input_text + " " + _GEOMETRY3K_INSTRUCTION

        output_text = self._process_text_value(example["answer"])
        image = example["images"]
        if len(image) != 1:
            raise ValueError(
                f"Expected 1 image, but got {len(image)} images in example."
            )
        image = image[0]

        messages = (
            [
                Message(
                    role=Role.SYSTEM,
                    content=_GEOMETRY3K_INSTRUCTION,
                )
            ]
            if self._add_system_instruction
            else []
        ) + [
            Message(
                role=Role.USER,
                content=[
                    ContentItem(
                        type=Type.IMAGE_BINARY,
                        binary=image["bytes"],
                    ),
                    ContentItem(type=Type.TEXT, content=input_text),
                ],
            ),
            Message(role=Role.ASSISTANT, content=output_text),
        ]

        return Conversation(messages=messages)
