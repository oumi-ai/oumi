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

from typing_extensions import override  # noqa: I001

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)


@register_dataset("allenai/pixmo-cap-qa")
class PixmoCapQADataset(VisionLanguageSftDataset):
    """Dataset class for the `allenai/pixmo-cap-qa` dataset.

    The dataset is affected by some image URLs having a 404 issue.
    """

    default_dataset = "allenai/pixmo-cap-qa"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform the example into a Conversation object.

        Sample "question": "[USER] Can you come up with a joke? [ASSISTANT]"
        It starts with a [USER] and ends with an [ASSISTANT] role tag.
        The Assistant response appears in the "answer" field.
        """
        messages: list[Message] = []
        messages.append(
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_URL, content=example["image_url"]),
                ],
            )
        )
        questions = example["question"].strip().split("\n")
        for question in questions:
            if question.startswith("[USER]"):
                question = question[len("[USER]") :].strip()
                messages.append(Message(role=Role.USER, content=question))
            elif question.startswith("[ASSISTANT]"):
                answer = question[len("[ASSISTANT]") :].strip()
                messages.append(Message(role=Role.ASSISTANT, content=answer))
        messages.append(Message(role=Role.ASSISTANT, content=example["answer"]))
        return Conversation(messages=messages)
