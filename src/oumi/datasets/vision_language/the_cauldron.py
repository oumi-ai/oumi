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

from typing import Any

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type


@register_dataset("the_cauldron")
class TheCauldronDataset(VisionLanguageSftDataset):
    """Dataset class for the `HuggingFaceM4/the_cauldron` dataset.

    The `HuggingFaceM4/the_cauldron` dataset is a comprehensive collection of
    50 vision-language datasets, primarily training sets, used
    for fine-tuning the Idefics2 vision-language model.
    The datasets cover various domains such as general visual question answering,
    captioning, OCR, document understanding, chart/figure understanding,
    table understanding, reasoning, logic, maths, textbook/academic questions,
    differences between images, and screenshot to code.
    """

    default_dataset = "HuggingFaceM4/the_cauldron"

    def transform_conversation(self, example: dict[str, Any]) -> Conversation:
        """Transform raw data into a conversation with images."""
        # Transform the raw example into a Conversation object
        # 'example' represents one row of the raw dataset
        # Structure of 'example':
        # {
        #     'image_bytes': bytes,  # PNG bytes of the image
        #     'question': str,       # The user's question about the image
        #     'answer': str          # The assistant's response
        # }
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY, binary=example["image_bytes"]
                        ),
                        ContentItem(type=Type.TEXT, content=example["question"]),
                    ],
                ),
                Message(role=Role.ASSISTANT, content=example["answer"]),
            ]
        )

        return conversation
