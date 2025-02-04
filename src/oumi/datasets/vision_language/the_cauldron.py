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

from typing import Any, Dict

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, ContentItem, Message, Role, Type


@register_dataset("the_cauldron")
class TheCauldronDataset(VisionLanguageSftDataset):
    """Dataset class for the `HuggingFaceM4/the_cauldron` dataset."""

    default_dataset = "HuggingFaceM4/the_cauldron"

    def transform_conversation(self, example: Dict[str, Any]) -> Conversation:
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
                Message(role=Role.USER, content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=example['image_bytes']),
                    ContentItem(type=Type.TEXT, content=example['question']),
                ]),
                Message(role=Role.ASSISTANT, content=example['answer'])
            ]
        )

        return conversation
