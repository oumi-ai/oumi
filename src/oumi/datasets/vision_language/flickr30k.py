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


@register_dataset("nlphuji/flickr30k")
class Flickr30kDataset(VisionLanguageSftDataset):
    """Dataset class for the `nlphuji/flickr30k` dataset."""

    default_dataset = "nlphuji/flickr30k"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = "Describe this image:"
        output_text = example["caption"][0]

        return Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=example["image"]["bytes"],
                        ),
                        ContentItem(type=Type.TEXT, content=input_text),
                    ],
                ),
                Message(role=Role.ASSISTANT, content=output_text),
            ]
        )
