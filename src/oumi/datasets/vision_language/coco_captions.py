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

_COCO_COLUMN_SENTENCES = "sentences"
_COCO_COLUMN_RAW = "raw"
_COCO_COLUMN_IMAGE = "image"
_COCO_COLUMN_PATH = "path"
_COCO_COLUMN_BYTES = "bytes"


@register_dataset("coco_captions")
class COCOCaptionsDataset(VisionLanguageSftDataset):
    """Dataset class for the `HuggingFaceM4/COCO` dataset."""

    default_dataset = "HuggingFaceM4/COCO"
    default_prompt = "Describe this image:"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = self.default_prompt

        for required_key in (_COCO_COLUMN_SENTENCES, _COCO_COLUMN_IMAGE):
            if required_key not in example:
                raise ValueError(
                    "Training example doesn't contain '{required_key}' key. "
                    f"Available keys: {example.keys()}."
                )

        if _COCO_COLUMN_RAW not in example[_COCO_COLUMN_SENTENCES]:
            raise ValueError(
                "Training example doesn't contain 'sentences.raw' key. Available keys "
                f"under 'sentences.': {example[_COCO_COLUMN_SENTENCES].keys()}."
            )
        output_text = example[_COCO_COLUMN_SENTENCES][_COCO_COLUMN_RAW]

        user_items: list[ContentItem] = []

        if _COCO_COLUMN_BYTES in example[_COCO_COLUMN_IMAGE]:
            user_items.append(
                ContentItem(
                    binary=example[_COCO_COLUMN_IMAGE][_COCO_COLUMN_BYTES],
                    type=Type.IMAGE_BINARY,
                )
            )
        elif _COCO_COLUMN_PATH in example[_COCO_COLUMN_IMAGE]:
            user_items.append(
                ContentItem(
                    content=example[_COCO_COLUMN_IMAGE][_COCO_COLUMN_PATH],
                    type=Type.IMAGE_PATH,
                )
            )
        else:
            raise ValueError(
                "Training example contains none of required keys: "
                "'image.bytes', 'image.path'. "
                f"Available keys under 'image.': {example[_COCO_COLUMN_IMAGE].keys()}."
            )

        user_items.append(ContentItem(type=Type.TEXT, content=input_text))

        return Conversation(
            messages=[
                Message(role=Role.USER, content=user_items),
                Message(role=Role.ASSISTANT, content=output_text),
            ]
        )
