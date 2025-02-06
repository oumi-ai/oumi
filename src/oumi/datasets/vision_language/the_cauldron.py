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

import numpy as np

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type


@register_dataset("HuggingFaceM4/the_cauldron")
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
        for required_key in ("images", "texts"):
            if required_key not in example:
                raise ValueError(
                    f"Example doesn't contain '{required_key}'. "
                    f"Actual keys: {sorted(example.keys())}"
                )

            if not (isinstance(example[required_key], (list, np.ndarray))):
                actual_type = type(example[required_key])
                raise ValueError(
                    f"Example's '{required_key}' must be a list or np.ndarray. "
                    f"Actual type: {actual_type}"
                )

        images_list: list[Any] = []
        if isinstance(example["images"], np.ndarray):
            images_list = example["images"].tolist()
        else:
            images_list = example["images"]
        num_images = len(images_list)
        if num_images <= 0:
            raise ValueError("Example contains no images.")

        image_content_items: list[ContentItem] = []
        for idx, image_item in enumerate(images_list):
            if not isinstance(image_item, dict):
                actual_type = type(image_item)
                raise ValueError(
                    f"Example image type is not `dict`. Actual type: {actual_type} "
                    f"for image {idx + 1} of {num_images}"
                )
            image_bytes = image_item["bytes"]
            if not isinstance(image_bytes, bytes):
                actual_type = type(image_bytes)
                raise ValueError(
                    f"Example image type is not `bytes`. Actual type: {actual_type} "
                    f"for image {idx + 1} of {num_images}"
                )
            image_content_items.append(
                ContentItem(type=Type.IMAGE_BINARY, binary=image_bytes)
            )

        texts_list: list[dict] = []
        if isinstance(example["texts"], np.ndarray):
            texts_list = example["texts"].tolist()
        else:
            texts_list = example["texts"]
        num_texts = len(texts_list)
        if num_texts <= 0:
            raise ValueError(f"Example must contain some 'texts'. Got: {num_texts}")

        messages_list: list[Message] = []
        for idx, text_entry in enumerate(texts_list):
            if not isinstance(text_entry, dict):
                actual_type = type(text_entry)
                raise ValueError(
                    f"Texts entry must be a `dict`. "
                    f"Actual type: {actual_type} "
                    f"for text entry {idx + 1} of {num_texts}"
                )
            elif not (("user" in text_entry) and ("assistant" in text_entry)):
                raise ValueError(
                    f"Texts entry must contain both 'user' and 'assistant' keys. "
                    f"Got: {sorted(text_entry.keys())} "
                    f"for text entry {idx + 1} of {num_texts}"
                )
            if idx == 0:
                # Only include image(s) once for the first turn.
                messages_list.append(
                    Message(
                        role=Role.USER,
                        content=(
                            image_content_items
                            + [
                                ContentItem(type=Type.TEXT, content=text_entry["user"]),
                            ]
                        ),
                    )
                )
            else:
                messages_list.append(
                    Message(role=Role.USER, content=text_entry["user"])
                )

            messages_list.append(
                Message(role=Role.ASSISTANT, content=text_entry["assistant"]),
            )

        return Conversation(messages=messages_list)
