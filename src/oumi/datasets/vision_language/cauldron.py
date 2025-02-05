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

import warnings
from typing import Optional

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


@register_dataset(registry_name="HuggingFaceM4/the_cauldron")
class CauldronDataset(VisionLanguageSftDataset):
    """Dataset class for the `HuggingFaceM4/the_cauldron` dataset."""

    default_dataset = "HuggingFaceM4/the_cauldron"
    default_subset = "geomverse"

    def __init__(self, *, subset: Optional[str] = None, **kwargs):
        """Initializes a new instance of the CauldronDataset class."""
        if subset in ["multihiertt", "mimic_cgd", "nlvr2", "raven", "spot_the_diff"]:
            raise ValueError(
                f"The {subset} subset of Cauldron includes examples with more than a"
                " single image and is currently not supported by Oumi."
            )

        super().__init__(
            subset=subset if subset else self.default_subset,
            **kwargs,
        )

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single example into a Conversation object."""
        images = example["images"]
        if len(images) != 1:
            raise ValueError(
                "Exactly one image per conversation is expected. "
                f"{len(images)} were given."
            )
        image_bytes = images[0]["bytes"]

        user_text = example["texts"][0].get("user", "")
        assistant_text = example["texts"][0].get("assistant", "")

        if not user_text or not assistant_text:
            warnings.warn("Empty user or assistant text found in example.")

        user_content = [
            ContentItem(
                type=Type.IMAGE_BINARY,
                binary=image_bytes,
            ),
            ContentItem(type=Type.TEXT, content=user_text),
        ]
        assistant_content = [ContentItem(type=Type.TEXT, content=assistant_text)]

        messages = [
            Message(role=Role.USER, content=user_content),
            Message(role=Role.ASSISTANT, content=assistant_content),
        ]

        return Conversation(messages=messages)
