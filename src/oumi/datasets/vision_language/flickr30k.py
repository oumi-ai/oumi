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

_DEPRECATION_WARNING = (
    "The nlphuji/flickr30k dataset uses a loading script that is no longer supported "
    "in datasets>=4.0.0. This dataset wrapper is deprecated and will be removed in "
    "a future version. To use this dataset, pin datasets<4.0.0 in your environment."
)


@register_dataset("nlphuji/flickr30k")
class Flickr30kDataset(VisionLanguageSftDataset):
    """Dataset class for the `nlphuji/flickr30k` dataset.

    .. deprecated::
        This dataset is deprecated due to HuggingFace datasets>=4.0.0 removing
        support for dataset loading scripts. The underlying dataset requires
        ``datasets<4.0.0`` to function.
    """

    default_dataset = "nlphuji/flickr30k"

    def __init__(self, **kwargs):
        """Initialize the dataset with a deprecation warning."""
        warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=2)
        super().__init__(**kwargs)

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
