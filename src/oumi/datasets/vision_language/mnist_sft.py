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

from typing import Any, Optional

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


@register_dataset("mnist_sft")
class MnistSftDataset(VisionLanguageSftDataset):
    """MNIST dataset formatted as SFT data.

    MNIST is a well-known small dataset, can be useful for quick tests, prototyping,
    debugging.
    """

    default_dataset = "ylecun/mnist"

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the MnistSftDataset class."""
        super().__init__(
            dataset_name="ylecun/mnist",
            **kwargs,
        )

    @staticmethod
    def _to_digit(value: Any) -> int:
        result: int = 0
        try:
            result = int(value)
        except Exception:
            raise ValueError(
                f"Failed to convert MNIST 'label' ({value}) to an integer!"
            )
        if not (result >= 0 and result <= 9):
            raise ValueError(f"MNIST digit ({result}) is not in [0,9] range!")
        return result

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single MNIST example into a Conversation object."""
        input_text = "What digit is in this picture?"
        output_digit = self._to_digit(example["label"])

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
                Message(role=Role.ASSISTANT, content=str(output_digit)),
            ]
        )
