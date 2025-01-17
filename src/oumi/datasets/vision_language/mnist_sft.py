from typing import Any

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
    """MNIST dataset in formatted as SFT data."""

    default_dataset = "ylecun/mnist"

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
        input_text = "What digit is in the picture?"
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
