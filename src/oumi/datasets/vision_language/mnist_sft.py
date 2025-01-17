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

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single MNIST example into a Conversation object."""
        input_text = "What digit is in the picture?"
        output_text = example["label"][0]

        return Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=example["image"],
                        ),
                        ContentItem(type=Type.TEXT, content=input_text),
                    ],
                ),
                Message(role=Role.ASSISTANT, content=output_text),
            ]
        )
