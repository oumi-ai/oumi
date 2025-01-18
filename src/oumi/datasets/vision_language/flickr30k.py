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
