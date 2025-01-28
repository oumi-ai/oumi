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


@register_dataset("the_cauldron")
class CauldronDataset(VisionLanguageSftDataset):
    """Dataset class for the `HuggingFaceM4/the_cauldron` dataset."""

    default_dataset = "HuggingFaceM4/the_cauldron"

    def __init__(
        self,
        *,
        name,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the CauldronDataset class."""
        super().__init__(
            dataset_name="HuggingFaceM4/the_cauldron",
            subset=name,
            split="train",  # The Cauldron dataset doesn't have other splits
            **kwargs,
        )

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single example into a Conversation object."""
        images = example["images"]
        if len(images) != 1:
            raise ValueError("Exactly one image per conversation is expected.")
        image_bytes = images[0]["bytes"]

        user_text = example["texts"][0].get("user", "")
        assistant_text = example["texts"][0].get("assistant", "")

        if user_text == "" or assistant_text == "":
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
