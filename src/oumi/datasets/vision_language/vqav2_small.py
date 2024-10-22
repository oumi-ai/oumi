from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role, Type


@register_dataset("merve/vqav2-small")
class Vqav2SmallDataset(VisionLanguageSftDataset):
    default_dataset = "merve/vqav2-small"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        print(example)
        input_text = example["question"][0]
        output_text = example["multiple_choice_answer"][0]

        messages = [
            Message(
                role=Role.USER,
                binary=example["image"],
                type=Type.IMAGE_BINARY,
            ),
            Message(role=Role.USER, content=input_text),
            Message(role=Role.ASSISTANT, content=output_text),
        ]

        return Conversation(messages=messages)
