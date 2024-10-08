from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role


@register_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
class COCOCaptionsDataset(VisionLanguageSftDataset):
    """Dataset class for the HuggingFaceH4/llava-instruct-mix-vsft dataset."""

    default_dataset = "HuggingFaceH4/llava-instruct-mix-vsft"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        example_messages = example.get("messages")

        if example_messages is None:
            raise ValueError("Conversation is None")

        print(example)

        messages = []
        for message in example_messages:
            print(message)
            if message["from"] == "human":
                role = Role.USER
            elif message["from"] == "gpt":
                role = Role.ASSISTANT
            else:
                raise ValueError(f"Unknown role: {message['from']}")
            content = message.get("value", "")
            messages.append(Message(role=role, content=content))

        raise ValueError("stop")
        # return Conversation(messages=messages)
