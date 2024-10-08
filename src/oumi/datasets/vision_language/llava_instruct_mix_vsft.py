from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role
from oumi.utils.logging import logger


@register_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
class LlavaInstructMixVsftDataset(VisionLanguageSftDataset):
    """Dataset class for the HuggingFaceH4/llava-instruct-mix-vsft dataset."""

    default_dataset = "HuggingFaceH4/llava-instruct-mix-vsft"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        print(example)
        example_messages = example.get("messages")

        if not example_messages:
            raise ValueError("No messages in input example.")

        images = example.get("images")
        if not images:
            raise ValueError("No images in input example.")
        elif not isinstance(images, list):
            raise ValueError("The 'images' field is not a list.")
        elif len(images) != 1:
            logger.warning(f"Example contains multiple images: {len(images)}")

        messages = []
        for message in example_messages:
            print(message)
            if message["from"] == "user":
                role = Role.USER
            elif message["from"] == "assistant":
                role = Role.ASSISTANT
            else:
                raise ValueError(f"Unknown role: {message['from']}")
            content_list = message.get("content")
            if not content_list:
                raise ValueError("Missing or empty `content` field in message.")
            elif not isinstance(content_list, list):
                raise ValueError("The `content` field is not a list.")

            if role == Role.USER:
                if len(content_list) != 2:
                    raise ValueError(
                        f"The `content` field for {role} must "
                        f"contain exactly 2 elements (question and image). "
                        f"Actual: {len(content_list)}"
                    )
            else:
                assert role == Role.ASSISTANT
                if len(content_list) != 1:
                    raise ValueError(
                        f"The `content` field for {role} must "
                        f"contain exactly 1 element (response). "
                        f"Actual: {len(content_list)}"
                    )
                response_type = content_list[0]["type"]
                if response_type != "text":
                    raise ValueError(
                        f"{role} response is expected to be text. "
                        f"Actual: {response_type}"
                    )

                messages.append(Message(role=role, content=content_list[0]["text"]))

        raise ValueError("stop")
        # return Conversation(messages=messages)
