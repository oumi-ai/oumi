from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import (
    Conversation,
    Message,
    MessageContentItem,
    Role,
    Type,
)
from oumi.utils.logging import logger


@register_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
class LlavaInstructMixVsftDataset(VisionLanguageSftDataset):
    """Dataset class for the HuggingFaceH4/llava-instruct-mix-vsft dataset."""

    default_dataset = "HuggingFaceH4/llava-instruct-mix-vsft"

    def _process_text_value(self, s: str) -> str:
        # The data contains occasional `\n` at the beginning or end
        # of text values. Let's strip them.
        return s.strip() if s else ""

    def _parse_user_messages(
        self, message_list: list[dict], images: list[dict]
    ) -> Message:
        role = Role.USER
        if len(message_list) not in (1, 2):
            raise ValueError(
                f"The `content` field for '{role}' must "
                f"contain 1 or 2 elements (question, and, optionally, image). "
                f"Actual: {len(message_list)}"
            )

        text_items: list[MessageContentItem] = []
        image_items: list[MessageContentItem] = []
        for user_message in message_list:
            message_type = user_message["type"]
            if message_type == "text":
                text_items.append(
                    MessageContentItem(
                        type=Type.TEXT,
                        content=self._process_text_value(user_message["text"]),
                    )
                )
            elif message_type == "image":
                image_index = int(user_message["index"])
                if not (image_index >= 0 and image_index < len(images)):
                    raise ValueError(
                        f"Image index is out-of-bounds. "
                        f"Index: {image_index} "
                        f"Image count: {len(images)}"
                    )
                image_dict = images[image_index]
                if "bytes" in image_dict and image_dict["bytes"]:
                    image_items.append(
                        MessageContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=image_dict["bytes"],
                        )
                    )
                elif "path" in image_dict and image_dict["path"]:
                    image_items.append(
                        MessageContentItem(
                            type=Type.IMAGE_PATH,
                            content=image_dict["path"],
                        )
                    )
                else:
                    raise ValueError(
                        f"Image element must include 'bytes' or 'path'. "
                        f"Actual keys: {image_dict.keys()}"
                    )
            else:
                raise ValueError(
                    f"{role}'s question has unknown type: '{message_type}'"
                )

        if len(text_items) != 1:
            raise ValueError(
                f"{role}'s turn must include 1 text question. "
                f"Actual: {len(text_items)}"
            )
        if len(image_items) > 1:
            raise ValueError(
                f"{role}'s turn must include max 1 image. "
                f"Actual: {len(image_items)}"
            )

        # Add image messages before text messages!
        return Message(role=role, content=(image_items + text_items))

    def _parse_assistant_messages(self, message_list: list[dict]) -> Message:
        role = Role.ASSISTANT
        if len(message_list) != 1:
            raise ValueError(
                f"The `content` field for {role} must "
                f"contain exactly 1 element (response). "
                f"Actual: {len(message_list)}"
            )
        response_type = message_list[0]["type"]
        if response_type != "text":
            raise ValueError(
                f"{role}'s response is expected to be text. Actual: {response_type}"
            )

        return Message(
            role=role,
            content=self._process_text_value(message_list[0]["text"]),
        )

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        example_messages = example.get("messages")

        if example_messages is None or len(example_messages) == 0:
            raise ValueError("No messages in input example.")

        images = example.get("images")
        if images is None or len(images) == 0:
            raise ValueError("No images in input example.")
        elif len(images) != 1:
            logger.warning(f"Example contains multiple images: {len(images)}")

        messages: list[Message] = []
        for message in example_messages:
            message_list = message.get("content")
            if message_list is None or len(message_list) == 0:
                raise ValueError("Missing or empty `content` field in message.")

            if message["role"] == "user":
                messages.append(self._parse_user_messages(message_list, images))
            elif message["role"] == "assistant":
                messages.append(self._parse_assistant_messages(message_list))
            else:
                raise ValueError(f"Unknown role: {message['from']}")

        return Conversation(messages=messages)
