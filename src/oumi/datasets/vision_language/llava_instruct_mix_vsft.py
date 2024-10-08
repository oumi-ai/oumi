from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role, Type
from oumi.utils.logging import logger


@register_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
class LlavaInstructMixVsftDataset(VisionLanguageSftDataset):
    """Dataset class for the HuggingFaceH4/llava-instruct-mix-vsft dataset."""

    default_dataset = "HuggingFaceH4/llava-instruct-mix-vsft"

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

        messages = []
        for message in example_messages:
            print(message)
            if message["role"] == "user":
                role = Role.USER
            elif message["role"] == "assistant":
                role = Role.ASSISTANT
            else:
                raise ValueError(f"Unknown role: {message['from']}")
            message_list = message.get("content")
            if message_list is None or len(message_list) == 0:
                raise ValueError("Missing or empty `content` field in message.")

            if role == Role.USER:
                if len(message_list) not in (1, 2):
                    raise ValueError(
                        f"The `content` field for '{role}' must "
                        f"contain 1 or 2 elements (question, and, optionally, image). "
                        f"Actual: {len(message_list)}"
                    )

                num_texts: int = 0
                num_images: int = 0
                for user_message in message_list:
                    message_type = user_message["type"]
                    if message_type == "text":
                        messages.append(
                            Message(role=role, content=user_message["text"])
                        )
                        num_texts += 1
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
                            messages.append(
                                Message(
                                    role=role,
                                    type=Type.IMAGE_BINARY,
                                    binary=image_dict["bytes"],
                                )
                            )
                        elif "path" in image_dict and image_dict["path"]:
                            messages.append(
                                Message(
                                    role=role,
                                    type=Type.IMAGE_PATH,
                                    content=image_dict["path"],
                                )
                            )
                        else:
                            raise ValueError(
                                f"Image element must include 'bytes' or 'path'. "
                                f"Actual keys: {image_dict.keys()}"
                            )

                        num_images += 1
                    else:
                        raise ValueError(
                            f"{role}'s question has unknown type: '{message_type}'"
                        )

                if num_texts != 1:
                    raise ValueError(
                        f"{role}'s turn must include 1 text question. "
                        f"Actual: {num_texts}"
                    )
                if num_images > 1:
                    raise ValueError(
                        f"{role}'s turn must include max 1 image. "
                        f"Actual: {num_images}"
                    )
            else:
                assert role == Role.ASSISTANT
                if len(message_list) != 1:
                    raise ValueError(
                        f"The `content` field for {role} must "
                        f"contain exactly 1 element (response). "
                        f"Actual: {len(message_list)}"
                    )
                response_type = message_list[0]["type"]
                if response_type != "text":
                    raise ValueError(
                        f"{role}'s response is expected to be text. "
                        f"Actual: {response_type}"
                    )

                messages.append(Message(role=role, content=message_list[0]["text"]))

        return Conversation(messages=messages)
