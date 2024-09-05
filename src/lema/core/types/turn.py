from enum import Enum
from typing import Dict, List, Optional

import pydantic


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:
        """Return the string representation of the Role enum.

        Returns:
            str: The string value of the Role enum.
        """
        return self.value


class Type(str, Enum):
    TEXT = "text"
    IMAGE_PATH = "image_path"
    IMAGE_URL = "image_url"
    IMAGE_BINARY = "image_binary"


class Message(pydantic.BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    role: Role
    type: Type = Type.TEXT
    binary: Optional[bytes] = None

    def model_post_init(self, __context) -> None:
        """Post-initialization method for the Message model.

        This method is automatically called after the model is initialized.
        It performs additional validation to ensure that either content or binary
        is provided for the message.

        Raises:
            ValueError: If both content and binary are None.
        """
        if self.content is None and self.binary is None:
            raise ValueError

    def is_image(self) -> bool:
        """Checks if the message contains an image."""
        return self.type in (Type.IMAGE_BINARY, Type.IMAGE_URL, Type.IMAGE_PATH)

    def is_text(self) -> bool:
        """Checks if the message contains text."""
        return self.type == Type.TEXT


class Conversation(pydantic.BaseModel):
    conversation_id: Optional[str] = None
    messages: List[Message]
    metadata: Dict[str, str] = {}

    def __getitem__(self, idx: int) -> Message:
        """Get the message at the specified index.

        Args:
            idx (int): The index of the message to retrieve.

        Returns:
            Any: The message at the specified index.
        """
        return self.messages[idx]
