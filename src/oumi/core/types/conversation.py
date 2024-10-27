import base64
from enum import Enum
from typing import Any, Optional, Union

import pydantic
from jinja2 import Template


class Role(str, Enum):
    """Role of the entity sending the message."""

    SYSTEM = "system"
    """Represents a system message in the conversation."""

    USER = "user"
    """Represents a user message in the conversation."""

    ASSISTANT = "assistant"
    """Represents an assistant message in the conversation."""

    TOOL = "tool"
    """Represents a tool message in the conversation."""

    def __str__(self) -> str:
        """Return the string representation of the Role enum.

        Returns:
            str: The string value of the Role enum.
        """
        return self.value


class Type(str, Enum):
    """Type of the message."""

    TEXT = "text"
    """Represents a text message."""

    IMAGE_PATH = "image_path"
    """Represents an image referenced by its file path."""

    IMAGE_URL = "image_url"
    """Represents an image referenced by its URL."""

    IMAGE_BINARY = "image_binary"
    """Represents an image stored as binary data."""


class Message(pydantic.BaseModel):
    """A message in a conversation.

    This class represents a single message within a conversation, containing
    various attributes such as content, role, type, and optional binary data.

    Note:
        Either content or binary must be provided when creating a Message instance.
    """

    id: Optional[str] = None
    """Optional unique identifier for the message.

    This attribute can be used to assign a specific identifier to the message,
    which may be useful for tracking or referencing messages within a conversation.

    Returns:
        Optional[str]: The unique identifier of the message, if set; otherwise None.
    """

    content: Optional[str] = None
    """Optional text content of the message.

    One of content or binary must be provided.
    """

    binary: Optional[bytes] = None
    """Optional binary data for the message, used for image data

    One of content or binary must be provided.
    """

    role: Role
    """The role of the entity sending the message (e.g., user, assistant, system)."""

    type: Type = Type.TEXT
    """The type of the message content (e.g., text, image path, image URL)."""

    @pydantic.field_serializer("binary")
    def _encode_binary(self, value: Optional[bytes]) -> str:
        """Encode binary value as base64 ASCII string.

        This is needed for compatibility with JSON.
        """
        if value is None or len(value) == 0:
            return ""
        return base64.b64encode(value).decode("ascii")

    @pydantic.field_validator("binary", mode="before")
    def _decode_binary(cls, value: Optional[Union[str, bytes]]) -> Optional[bytes]:
        if value is None:
            return None
        elif isinstance(value, str):
            return base64.b64decode(value)
        return value

    def model_post_init(self, __context) -> None:
        """Post-initialization method for the Message model.

        This method is automatically called after the model is initialized.
        It performs additional validation to ensure that either content or binary
        is provided for the message.

        Raises:
            ValueError: If both content and binary are None.
        """
        if self.content is None and self.binary is None:
            raise ValueError(
                "Either content or binary must be provided for the message."
            )

    def is_image(self) -> bool:
        """Checks if the message contains an image."""
        return self.type in (Type.IMAGE_BINARY, Type.IMAGE_URL, Type.IMAGE_PATH)

    def is_text(self) -> bool:
        """Checks if the message contains text."""
        return self.type == Type.TEXT

    def __repr__(self) -> str:
        """Returns a string representation of the message."""
        id_str = ""
        if self.id:
            id_str = f"{self.id} - "
        content = (self.content or "") if self.is_text() else f"<{self.type.upper() }>"
        return f"{id_str}{self.role.upper()}: {content}"


class Conversation(pydantic.BaseModel):
    """Represents a conversation, which is a sequence of messages."""

    conversation_id: Optional[str] = None
    """Optional unique identifier for the conversation.

    This attribute can be used to assign a specific identifier to the conversation,
    which may be useful for tracking or referencing conversations in a larger context.
    """

    messages: list[Message]
    """List of Message objects that make up the conversation."""

    metadata: dict[str, Any] = {}
    """Optional metadata associated with the conversation.

    This attribute allows for storing additional information about the conversation
    in a key-value format. It can be used to include any relevant contextual data.
    """

    def __getitem__(self, idx: int) -> Message:
        """Gets the message at the specified index.

        Args:
            idx (int): The index of the message to retrieve.

        Returns:
            Any: The message at the specified index.
        """
        return self.messages[idx]

    def first_message(self, role: Optional[Role] = None) -> Optional[Message]:
        """Gets the first message in the conversation, optionally filtered by role.

        Args:
            role: The role to filter messages by.
                If None, considers all messages.

        Returns:
            Optional[Message]: The first message matching the criteria,
                or None if no messages are found.
        """
        messages = self.filter_messages(role)
        return messages[0] if len(messages) > 0 else None

    def last_message(self, role: Optional[Role] = None) -> Optional[Message]:
        """Gets the last message in the conversation, optionally filtered by role.

        Args:
            role: The role to filter messages by.
                If None, considers all messages.

        Returns:
            Optional[Message]: The last message matching the criteria,
                or None if no messages are found.
        """
        messages = self.filter_messages(role)
        return messages[-1] if len(messages) > 0 else None

    def filter_messages(self, role: Optional[Role] = None) -> list[Message]:
        """Gets all messages in the conversation, optionally filtered by role.

        Args:
            role: The role to filter messages by.
                If None, returns all messages.

        Returns:
            List[Message]: A list of all messages matching the criteria.
        """
        if role is not None:
            messages = [message for message in self.messages if role == message.role]
        else:
            messages = self.messages
        return messages

    def to_dict(self):
        """Converts the conversation to a dictionary."""
        return self.model_dump(
            mode="json", exclude_unset=True, exclude_defaults=False, exclude_none=True
        )

    def append_id_to_string(self, s: str) -> str:
        """Appends conversation ID to a string.

        Can be useful for log or exception errors messages to allow users
        to identify relevant conversation.
        """
        if not self.conversation_id:
            return s
        suffix = f"Conversation id: {self.conversation_id}."
        return (s.strip() + " " + suffix) if s else suffix

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Converts a dictionary to a conversation."""
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Converts the conversation to a JSON string."""
        return self.model_dump_json(
            exclude_unset=True, exclude_defaults=False, exclude_none=True
        )

    @classmethod
    def from_json(cls, data: str) -> "Conversation":
        """Converts a JSON string to a conversation."""
        return cls.model_validate_json(data)

    def __repr__(self) -> str:
        """Returns a string representation of the conversation."""
        return "\n".join([repr(m) for m in self.messages])


class TemplatedMessage(pydantic.BaseModel):
    """Represents a templated message.

    This class is used to create messages with dynamic content using a template.
    The template can be rendered with variables to produce the final message content.
    """

    template: str
    """The template string used to generate the message content."""

    role: Role
    """The role of the message sender (e.g., USER, ASSISTANT, SYSTEM)."""

    @property
    def content(self) -> str:
        """Renders the content of the message."""
        template = Template(self.template)

        fields = self.model_dump()
        fields.pop("template")  # remove the template from the fields

        return template.render(**fields).strip()

    @property
    def message(self) -> Message:
        """Returns the message in oumi format."""
        content = str(self.content)
        return Message(content=content, role=self.role)
