import base64
from collections.abc import Generator
from enum import Enum
from typing import Any, NamedTuple, Optional, Union

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

    COMPOUND = "compound"
    """Message content is a list of `MessageContentItem`-s.

    The child items are allowed to be of mixed types.
    """


class MessageContentItemCounts(NamedTuple):
    total_items: int
    text_items: int
    image_items: int


class MessageContentItem(pydantic.BaseModel):
    """A sub-part of `Message.content`.

    For example, a multimodal message from `USER` may include
    two `MessageContentItem`-s: one for text, and another for image.

    Note:
        Either content or binary must be provided when creating an instance.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    type: Type
    """The type of the content (e.g., text, image path, image URL)."""

    content: Optional[str] = None
    """Optional text content of the content item.

    One of content or binary must be provided.
    """

    binary: Optional[bytes] = None
    """Optional binary data for the message content item, used for image data.

    One of content or binary must be provided.

    The field is required for `IMAGE_BINARY`, and can be optionally populated for
    `IMAGE_URL`, `IMAGE_PATH` in which case it must be the loaded bytes of
    the image specified in the `content` field.

    The field must be `None` for `TEXT`.
    """

    def is_image(self) -> bool:
        """Checks if the item contains an image."""
        return self.type in (Type.IMAGE_BINARY, Type.IMAGE_URL, Type.IMAGE_PATH)

    def is_text(self) -> bool:
        """Checks if the item contains text."""
        return self.type == Type.TEXT

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
        """Post-initialization method for the `MessageContentItem` model.

        This method is automatically called after the model is initialized.
        Performs additional validation e.g., to ensure that either content or binary
        is provided for the message.

        Raises:
            ValueError: If fields are set to invalid or inconsistent values.
        """
        if self.binary is None and self.content is None:
            raise ValueError(
                "Either content or binary must be provided for the message item "
                f"(Item type: {self.type})."
            )

        if self.is_image():
            if self.type == Type.IMAGE_BINARY and (
                self.binary is None or len(self.binary) == 0
            ):
                raise ValueError(
                    "No image bytes in message content item "
                    f"(Item type: {self.type})."
                )
            if self.type in (Type.IMAGE_PATH, Type.IMAGE_URL) and (
                self.content is None or len(self.content) == 0
            ):
                raise ValueError(f"Content not provided for {self.type} message item.")
        else:
            if self.binary is not None:
                raise ValueError(
                    "Binary can only be provided for the image message items "
                    f"(Item type: {self.type})."
                )

    def __repr__(self) -> str:
        """Returns a string representation of the message item."""
        return f"{self.content}" if self.is_text() else f"<{self.type.upper()}>"


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

    content: Optional[Union[str, list[MessageContentItem]]] = None
    """Optional text content of the message.

    One of content or binary must be provided.
    """

    binary: Optional[bytes] = None
    """Optional binary data for the message, used for image data.

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
        if self.type in (Type.TEXT, Type.IMAGE_BINARY, Type.IMAGE_URL, Type.IMAGE_PATH):
            if not (self.content is None or isinstance(self.content, str)):
                raise RuntimeError(
                    f"Unexpected content type: {type(self.content)} "
                    f"for message type: {self.type}. "
                    f"Consider {Type.COMPOUND}."
                )
        elif self.type == Type.COMPOUND:
            if not (self.content is None or isinstance(self.content, list)):
                raise RuntimeError(
                    f"Unexpected content type: {type(self.content)} "
                    f"for message type: {self.type}. "
                    f"Expected: `list`."
                )

    def _iter_content_items(
        self, *, return_text: bool = False, return_images: bool = False
    ) -> Generator[MessageContentItem, None, None]:
        """Returns a list of content items."""
        if self.type in (Type.TEXT, Type.IMAGE_BINARY, Type.IMAGE_URL, Type.IMAGE_PATH):
            if not (self.content is None or isinstance(self.content, str)):
                raise RuntimeError(
                    f"Unexpected content type: {type(self.content)} "
                    f"for message type: {self.type}. "
                    f"Consider {Type.COMPOUND}."
                )
            is_text = self.type == Type.TEXT
            is_image = not is_text
            if (return_text and is_text) or (return_images and is_image):
                yield MessageContentItem(
                    type=self.type, content=self.content, binary=self.binary
                )
        elif self.type == Type.COMPOUND and self.content is not None:
            assert isinstance(self.content, list), f"Type: {self.type}"
            if return_text and return_images:
                yield from self.content
            else:
                for item in self.content:
                    is_text = item.type == Type.TEXT
                    if (return_text and item.is_text()) or (
                        return_images and item.is_image()
                    ):
                        yield item

    def _iter_all_content_items(
        self, *, return_text: bool = False, return_images: bool = False
    ) -> Generator[MessageContentItem, None, None]:
        return self._iter_content_items(return_text=True, return_images=True)

    def count_content_items(self) -> MessageContentItemCounts:
        """Counts content items by type."""
        total_items: int = 0
        num_text_items: int = 0
        num_image_items: int = 0
        for item in self._iter_all_content_items():
            total_items += 1
            if item.is_text():
                num_text_items += 1
            elif item.is_image():
                num_image_items += 1

        return MessageContentItemCounts(
            total_items=total_items,
            text_items=num_text_items,
            image_items=num_image_items,
        )

    @property
    def content_items(self) -> list[MessageContentItem]:
        """Returns a list of text content items."""
        return [item for item in self._iter_all_content_items()]

    @property
    def image_content_items(self) -> list[MessageContentItem]:
        """Returns a list of image content items."""
        return [item for item in self._iter_content_items(return_images=True)]

    @property
    def text_content_items(self) -> list[MessageContentItem]:
        """Returns a list of text content items."""
        return [item for item in self._iter_content_items(return_text=True)]

    def compute_flattened_text_content(self, separator=" ") -> str:
        """Joins contents of all text items."""
        return separator.join(
            [(item.content or "") for item in self.text_content_items]
        )

    def contains_images(self) -> bool:
        """Checks if the message contains at least one image."""
        first_image = next(self._iter_content_items(return_images=True), None)
        return first_image is not None

    def contains_text(self) -> bool:
        """Checks if the message contains at least one text item."""
        first_text = next(self._iter_content_items(return_text=True), None)
        return first_text is not None

    def contains_image_content_items_only(self) -> bool:
        """Checks if the message contains only image items.

        At least one image item is required.
        """
        counts = self.count_content_items()
        return counts.image_items > 0 and counts.image_items == counts.total_items

    def contains_text_content_items_only(self) -> bool:
        """Checks if the message contains only text items.

        At least one text item is required.
        """
        counts = self.count_content_items()
        return counts.text_items > 0 and counts.text_items == counts.total_items

    def contains_single_text_content_item_only(self) -> bool:
        """Checks if the message contains exactly 1 text content item, and nothing else.

        These are the most common and simple messages, and may need special handling.
        """
        counts = self.count_content_items()
        return counts.text_items == 1 and counts.text_items == counts.total_items

    def __repr__(self) -> str:
        """Returns a string representation of the message."""
        id_str = ""
        if self.id:
            id_str = f"{self.id} - "
        return f"{id_str}{self.role.upper()}: " + " | ".join(
            [repr(item) for item in self._iter_all_content_items()]
        )


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
