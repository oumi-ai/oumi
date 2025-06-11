# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from collections.abc import Generator, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Final, NamedTuple, Optional, Union

import pydantic
from jinja2 import Template

import oumi.core.types.proto.generated.conversation_pb2 as pb2


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


_ROLE_TO_PROTO_ROLE_MAP: Final[Mapping[Role, pb2.Role]] = MappingProxyType(
    {
        Role.SYSTEM: pb2.Role.SYSTEM,
        Role.USER: pb2.Role.USER,
        Role.ASSISTANT: pb2.Role.ASSISTANT,
        Role.TOOL: pb2.Role.TOOL,
    }
)
_PROTO_ROLE_TO_ROLE_MAP: Final[Mapping[pb2.Role, Role]] = MappingProxyType(
    {v: k for k, v in _ROLE_TO_PROTO_ROLE_MAP.items()}
)


def _convert_role_to_proto_role(role: Role) -> pb2.Role:
    """Converts a Role enum to Protocol Buffer format."""
    result: pb2.Role = _ROLE_TO_PROTO_ROLE_MAP.get(role, pb2.Role.ROLE_UNSPECIFIED)
    if result == pb2.Role.ROLE_UNSPECIFIED:
        raise ValueError(f"Invalid role: {role}")
    return result


def _convert_proto_role_to_role(role: pb2.Role) -> Role:
    """Converts a Protocol Buffer role format to role."""
    result: Optional[Role] = _PROTO_ROLE_TO_ROLE_MAP.get(role, None)
    if result is None:
        raise ValueError(f"Invalid role: {role}")
    return result


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

    def __str__(self) -> str:
        """Return the string representation of the Type enum.

        Returns:
            str: The string value of the Type enum.
        """
        return self.value


_CONTENT_ITEM_TYPE_TO_PROTO_TYPE_MAP: Final[Mapping[Type, pb2.ContentPart.Type]] = (
    MappingProxyType(
        {
            Type.TEXT: pb2.ContentPart.TEXT,
            Type.IMAGE_PATH: pb2.ContentPart.IMAGE_PATH,
            Type.IMAGE_URL: pb2.ContentPart.IMAGE_URL,
            Type.IMAGE_BINARY: pb2.ContentPart.IMAGE_BINARY,
        }
    )
)
_CONTENT_ITEM_PROTO_TYPE_TO_TYPE_MAP: Final[Mapping[pb2.ContentPart.Type, Type]] = (
    MappingProxyType({v: k for k, v in _CONTENT_ITEM_TYPE_TO_PROTO_TYPE_MAP.items()})
)


def _convert_type_to_proto_type(content_type: Type) -> pb2.ContentPart.Type:
    """Converts a type enum to Protocol Buffer format."""
    result: pb2.ContentPart.Type = _CONTENT_ITEM_TYPE_TO_PROTO_TYPE_MAP.get(
        content_type, pb2.ContentPart.TYPE_UNSPECIFIED
    )
    if result == pb2.ContentPart.TYPE_UNSPECIFIED:
        raise ValueError(f"Invalid type: {content_type}")
    return result


def _convert_proto_type_to_type(content_type: pb2.ContentPart.Type) -> Type:
    """Converts a Protocol Buffer type format to type."""
    result: Optional[Type] = _CONTENT_ITEM_PROTO_TYPE_TO_TYPE_MAP.get(
        content_type, None
    )
    if result is None:
        raise ValueError(f"Invalid type: {content_type}")
    return result


class ContentItemCounts(NamedTuple):
    """Contains counts of content items in a message by type."""

    total_items: int
    """The total number of content items in a message."""

    text_items: int
    """The number of text content items in a message."""

    image_items: int
    """The number of image content items in a message."""


class ContentItem(pydantic.BaseModel):
    """A sub-part of `Message.content`.

    For example, a multimodal message from `USER` may include
    two `ContentItem`-s: one for text, and another for image.

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
        """Post-initialization method for the `ContentItem` model.

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
                    f"No image bytes in message content item (Item type: {self.type})."
                )
            if self.type in (Type.IMAGE_PATH, Type.IMAGE_URL) and (
                self.content is None or len(self.content) == 0
            ):
                raise ValueError(f"Content not provided for {self.type} message item.")
        else:
            if self.binary is not None:
                raise ValueError(
                    f"Binary can only be provided for images (Item type: {self.type})."
                )

    @staticmethod
    def from_proto(item_proto: pb2.ContentPart) -> "ContentItem":
        """Converts a Protocol Buffer to a content item."""
        if item_proto.HasField("blob") and item_proto.blob:
            return ContentItem(
                type=_convert_proto_type_to_type(item_proto.type),
                binary=item_proto.blob.binary_data,
                content=(item_proto.content or None),
            )
        return ContentItem(
            type=_convert_proto_type_to_type(item_proto.type),
            content=item_proto.content,
        )

    def to_proto(self) -> pb2.ContentPart:
        """Converts a content item to Protocol Buffer format."""
        if self.binary is not None and len(self.binary) > 0:
            return pb2.ContentPart(
                type=_convert_type_to_proto_type(self.type),
                blob=pb2.DataBlob(binary_data=self.binary),
                content=(self.content or None),
            )
        return pb2.ContentPart(
            type=_convert_type_to_proto_type(self.type),
            content=(self.content or None),
        )

    def __repr__(self) -> str:
        """Returns a string representation of the message item."""
        return f"{self.content}" if self.is_text() else f"<{self.type.upper()}>"


class Message(pydantic.BaseModel):
    """A message in a conversation.

    This class represents a single message within a conversation, containing
    various attributes such as role, content, identifier.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: Optional[str] = None
    """Optional unique identifier for the message.

    This attribute can be used to assign a specific identifier to the message,
    which may be useful for tracking or referencing messages within a conversation.

    Returns:
        Optional[str]: The unique identifier of the message, if set; otherwise None.
    """

    content: Union[str, list[ContentItem]]
    """Content of the message.

    For text messages, `content` can be set to a string value.
    For multimodal messages, `content` should be a list of content items of
    potentially different types e.g., text and image.
    """

    role: Role
    """The role of the entity sending the message (e.g., user, assistant, system)."""

    def model_post_init(self, __context) -> None:
        """Post-initialization method for the Message model.

        This method is automatically called after the model is initialized.
        It performs additional validation to ensure that either content or binary
        is provided for the message.

        Raises:
            ValueError: If both content and binary are None.
        """
        if not isinstance(self.content, (str, list)):
            raise ValueError(
                f"Unexpected content type: {type(self.content)}. "
                f"Must by a Python string or a list."
            )

    def _iter_content_items(
        self, *, return_text: bool = False, return_images: bool = False
    ) -> Generator[ContentItem, None, None]:
        """Returns a list of content items."""
        if isinstance(self.content, str):
            if return_text:
                yield ContentItem(type=Type.TEXT, content=self.content)
        elif isinstance(self.content, list):
            if return_text and return_images:
                yield from self.content
            else:
                for item in self.content:
                    if (return_text and item.is_text()) or (
                        return_images and item.is_image()
                    ):
                        yield item

    def _iter_all_content_items(self) -> Generator[ContentItem, None, None]:
        return self._iter_content_items(return_text=True, return_images=True)

    def count_content_items(self) -> ContentItemCounts:
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

        return ContentItemCounts(
            total_items=total_items,
            text_items=num_text_items,
            image_items=num_image_items,
        )

    @property
    def content_items(self) -> list[ContentItem]:
        """Returns a list of text content items."""
        return [item for item in self._iter_all_content_items()]

    @property
    def image_content_items(self) -> list[ContentItem]:
        """Returns a list of image content items."""
        return [item for item in self._iter_content_items(return_images=True)]

    @property
    def text_content_items(self) -> list[ContentItem]:
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
        """Checks if the message contains exactly 1 text item, and nothing else.

        These are the most common and simple messages, and may need special handling.
        """
        counts = self.count_content_items()
        return counts.text_items == 1 and counts.text_items == counts.total_items

    def contains_single_image_content_item_only(self) -> bool:
        """Checks if the message contains exactly 1 image item, and nothing else."""
        counts = self.count_content_items()
        return counts.image_items == 1 and counts.image_items == counts.total_items

    @staticmethod
    def from_proto(message_proto: pb2.Message) -> "Message":
        """Converts a Protocol Buffer to a message."""
        if (len(message_proto.parts) == 1) and (
            message_proto.parts[0].type == pb2.ContentPart.TEXT
        ):
            return Message(
                id=(message_proto.id or None),
                role=_convert_proto_role_to_role(message_proto.role),
                content=message_proto.parts[0].content,
            )

        return Message(
            id=(message_proto.id or None),
            role=_convert_proto_role_to_role(message_proto.role),
            content=[ContentItem.from_proto(part) for part in message_proto.parts],
        )

    def to_proto(self) -> pb2.Message:
        """Converts a message to Protocol Buffer format."""
        return pb2.Message(
            id=self.id,
            role=_convert_role_to_proto_role(self.role),
            parts=[item.to_proto() for item in self.content_items],
        )

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
        messages = self.filter_messages(role=role)
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
        messages = self.filter_messages(role=role)
        return messages[-1] if len(messages) > 0 else None

    def filter_messages(
        self,
        *,
        role: Optional[Role] = None,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> list[Message]:
        """Gets all messages in the conversation, optionally filtered by role.

        Args:
            role (Optional): The role to filter messages by. If None, no filtering
                by role is applied.
            filter_fn (Optional): A predicate to filter messages by. If the predicate
                returns True for a message, then the message is returned.
                Otherwise, the message is excluded.

        Returns:
            List[Message]: A list of all messages matching the criteria.
        """
        if role is not None:
            messages = [message for message in self.messages if role == message.role]
        else:
            messages = self.messages

        if filter_fn is not None:
            messages = [message for message in messages if filter_fn(message)]

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

    @staticmethod
    def from_proto(conversation_proto: pb2.Conversation) -> "Conversation":
        """Converts a conversation from Protocol Buffer format."""
        result: Conversation = Conversation(
            conversation_id=(conversation_proto.conversation_id or None),
            messages=[Message.from_proto(m) for m in conversation_proto.messages],
        )
        for key, value in conversation_proto.metadata.items():
            result.metadata[key] = str(value)
        return result

    def to_proto(self) -> pb2.Conversation:
        """Converts a conversation to Protocol Buffer format."""
        result = pb2.Conversation(
            conversation_id=self.conversation_id,
            messages=[m.to_proto() for m in self.messages],
        )
        if self.metadata is not None:
            for key, value in self.metadata.items():
                result.metadata[key] = str(value)
        return result

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
