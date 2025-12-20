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
import warnings
from collections.abc import Generator
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Union

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


class OutputFormat(str, Enum):
    """Output format for conversation serialization.

    This enum controls how conversations are serialized to dictionaries/JSON.
    """

    OUMI = "oumi"
    """Oumi format (HuggingFace-compatible).

    This is the default format, compatible with HuggingFace transformers.

    - Text: ``{"type": "text", "text": "..."}``
    - Image URL: ``{"type": "image", "url": "..."}``
    - Image path: ``{"type": "image", "path": "..."}``
    - Image binary: ``{"type": "image", "url": "data:image/png;base64,..."}``
    """

    OPENAI = "openai"
    """OpenAI API format.

    Use this format when calling OpenAI-compatible APIs.

    - Text: ``{"type": "text", "text": "..."}``
    - Image: ``{"type": "image_url", "image_url": {"url": "..."}}``
    """

    OUMI_LEGACY = "oumi_legacy"
    """Legacy Oumi format (deprecated).

    This format is provided for backward compatibility with existing pipelines.

    - Text: ``{"type": "text", "content": "..."}``
    - Image URL: ``{"type": "image_url", "content": "..."}``
    - Image path: ``{"type": "image_path", "content": "..."}``
    - Image binary: ``{"type": "image_binary", "binary": "..."}``
    """

    def __str__(self) -> str:
        """Return the string representation of the OutputFormat enum."""
        return self.value


class Type(str, Enum):
    """Type of the content item in a message."""

    TEXT = "text"
    """Represents a text message."""

    IMAGE = "image"
    """Represents an image (HuggingFace format).

    Use with `url` field for URLs/data URIs, or `path` field for local paths.
    """

    # Deprecated types - use IMAGE instead
    IMAGE_PATH = "image_path"
    """Represents an image referenced by its file path.

    .. deprecated::
        Use `Type.IMAGE` with the `path` field instead.
    """

    IMAGE_URL = "image_url"
    """Represents an image referenced by its URL.

    .. deprecated::
        Use `Type.IMAGE` with the `url` field instead.
    """

    IMAGE_BINARY = "image_binary"
    """Represents an image stored as binary data.

    .. deprecated::
        Use `Type.IMAGE` with the `url` field containing a data URI instead.
    """

    def __str__(self) -> str:
        """Return the string representation of the Type enum.

        Returns:
            str: The string value of the Type enum.
        """
        return self.value

    def is_deprecated(self) -> bool:
        """Check if this type is deprecated."""
        return self in (Type.IMAGE_PATH, Type.IMAGE_URL, Type.IMAGE_BINARY)


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

    Examples:
        Text content (HuggingFace format)::

            ContentItem(type=Type.TEXT, text="Hello, world!")

        Text content (legacy format, still supported)::

            ContentItem(type=Type.TEXT, content="Hello, world!")

        Image from URL (HuggingFace format)::

            ContentItem(type=Type.IMAGE, url="https://example.com/image.jpg")

        Image from local path (HuggingFace format)::

            ContentItem(type=Type.IMAGE, path="/path/to/image.jpg")

        Image with binary data (HuggingFace format)::

            ContentItem(type=Type.IMAGE, url="data:image/png;base64,...")

    Note:
        The legacy types ``IMAGE_URL``, ``IMAGE_PATH``, and ``IMAGE_BINARY``
        are deprecated. Use ``Type.IMAGE`` with the appropriate field instead.
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        populate_by_name=True,  # Accept both field names and aliases
    )

    type: Type
    """The type of the content (e.g., text, image)."""

    content: Optional[str] = None
    """Text content or legacy image URL/path.

    For ``Type.TEXT``: contains the text content.
    For deprecated ``IMAGE_URL``/``IMAGE_PATH``: contains the URL or path.

    Note:
        For new code, prefer using ``text`` for text content,
        and ``url``/``path`` for images with ``Type.IMAGE``.
    """

    # New HuggingFace-style fields
    text: Optional[str] = pydantic.Field(default=None)
    """Text content (HuggingFace format).

    Alternative to ``content`` for ``Type.TEXT``. When both are provided,
    ``text`` takes precedence for HuggingFace format output.
    """

    url: Optional[str] = None
    """Image URL or data URI (HuggingFace format).

    For ``Type.IMAGE``: contains the HTTP(S) URL or a data URI
    (e.g., ``data:image/png;base64,...``).
    """

    path: Optional[str] = None
    """Local file path (HuggingFace format).

    For ``Type.IMAGE``: contains the local filesystem path to the image.
    """

    binary: Optional[bytes] = None
    """Binary image data (internal use).

    Contains the raw image bytes. For ``Type.IMAGE``, this is populated
    when loading images from URLs or paths, or when providing inline data.

    For deprecated ``IMAGE_BINARY``: this field is required.
    For ``IMAGE_URL``/``IMAGE_PATH``: optionally populated with loaded bytes.
    """

    def is_image(self) -> bool:
        """Checks if the item contains an image."""
        return self.type in (Type.IMAGE, Type.IMAGE_BINARY, Type.IMAGE_URL, Type.IMAGE_PATH)

    def is_text(self) -> bool:
        """Checks if the item contains text."""
        return self.type == Type.TEXT

    def get_text(self) -> Optional[str]:
        """Returns the text content, checking both 'text' and 'content' fields."""
        return self.text if self.text is not None else self.content

    def get_image_url(self) -> Optional[str]:
        """Returns the image URL, checking both 'url' and 'content' fields."""
        if self.url is not None:
            return self.url
        if self.type in (Type.IMAGE_URL, Type.IMAGE):
            return self.content
        return None

    def get_image_path(self) -> Optional[str]:
        """Returns the image path, checking both 'path' and 'content' fields."""
        if self.path is not None:
            return self.path
        if self.type == Type.IMAGE_PATH:
            return self.content
        return None

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
        Performs additional validation and emits deprecation warnings.

        Raises:
            ValueError: If fields are set to invalid or inconsistent values.
        """
        # Emit deprecation warnings for legacy image types
        if self.type.is_deprecated():
            warnings.warn(
                f"Type.{self.type.name} is deprecated. Use Type.IMAGE with "
                f"{'url' if self.type == Type.IMAGE_URL else 'path' if self.type == Type.IMAGE_PATH else 'url (data URI)'} "
                f"field instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        # Validate based on type
        if self.type == Type.TEXT:
            # For TEXT, need either content or text field
            if self.content is None and self.text is None:
                raise ValueError(
                    "Either 'content' or 'text' must be provided for TEXT items."
                )
            if self.binary is not None:
                raise ValueError(
                    "Binary data cannot be provided for TEXT items."
                )
            if self.url is not None or self.path is not None:
                raise ValueError(
                    "The 'url' and 'path' fields cannot be used with TEXT items."
                )

        elif self.type == Type.IMAGE:
            # For IMAGE (HuggingFace format), need url, path, or binary
            has_url = self.url is not None and len(self.url) > 0
            has_path = self.path is not None and len(self.path) > 0
            has_binary = self.binary is not None and len(self.binary) > 0

            if not has_url and not has_path and not has_binary:
                raise ValueError(
                    "For Type.IMAGE, provide 'url', 'path', or 'binary' field."
                )
            if has_url and has_path:
                raise ValueError(
                    "Cannot provide both 'url' and 'path' for the same IMAGE item."
                )
            if self.content is not None or self.text is not None:
                raise ValueError(
                    "The 'content' and 'text' fields cannot be used with IMAGE items. "
                    "Use 'url' or 'path' instead."
                )

        elif self.type == Type.IMAGE_BINARY:
            # Legacy IMAGE_BINARY requires binary data
            if self.binary is None or len(self.binary) == 0:
                raise ValueError(
                    f"No image bytes in message content item (Item type: {self.type})."
                )

        elif self.type in (Type.IMAGE_PATH, Type.IMAGE_URL):
            # Legacy IMAGE_PATH/IMAGE_URL require content field
            if self.content is None or len(self.content) == 0:
                raise ValueError(f"Content not provided for {self.type} message item.")

        else:
            raise ValueError(f"Unknown content item type: {self.type}")

    def __repr__(self) -> str:
        """Returns a string representation of the message item."""
        if self.is_text():
            return f"{self.get_text()}"
        else:
            return f"<{self.type.value.upper()}>"

    def to_dict(
        self,
        output_format: OutputFormat = OutputFormat.OUMI,
    ) -> dict[str, Any]:
        """Serialize the content item to a dictionary.

        Args:
            output_format: The serialization format to use.
                Defaults to OUMI (HuggingFace-compatible format).

        Returns:
            Dictionary representation of the content item.
        """
        if self.type == Type.TEXT:
            return self._serialize_text(output_format)
        elif self.is_image():
            return self._serialize_image(output_format)
        else:
            raise ValueError(f"Unknown content item type: {self.type}")

    def _serialize_text(self, output_format: OutputFormat) -> dict[str, Any]:
        """Serialize text content item."""
        text_value = self.get_text() or ""

        if output_format == OutputFormat.OUMI_LEGACY:
            return {"type": "text", "content": text_value}
        else:  # OUMI or OPENAI - both use "text" key
            return {"type": "text", "text": text_value}

    def _serialize_image(self, output_format: OutputFormat) -> dict[str, Any]:
        """Serialize image content item."""
        if output_format == OutputFormat.OUMI:
            return self._serialize_image_oumi()
        elif output_format == OutputFormat.OPENAI:
            return self._serialize_image_openai()
        else:  # OUMI_LEGACY
            return self._serialize_image_legacy()

    def _serialize_image_oumi(self) -> dict[str, Any]:
        """Serialize image in Oumi (HuggingFace) format."""
        # Handle new IMAGE type
        if self.type == Type.IMAGE:
            if self.path:
                result: dict[str, Any] = {"type": "image", "path": self.path}
            else:
                result = {"type": "image", "url": self.url or ""}
            if self.binary:
                result["binary"] = base64.b64encode(self.binary).decode("ascii")
            return result

        # Handle legacy types - convert to new format
        if self.type == Type.IMAGE_PATH:
            result = {"type": "image", "path": self.content or ""}
            if self.binary:
                result["binary"] = base64.b64encode(self.binary).decode("ascii")
            return result
        elif self.type == Type.IMAGE_BINARY:
            # Convert binary to data URI
            b64 = base64.b64encode(self.binary or b"").decode("ascii")
            return {"type": "image", "url": f"data:image/png;base64,{b64}"}
        else:  # IMAGE_URL
            result = {"type": "image", "url": self.content or ""}
            if self.binary:
                result["binary"] = base64.b64encode(self.binary).decode("ascii")
            return result

    def _serialize_image_openai(self) -> dict[str, Any]:
        """Serialize image in OpenAI format."""
        # Get the URL - either from url field, content field, or generate from binary
        if self.url:
            url = self.url
        elif self.binary and (self.type in (Type.IMAGE, Type.IMAGE_BINARY)):
            b64 = base64.b64encode(self.binary).decode("ascii")
            url = f"data:image/png;base64,{b64}"
        elif self.content:
            url = self.content
        else:
            url = ""

        return {
            "type": "image_url",
            "image_url": {"url": url},
        }

    def _serialize_image_legacy(self) -> dict[str, Any]:
        """Serialize image in legacy Oumi format."""
        # Handle new IMAGE type - map back to legacy format
        if self.type == Type.IMAGE:
            if self.path:
                result: dict[str, Any] = {"type": "image_path", "content": self.path}
                if self.binary:
                    result["binary"] = base64.b64encode(self.binary).decode("ascii")
                return result
            elif self.binary and not self.url:
                # Pure binary without URL
                return {
                    "type": "image_binary",
                    "binary": base64.b64encode(self.binary).decode("ascii"),
                }
            else:
                result = {"type": "image_url", "content": self.url or ""}
                if self.binary:
                    result["binary"] = base64.b64encode(self.binary).decode("ascii")
                return result

        # Handle legacy types - preserve original format
        if self.type == Type.IMAGE_BINARY:
            return {
                "type": "image_binary",
                "binary": base64.b64encode(self.binary or b"").decode("ascii"),
            }
        elif self.type == Type.IMAGE_PATH:
            result: dict[str, Any] = {"type": "image_path", "content": self.content or ""}
            if self.binary:
                result["binary"] = base64.b64encode(self.binary).decode("ascii")
            return result
        else:  # IMAGE_URL
            result = {"type": "image_url", "content": self.content or ""}
            if self.binary:
                result["binary"] = base64.b64encode(self.binary).decode("ascii")
            return result


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

    def to_dict(
        self,
        output_format: OutputFormat = OutputFormat.OUMI,
    ) -> dict[str, Any]:
        """Serialize the message to a dictionary.

        Args:
            output_format: The serialization format to use.
                Defaults to OUMI (HuggingFace-compatible format).

        Returns:
            Dictionary representation of the message.
        """
        result: dict[str, Any] = {"role": str(self.role)}

        if self.id is not None:
            result["id"] = self.id

        # Handle content
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [
                item.to_dict(output_format=output_format) for item in self.content
            ]

        return result

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

    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
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

    def to_dict(
        self,
        output_format: OutputFormat = OutputFormat.OUMI,
    ) -> dict[str, Any]:
        """Converts the conversation to a dictionary.

        Args:
            output_format: The serialization format to use.
                Defaults to OUMI (HuggingFace-compatible format).

        Returns:
            Dictionary representation of the conversation.
        """
        result: dict[str, Any] = {}

        if self.conversation_id:
            result["conversation_id"] = self.conversation_id

        result["messages"] = [
            msg.to_dict(output_format=output_format) for msg in self.messages
        ]

        if self.metadata:
            result["metadata"] = self.metadata

        return result

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

    def to_json(
        self,
        output_format: OutputFormat = OutputFormat.OUMI,
    ) -> str:
        """Converts the conversation to a JSON string.

        Args:
            output_format: The serialization format to use.
                Defaults to OUMI (HuggingFace-compatible format).

        Returns:
            JSON string representation of the conversation.
        """
        import json

        return json.dumps(self.to_dict(output_format=output_format))

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
