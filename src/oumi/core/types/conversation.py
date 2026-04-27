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
from collections.abc import Callable, Generator
from enum import Enum
from typing import Any, NamedTuple

import pydantic
from jinja2 import Template
from transformers.utils.chat_template_utils import get_json_schema


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

    def __str__(self) -> str:
        """Return the string representation of the Type enum.

        Returns:
            str: The string value of the Type enum.
        """
        return self.value


class FinishReason(str, Enum):
    """Reason why the model stopped generating tokens."""

    STOP = "stop"
    """Model hit a natural stopping point or stop sequence."""

    LENGTH = "length"
    """Model reached max_tokens limit."""

    TOOL_CALLS = "tool_calls"
    """Model made a tool/function call."""

    CONTENT_FILTER = "content_filter"
    """Content was filtered due to safety/moderation."""

    ERROR = "error"
    """Generation failed due to an error."""

    UNKNOWN = "unknown"
    """Finish reason could not be determined."""

    def __str__(self) -> str:
        """Return the string representation of the FinishReason enum.

        Returns:
            str: The string value of the FinishReason enum.
        """
        return self.value


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

    content: str | None = None
    """Optional text content of the content item.

    One of content or binary must be provided.
    """

    binary: bytes | None = None
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
    def _encode_binary(self, value: bytes | None) -> str:
        """Encode binary value as base64 ASCII string.

        This is needed for compatibility with JSON.
        """
        if value is None or len(value) == 0:
            return ""
        return base64.b64encode(value).decode("ascii")

    @pydantic.field_validator("binary", mode="before")
    def _decode_binary(cls, value: str | bytes | None) -> bytes | None:
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

    def __repr__(self) -> str:
        """Returns a string representation of the message item."""
        return f"{self.content}" if self.is_text() else f"<{self.type.upper()}>"


class Message(pydantic.BaseModel):
    """A message in a conversation.

    This class represents a single message within a conversation, containing
    various attributes such as role, content, identifier.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str | None = None
    """Optional unique identifier for the message.

    This attribute can be used to assign a specific identifier to the message,
    which may be useful for tracking or referencing messages within a conversation.

    Returns:
        Optional[str]: The unique identifier of the message, if set; otherwise None.
    """

    content: str | list[ContentItem] | None = None
    """Content of the message.

    For text messages, `content` can be set to a string value.
    For multimodal messages, `content` should be a list of content items of
    potentially different types e.g., text and image.
    May be ``None`` on assistant messages that only contain ``tool_calls``
    (OpenAI tool-calling wire format).
    """

    role: Role
    """The role of the entity sending the message (e.g., user, assistant, system)."""

    tool_calls: list[dict] | None = None
    """Structured tool calls emitted by an assistant message.

    Uses the OpenAI function-calling wire format, e.g.::

        [{"id": "call_abc", "type": "function",
          "function": {"name": "get_weather", "arguments": "{...}"}}]

    Only set on assistant messages. The chat template renders these via
    ``tokenizer.apply_chat_template(messages, tools=...)``.
    """

    tool_call_id: str | None = None
    """Identifier linking a tool response message to the call it responds to.

    Only set on messages with ``role == 'tool'``, matching the ``id`` of a
    prior assistant ``tool_calls`` entry.
    """

    def model_post_init(self, __context) -> None:
        """Post-initialization method for the Message model.

        This method is automatically called after the model is initialized.
        It validates that the message has at least one of ``content`` or
        ``tool_calls`` and that ``content``, when set, is a string or list.

        Raises:
            ValueError: If both ``content`` and ``tool_calls`` are ``None``,
                or if ``content`` has an unsupported type.
        """
        if self.content is None and not self.tool_calls:
            raise ValueError(
                "Message must have at least one of `content` or `tool_calls`."
            )
        if self.content is not None and not isinstance(self.content, str | list):
            raise ValueError(
                f"Unexpected content type: {type(self.content)}. "
                f"Must be a Python string, a list, or None."
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

    conversation_id: str | None = None
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

    tools: list[dict | Callable] | None = None
    """Tool definitions available to the model for this conversation.

    Accepts either OpenAI-format dicts or Python callables. Callables are
    converted to dicts at validation time via
    ``transformers.utils.chat_template_utils.get_json_schema``, which requires
    the callable to have a Google-style docstring and type hints on every
    user-facing argument. After validation, every entry is a dict regardless
    of the input shape, so persistence (``to_dict()``) and the HF Dataset
    schema are unaffected.

    Uses the OpenAI function-calling schema, e.g.::

        [{"type": "function",
          "function": {"name": "get_weather",
                       "description": "...",
                       "parameters": {...}}}]

    When set, ``to_dict()`` emits ``tools`` as a top-level key, which TRL's
    ``SFTTrainer`` forwards to ``tokenizer.apply_chat_template(messages, tools=...)``
    so the model's chat template can render tools natively.
    """

    @pydantic.field_validator("tools", mode="before")
    @classmethod
    def _coerce_callable_tools_to_openai_schema(
        cls, raw_tools: list[dict | Callable] | None
    ) -> list[dict] | None:
        """Convert any callable tool definitions to OpenAI-format dicts.

        Pydantic validators run on the input value before final field
        assignment. We accept callables for ergonomics in inference / agent
        flows but normalise to dicts so storage stays uniform.
        """
        if raw_tools is None:
            return None
        coerced_tools: list[dict] = []
        for tool in raw_tools:
            if isinstance(tool, dict):
                coerced_tools.append(tool)
            elif callable(tool):
                coerced_tools.append(get_json_schema(tool))
            else:
                raise ValueError(
                    f"Each tool must be a dict (OpenAI schema) or a callable; "
                    f"got {type(tool).__name__}."
                )
        return coerced_tools

    def __getitem__(self, idx: int) -> Message:
        """Gets the message at the specified index.

        Args:
            idx (int): The index of the message to retrieve.

        Returns:
            Any: The message at the specified index.
        """
        return self.messages[idx]

    def first_message(self, role: Role | None = None) -> Message | None:
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

    def last_message(self, role: Role | None = None) -> Message | None:
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
        role: Role | None = None,
        filter_fn: Callable[[Message], bool] | None = None,
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
        data = self.model_dump(
            mode="json", exclude_unset=True, exclude_defaults=False, exclude_none=True
        )
        # Preserve `content: null` on assistant messages that only carry
        # `tool_calls`, matching the OpenAI wire format. `exclude_none=True`
        # strips `None` globally, which would drop the `content` key and
        # trip HF chat templates that don't tolerate a missing `content`.
        for msg_dict, msg in zip(data.get("messages", []), self.messages):
            if msg.tool_calls and "content" not in msg_dict:
                msg_dict["content"] = None
        return data

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
