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
from typing import Any, Union

import PIL.Image

from oumi.core.types.conversation import ContentItem, Message, Type
from oumi.utils.image_utils import (
    DEFAULT_IMAGE_MODE,
    load_image_png_bytes_from_path,
    load_image_png_bytes_from_url,
    load_pil_image_from_bytes,
    load_pil_image_from_path,
    load_pil_image_from_url,
)


def load_image_bytes_to_content_item(
    item: ContentItem, mode: str = DEFAULT_IMAGE_MODE
) -> ContentItem:
    """Ensures that message content item contains inline image bytes if it's an image.

    Loads image content if image type is `IMAGE_URL` or `IMAGE_PATH`.
    Otherwise returns the input content item w/o any changes.

    Args:
        item: An input message content item.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        A content item guaranteed to be `IMAGE_BINARY` if an input content item
        was any of image types (`IMAGE_URL`, `IMAGE_PATH`, `IMAGE_BINARY`).
    """
    if item.type in (Type.IMAGE_PATH, Type.IMAGE_URL):
        if item.type == Type.IMAGE_PATH:
            if item.content is None:
                raise ValueError("Image path is None")
            png_bytes = load_image_png_bytes_from_path(item.content, mode=mode)
        else:
            assert item.type == Type.IMAGE_URL
            if item.content is None:
                raise ValueError("Image URL is None")
            png_bytes = load_image_png_bytes_from_url(item.content, mode=mode)

        return ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes)

    return item


def load_pil_image_from_content_item(
    image_item: ContentItem, mode: str = DEFAULT_IMAGE_MODE
) -> PIL.Image.Image:
    """Loads a PIL image from a message content item.

    Args:
        image_item: A content item representing an image.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        Image.Image: A PIL image.
    """
    if image_item.type == Type.IMAGE_PATH:
        if image_item.content is None:
            raise ValueError("Image path is None")
        image_bin = load_pil_image_from_path(image_item.content, mode=mode)
    elif image_item.type == Type.IMAGE_URL:
        if image_item.content is None:
            raise ValueError("Image URL is None")
        image_bin = load_pil_image_from_url(image_item.content, mode=mode)
    elif image_item.type == Type.IMAGE_BINARY:
        if image_item.binary is None:
            raise ValueError("Image binary is None")
        image_bin = load_pil_image_from_bytes(image_item.binary, mode=mode)
    else:
        raise ValueError(
            f"Unsupported content item type: {image_item.type}. Not an image!"
        )

    return image_bin


def base64encode_content_item_image_bytes(
    item: ContentItem, *, add_mime_prefix: bool = True
) -> str:
    """Creates base-64 encoded image bytes as ASCII string value.

    Args:
        item: An input message content item of image type
            (one of `IMAGE_BINARY`, `IMAGE_PATH, `IMAGE_URL`)
            with the pre-populated `binary` field.
        add_mime_prefix: Whether to add MIME prefix `data:image/png;base64,`

    Returns:
        String containing base64 encoded image bytes `<BASE64_VALUE>`.
        If `add_mime_prefix` is True, then the following format is used:
        `data:image/png;base64,<BASE64_VALUE>`.
    """
    if not item.is_image():
        raise ValueError(f"Message type is not an image: {item.type}")
    elif not item.binary:
        raise ValueError(f"No image bytes in message: {item.type}")

    base64_str = base64.b64encode(item.binary).decode(encoding="utf8")
    return ("data:image/png;base64," + base64_str) if add_mime_prefix else base64_str


_JSON_DICT_KEY_TYPE: str = "type"
_JSON_DICT_KEY_TEXT: str = "text"
_JSON_DICT_KEY_IMAGE_URL: str = "image_url"
_JSON_DICT_KEY_URL: str = "url"


def convert_message_content_item_to_json_dict(
    item: ContentItem,
) -> dict[str, Any]:
    """Returns the content for a message content item.

    Args:
        item: The message content item to get the content for.

    Returns:
        Dict[str, Any]: The content for the message.
    """
    if item.type == Type.TEXT:
        return {
            _JSON_DICT_KEY_TYPE: Type.TEXT.value,
            _JSON_DICT_KEY_TEXT: (item.content or ""),
        }
    elif not item.is_image():
        raise ValueError(f"Unsupported message type: {item.type}")

    if not item.binary and item.type != Type.IMAGE_URL:
        item = load_image_bytes_to_content_item(item)

    if item.binary:
        b64_image = base64encode_content_item_image_bytes(item, add_mime_prefix=True)
        return {
            _JSON_DICT_KEY_TYPE: Type.IMAGE_URL.value,
            _JSON_DICT_KEY_IMAGE_URL: {_JSON_DICT_KEY_URL: b64_image},
        }

    assert (
        item.type == Type.IMAGE_URL
    ), f"Unexpected message type: {item.type}. Must be a code bug."
    return {
        _JSON_DICT_KEY_TYPE: Type.IMAGE_URL.value,
        _JSON_DICT_KEY_IMAGE_URL: {_JSON_DICT_KEY_URL: item.content or ""},
    }


def convert_content_items_to_json_list(
    content_items: list[ContentItem],
) -> list[dict[str, Any]]:
    """Converts content items to a list of JSON dicts.

    Args:
        content_items: A list of content items.

    Returns:
        list[Dict[str, Any]]: The list of all content items encoded as JSON dicts.
    """
    return [convert_message_content_item_to_json_dict(item) for item in content_items]


def convert_message_to_json_content_list(
    message: Message,
) -> list[dict[str, Any]]:
    """Returns the message content as a list of its content items encoded as JSON dicts.

    Args:
        message: The message to get the content for.

    Returns:
        list[Dict[str, Any]]: The content for the message for all content items.
    """
    return convert_content_items_to_json_list(message.content_items)


def convert_message_to_json_content(
    message: Message,
) -> Union[str, list[dict[str, Any]]]:
    """Returns the message content.

    Args:
        message: The message to get the content for.

    Returns:
        The content for the message returned either as a single string,
        or as a list of content items.
    """
    if isinstance(message.content, str):
        return message.content

    assert isinstance(message.content, list)
    return convert_content_items_to_json_list(message.content_items)


def create_list_of_message_json_dicts(
    messages: list[Message],
    *,
    group_adjacent_same_role_turns: bool,
) -> list[dict[str, Any]]:
    """Returns a list of JSON dictionaries representing messages.

    Loads image bytes and encodes them as base64.

    Args:
        messages: The input messages.
        group_adjacent_same_role_turns: Whether to pack adjacent messages
            from the same role into a single element in output list.

    Returns:
        list[Dict[str, Any]]: The list of messages encoded as nested JSON dicts.
    """
    num_messages = len(messages)
    result = []
    idx = 0
    while idx < num_messages:
        end_idx = idx + 1
        if group_adjacent_same_role_turns:
            while end_idx < num_messages and (
                messages[idx].role == messages[end_idx].role
            ):
                end_idx += 1

        item: dict[str, Any] = {
            "role": messages[idx].role.value,
        }
        group_size = end_idx - idx
        if group_size == 1 and messages[idx].contains_single_text_content_item_only():
            # Set "content" to a primitive string value, which is the common
            # convention for text-only models.
            item["content"] = messages[idx].text_content_items[0].content
        else:
            # Set "content" to be a list of dictionaries for more complex cases.
            content_list = []
            while idx < end_idx:
                content_list.extend(convert_message_to_json_content_list(messages[idx]))
                idx += 1
            item["content"] = content_list

        idx = end_idx
        result.append(item)

    return result
