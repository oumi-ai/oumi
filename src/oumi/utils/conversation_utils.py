import base64
from typing import Any

import requests

from oumi.core.types.conversation import ContentItem, Type
from oumi.utils.image_utils import (
    create_png_bytes_from_image_bytes,
    load_image_png_bytes_from_path,
)
from oumi.utils.logging import logger


def load_image_bytes_to_content_item(item: ContentItem) -> ContentItem:
    """Ensures that message content item contains inline image bytes if it's an image.

    Loads image content if image type is `IMAGE_URL` or `IMAGE_PATH`.
    Otherwise returns the input content item w/o any changes.

    Args:
        item: An input message content item.

    Returns:
        A content item guaranteed to be `IMAGE_BINARY` if an input content item
        was any of image types (`IMAGE_URL`, `IMAGE_PATH`, `IMAGE_BINARY`).
    """
    if item.type in (Type.IMAGE_PATH, Type.IMAGE_URL):
        if item.type == Type.IMAGE_PATH:
            if item.content is None:
                raise ValueError("Image path is None")
            png_bytes = load_image_png_bytes_from_path(item.content)
        else:
            assert item.type == Type.IMAGE_URL
            if item.content is None:
                raise ValueError("Image URL is None")
            try:
                response = requests.get(item.content, stream=True)
                response.raise_for_status()
            except requests.exceptions.RequestException:
                logger.exception(f"Failed to download image: '{item.content}'")
                raise
            png_bytes = create_png_bytes_from_image_bytes(response.content)

        return ContentItem(type=Type.IMAGE_BINARY, binary=png_bytes)

    return item


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
