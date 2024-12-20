import base64
import io
from pathlib import Path
from typing import Optional, Union

import PIL.Image
import requests

from oumi.core.types.conversation import ContentItem, Type
from oumi.utils.logging import logger


def create_png_bytes_from_image(pil_image: PIL.Image.Image) -> bytes:
    """Encodes PIL image into PNG format, and returns PNG image bytes.

    Args:
        pil_image: An input image.

    Returns:
        bytes: PNG bytes representation of the image.
    """
    try:
        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        logger.error("Failed to convert an image to PNG bytes.")
        raise


def load_image_png_bytes_from_path(input_image_filepath: Union[str, Path]) -> bytes:
    """Loads an image from a path, converts it to PNG, and returns image bytes.

    Args:
        input_image_filepath: A file path of an image.
            The image can be in any format supported by PIL.

    Returns:
        bytes: PNG bytes representation of the image.
    """
    if not input_image_filepath:
        raise ValueError("Empty image file path.")
    input_image_filepath = Path(input_image_filepath)
    if not input_image_filepath.is_file():
        raise ValueError(
            f"Image path is not a file: {input_image_filepath}"
            if input_image_filepath.exists()
            else f"Image path doesn't exist: {input_image_filepath}"
        )

    try:
        pil_image = PIL.Image.open(input_image_filepath).convert("RGB")
    except Exception:
        logger.error(f"Failed to load an image from path: {input_image_filepath}")
        raise

    return create_png_bytes_from_image(pil_image)


def load_image_from_bytes(image_bytes: Optional[bytes]) -> PIL.Image.Image:
    """Loads an image from raw image bytes.

    Args:
        image_bytes: A input image bytes. Can be in any image format supported by PIL.

    Returns:
        PIL.Image.Image: PIL representation of the image.
    """
    if image_bytes is None or len(image_bytes) == 0:
        raise ValueError("No image bytes.")

    try:
        pil_image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.error(
            f"Failed to load an image from raw image bytes ({len(image_bytes)} bytes)."
        )
        raise
    return pil_image


def create_png_bytes_from_image_bytes(image_bytes: Optional[bytes]) -> bytes:
    """Loads an image from raw image bytes, and converts to PNG image bytes.

    Args:
        image_bytes: A input image bytes. Can be in any image format supported by PIL.

    Returns:
        bytes: PNG bytes representation of the image.
    """
    pil_image = load_image_from_bytes(image_bytes)
    return create_png_bytes_from_image(pil_image)


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


def base64encode_image_bytes(item: ContentItem, *, add_mime_prefix: bool = True) -> str:
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
