import base64
import copy
import io
from pathlib import Path
from typing import Optional, Union

import PIL.Image
import requests

from oumi.core.types.conversation import Message, Type
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


def load_image_bytes_to_message(message: Message) -> Message:
    """Ensures that message contains inline image bytes if it's an image.

    Loads image content if image type is `IMAGE_URL` or `IMAGE_PATH`.
    Otherwise returns the input message w/o any changes.

    Args:
        message: An input message.

    Returns:
        A message guaranteed to be `IMAGE_BINARY` if an input message
        was any of image types (`IMAGE_URL`, `IMAGE_PATH`, `IMAGE_BINARY`).
    """
    if message.type in (Type.IMAGE_PATH, Type.IMAGE_URL):
        message = copy.deepcopy(message)
        if message.type == Type.IMAGE_PATH:
            if message.content is None:
                raise ValueError("Image path is None")
            png_bytes = load_image_png_bytes_from_path(message.content)
        else:
            assert message.type == Type.IMAGE_URL
            if message.content is None:
                raise ValueError("Image URL is None")
            try:
                response = requests.get(message.content, stream=True)
                response.raise_for_status()
            except requests.exceptions.RequestException:
                logger.exception(f"Failed to download image: '{message.content}'")
                raise
            png_bytes = create_png_bytes_from_image_bytes(response.content)

        message.type = Type.IMAGE_BINARY
        message.binary = png_bytes
        message.content = None
        return message

    return message


def base64encode_image_bytes(message: Message, *, add_mime_prefix: bool = True) -> str:
    """Creates base-64 encoded image bytes as ASCII string value.

    Args:
        message: An input message of image type
            (one of `IMAGE_BINARY`, `IMAGE_PATH, `IMAGE_URL`)
            with the pre-populated `binary` field.
        add_mime_prefix: Whether to add MIME prefix `data:image/png;base64,`

    Returns:
        String containing base64 encoded image bytes `<BASE64_VALUE>`.
        If `add_mime_prefix` is True, then the following format is used:
        `data:image/png;base64,<BASE64_VALUE>`.
    """
    if not message.is_image():
        raise ValueError(f"Message type is not an image: {message.type}")
    elif not message.binary:
        raise ValueError(f"No image bytes in message: {message.type}")

    base64_str = base64.b64encode(message.binary).decode(encoding="utf8")
    return ("data:image/png;base64," + base64_str) if add_mime_prefix else base64_str
