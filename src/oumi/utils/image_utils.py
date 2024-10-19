import io
from pathlib import Path
from typing import Optional, Union

import PIL.Image

from oumi.utils.logging import logger


def create_png_bytes_from_image(pil_image: PIL.Image.Image) -> bytes:
    """Returns image bytes in PNG format."""
    try:
        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        logger.error("Failed to convert an image to PNG bytes.")
        raise


def load_image_png_bytes_from_path(input_image_filepath: Union[str, Path]) -> bytes:
    """Loads an image from a path, converts it to PNG, and returns image bytes."""
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
    """Loads an image from raw image bytes."""
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
