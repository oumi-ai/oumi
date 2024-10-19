import io
from pathlib import Path
from typing import Optional, Union

import PIL.Image

from oumi.utils.logging import logger


def load_image_png_bytes(input_image_filepath: Union[str, Path]) -> bytes:
    """Loads an image from a path, converts it to PNG, and returns image bytes."""
    try:
        pil_image = PIL.Image.open(input_image_filepath).convert("RGB")

        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        logger.error(f"Failed to load an image from path: {input_image_filepath}")
        raise


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
