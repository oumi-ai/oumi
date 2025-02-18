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

import io
from importlib.util import find_spec
from pathlib import Path
from typing import Final, Optional, Union

import PIL.Image
import requests

from oumi.utils.logging import logger

# For details on image modes, see
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
DEFAULT_IMAGE_MODE: Final[str] = "RGB"

_FILE_URL_PREFIX: Final[str] = "file://"
_DEFAULT_PDF_DPI: Final[int] = 200


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


def create_png_bytes_from_image_list(pil_images: list[PIL.Image.Image]) -> list[bytes]:
    """Encodes PIL images into PNG format, and returns PNG image bytes.

    Args:
        pil_images: A list of input images.

    Returns:
        A list of PNG-encoded images.
    """
    return [create_png_bytes_from_image(image) for image in pil_images]


def convert_pil_image_mode(
    image: PIL.Image.Image, *, mode: Optional[str]
) -> PIL.Image.Image:
    """Converts a PIL image to the requested mode (if it's not in that mode already) .

    Args:
        image: An input image.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        An image in the requested mode . If an input image was already in the correct
        mode then return it for efficiency.
        Otherwise, a different image object is returned.
    """
    if not mode or image.mode == mode:
        # Return the original object for better performance.
        return image

    old_mode = image.mode
    try:
        return image.convert(mode)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert an image from {old_mode} to {mode} mode!"
        ) from e


def load_pil_image_from_path(
    input_image_filepath: Union[str, Path], mode: str = DEFAULT_IMAGE_MODE
) -> PIL.Image.Image:
    """Loads an image from a path.

    Args:
        input_image_filepath: A file path of an image.
            The image can be in any format supported by PIL.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        bytes: PNG bytes representation of the image.
    """
    if not input_image_filepath:
        raise ValueError("Empty image file path.")

    if isinstance(
        input_image_filepath, str
    ) and input_image_filepath.lower().startswith(_FILE_URL_PREFIX):
        input_image_filepath = input_image_filepath[len(_FILE_URL_PREFIX) :]

    input_image_filepath = Path(input_image_filepath)
    if not input_image_filepath.is_file():
        raise ValueError(
            f"Image path is not a file: {input_image_filepath}"
            if input_image_filepath.exists()
            else f"Image path doesn't exist: {input_image_filepath}"
        )

    try:
        pil_image = convert_pil_image_mode(
            PIL.Image.open(input_image_filepath), mode=mode
        )
    except Exception:
        logger.error(f"Failed to load an image from path: {input_image_filepath}")
        raise
    return pil_image


def load_pil_image_from_url(
    input_image_url: str, mode: str = DEFAULT_IMAGE_MODE
) -> PIL.Image.Image:
    """Loads a PIL image from a URL.

    Args:
        input_image_url: An image URL.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        bytes: PNG bytes representation of the image.
    """
    if not input_image_url:
        raise ValueError("Empty image URL!")

    try:
        response = requests.get(input_image_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        logger.exception(f"Failed to download image: '{input_image_url}'")
        raise
    return load_pil_image_from_bytes(response.content, mode=mode)


def load_pil_image_from_bytes(
    image_bytes: Optional[bytes], mode: str = DEFAULT_IMAGE_MODE
) -> PIL.Image.Image:
    """Loads an image from raw image bytes.

    Args:
        image_bytes: A input image bytes. Can be in any image format supported by PIL.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        PIL.Image.Image: PIL representation of the image.
    """
    if image_bytes is None or len(image_bytes) == 0:
        raise ValueError("No image bytes.")

    try:
        pil_image = convert_pil_image_mode(
            PIL.Image.open(io.BytesIO(image_bytes)), mode=mode
        )
    except Exception:
        logger.error(
            f"Failed to load an image from raw image bytes ({len(image_bytes)} bytes)."
        )
        raise
    return pil_image


def _check_pdf2image_dependency():
    if not find_spec("pdf2image"):
        raise RuntimeError(
            "Failed to find the required dependency package: 'pdf2image'. "
            "Run `pip install oumi[file_formats]`, and try again."
        )


def load_pdf_pages_from_path(
    input_pdf_filepath: Union[str, Path],
    *,
    dpi: int = _DEFAULT_PDF_DPI,
    mode: str = DEFAULT_IMAGE_MODE,
) -> list[PIL.Image.Image]:
    """Loads PDF pages as PIL images from a path.

    Args:
        input_pdf_filepath: A file path of an PDF document.
        dpi: Resolution to use for PDF page images (dots per inch).
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        PDF pages as PIL images (PIL.Image.Image).
    """
    if not input_pdf_filepath:
        raise ValueError("Empty PDF file path.")

    if isinstance(input_pdf_filepath, str) and input_pdf_filepath.lower().startswith(
        _FILE_URL_PREFIX
    ):
        input_pdf_filepath = input_pdf_filepath[len(_FILE_URL_PREFIX) :]

    input_filepath = Path(input_pdf_filepath)
    if not input_filepath.is_file():
        raise ValueError(
            f"PDF path is not a file: {input_filepath}"
            if input_filepath.exists()
            else f"PDF path doesn't exist: {input_filepath}"
        )

    _check_pdf2image_dependency()
    import pdf2image  # pyright: ignore[reportMissingImports]

    page_images = pdf2image.convert_from_path(input_filepath, dpi=dpi)
    num_pages = len(page_images)
    for page_idx in range(num_pages):
        try:
            page_images[page_idx] = convert_pil_image_mode(
                page_images[page_idx], mode=mode
            )
        except Exception:
            logger.error(
                "Failed to convert image mode for PDF page "
                f"{page_idx + 1} of {num_pages}: {input_filepath}"
            )
            raise
    return page_images


def load_pdf_pages_from_bytes(
    pdf_bytes: Optional[bytes],
    *,
    dpi: int = _DEFAULT_PDF_DPI,
    mode: str = DEFAULT_IMAGE_MODE,
) -> list[PIL.Image.Image]:
    """Loads PDF pages as PIL images from raw PDF file bytes.

    Args:
        pdf_bytes: PDF file content.
        dpi: Resolution to use for PDF page images (dots per inch).
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        PDF pages as PIL images (PIL.Image.Image).
    """
    if pdf_bytes is None or len(pdf_bytes) == 0:
        raise ValueError("No PDF bytes.")

    _check_pdf2image_dependency()
    import pdf2image  # pyright: ignore[reportMissingImports]

    page_images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi)
    num_pages = len(page_images)
    for page_idx in range(num_pages):
        try:
            page_images[page_idx] = convert_pil_image_mode(
                page_images[page_idx], mode=mode
            )
        except Exception:
            logger.error(
                "Failed to convert image mode for PDF page "
                f"{page_idx + 1} or {num_pages}"
            )
            raise
    return page_images


def load_pdf_pages_from_url(
    pdf_url: str, *, dpi: int = _DEFAULT_PDF_DPI, mode: str = DEFAULT_IMAGE_MODE
) -> list[PIL.Image.Image]:
    """Loads PDF pages as PIL images from from PDF URL.

    Args:
        pdf_url: A PDF URL.
        dpi: Resolution to use for PDF page images (dots per inch).
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        PDF pages as PIL images (PIL.Image.Image).
    """
    if not pdf_url:
        raise ValueError("Empty PDF URL!")

    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        logger.exception(f"Failed to download PDF: '{pdf_url}'")
        raise
    return load_pdf_pages_from_bytes(response.content, dpi=dpi, mode=mode)


def create_png_bytes_from_image_bytes(
    image_bytes: Optional[bytes], mode: str = DEFAULT_IMAGE_MODE
) -> bytes:
    """Loads an image from raw image bytes, and converts to PNG image bytes.

    Args:
        image_bytes: A input image bytes. Can be in any image format supported by PIL.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        bytes: PNG bytes representation of the image.
    """
    pil_image = load_pil_image_from_bytes(image_bytes, mode=mode)
    return create_png_bytes_from_image(pil_image)


def load_image_png_bytes_from_path(
    input_image_filepath: Union[str, Path], mode: str = DEFAULT_IMAGE_MODE
) -> bytes:
    """Loads an image from a path, converts it to PNG, and returns image bytes.

    Args:
        input_image_filepath: A file path of an image.
            The image can be in any format supported by PIL.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        bytes: PNG bytes representation of the image.
    """
    pil_image = load_pil_image_from_path(input_image_filepath, mode=mode)
    return create_png_bytes_from_image(pil_image)


def load_image_png_bytes_from_url(
    input_image_url: str, mode: str = DEFAULT_IMAGE_MODE
) -> bytes:
    """Loads an image from a URL, converts it to PNG, and returns image bytes.

    Args:
        input_image_url: An image URL.
        mode: The requested image mode e.g., "RGB", "HSV", "RGBA",
            "P" (8-bit pixels, using a color palette).
            For details, see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes

    Returns:
        bytes: PNG bytes representation of the image.
    """
    pil_image = load_pil_image_from_url(input_image_url, mode=mode)
    return create_png_bytes_from_image(pil_image)
