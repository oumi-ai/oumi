import io
import tempfile
from pathlib import Path

import PIL.Image
import pytest

from oumi.utils.image_utils import (
    convert_pil_image_mode,
    create_png_bytes_from_image,
    create_png_bytes_from_image_bytes,
    load_image_png_bytes_from_path,
    load_pil_image_from_bytes,
)


def _create_jpg_bytes_from_image(pil_image: PIL.Image.Image) -> bytes:
    output = io.BytesIO()
    pil_image.save(output, format="JPEG")
    return output.getvalue()


def test_load_image_from_empty_bytes():
    with pytest.raises(ValueError, match="No image bytes"):
        load_pil_image_from_bytes(None)

    with pytest.raises(ValueError, match="No image bytes"):
        load_pil_image_from_bytes(b"")


def test_load_image_from_bytes():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    assert len(png_bytes) > 50

    pil_image_reloaded = load_pil_image_from_bytes(png_bytes)
    assert pil_image_reloaded.size == pil_image.size


def test_create_png_bytes_from_image_bytes():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    jpg_bytes = _create_jpg_bytes_from_image(pil_image)
    assert len(jpg_bytes) > 50

    png_bytes = create_png_bytes_from_image_bytes(jpg_bytes)

    pil_image_reloaded = load_pil_image_from_bytes(png_bytes)
    assert pil_image_reloaded.size == pil_image.size


def test_ensure_pil_image_mode():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    assert id(convert_pil_image_mode(pil_image, mode="RGB")) == id(pil_image)
    assert id(convert_pil_image_mode(pil_image, mode="")) == id(pil_image)

    pil_image_rgba = convert_pil_image_mode(pil_image, mode="RGBA")
    assert id(pil_image_rgba) != id(pil_image)
    assert pil_image_rgba.size == pil_image.size
    assert id(convert_pil_image_mode(pil_image_rgba, mode="")) == id(pil_image_rgba)

    pil_image_rgb2 = convert_pil_image_mode(pil_image_rgba, mode="RGB")
    assert id(pil_image_rgb2) != id(pil_image)
    assert pil_image_rgb2.size == pil_image.size


def test_load_image_png_bytes_from_empty_path():
    with pytest.raises(ValueError, match="Empty image file path"):
        load_image_png_bytes_from_path("")


def test_load_image_png_bytes_from_dir():
    with pytest.raises(ValueError, match="Image path is not a file"):
        load_image_png_bytes_from_path(Path())

    with tempfile.TemporaryDirectory() as output_temp_dir:
        with pytest.raises(ValueError, match="Image path is not a file"):
            load_image_png_bytes_from_path(output_temp_dir)
        with pytest.raises(ValueError, match="Image path is not a file"):
            load_image_png_bytes_from_path(Path(output_temp_dir))


def test_load_image_png_bytes_from_path():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    assert len(png_bytes) > 50

    with tempfile.TemporaryDirectory() as output_temp_dir:
        png_filename: Path = Path(output_temp_dir) / "test.png"
        with png_filename.open(mode="wb") as f:
            f.write(png_bytes)

        loaded_png_bytes1 = load_image_png_bytes_from_path(png_filename)
        assert len(loaded_png_bytes1) > 50

        loaded_png_bytes2 = load_image_png_bytes_from_path(Path(png_filename))
        assert loaded_png_bytes1 == loaded_png_bytes2

        loaded_png_bytes2 = load_image_png_bytes_from_path(str(png_filename))
        assert loaded_png_bytes1 == loaded_png_bytes2

        loaded_png_bytes2 = load_image_png_bytes_from_path(f"file://{png_filename}")
        assert loaded_png_bytes1 == loaded_png_bytes2
