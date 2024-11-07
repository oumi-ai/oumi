import copy
import io
import tempfile
from pathlib import Path

import PIL.Image
import pytest

from oumi.core.types.conversation import Message, Role, Type
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
    create_png_bytes_from_image_bytes,
    load_image_bytes_to_message,
    load_image_from_bytes,
    load_image_png_bytes_from_path,
)


def _create_jpg_bytes_from_image(pil_image: PIL.Image.Image) -> bytes:
    output = io.BytesIO()
    pil_image.save(output, format="JPEG")
    return output.getvalue()


def test_load_image_from_empty_bytes():
    with pytest.raises(ValueError, match="No image bytes"):
        load_image_from_bytes(None)

    with pytest.raises(ValueError, match="No image bytes"):
        load_image_from_bytes(b"")


def test_load_image_from_bytes():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    assert len(png_bytes) > 50

    pil_image_reloaded = load_image_from_bytes(png_bytes)
    assert pil_image_reloaded.size == pil_image.size


def test_create_png_bytes_from_image_bytes():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    jpg_bytes = _create_jpg_bytes_from_image(pil_image)
    assert len(jpg_bytes) > 50

    png_bytes = create_png_bytes_from_image_bytes(jpg_bytes)

    pil_image_reloaded = load_image_from_bytes(png_bytes)
    assert pil_image_reloaded.size == pil_image.size


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


def test_load_image_bytes_to_message_noop_text():
    input_message = Message(role=Role.ASSISTANT, type=Type.TEXT, content="hello")
    saved_input_message = copy.deepcopy(input_message)

    output_message = load_image_bytes_to_message(input_message)
    assert id(output_message) == id(input_message)
    assert output_message == saved_input_message


def test_load_image_bytes_to_message_noop_image_binary():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    input_message = Message(
        role=Role.USER,
        type=Type.IMAGE_BINARY,
        binary=create_png_bytes_from_image(pil_image),
    )
    saved_input_message = copy.deepcopy(input_message)

    output_message = load_image_bytes_to_message(input_message)
    assert id(output_message) == id(input_message)
    assert output_message == saved_input_message


def test_load_image_bytes_to_message_noop_image_path():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)

    with tempfile.TemporaryDirectory() as output_temp_dir:
        png_filename: Path = Path(output_temp_dir) / "test.png"
        with png_filename.open(mode="wb") as f:
            f.write(png_bytes)

        input_message = Message(
            role=Role.USER, type=Type.IMAGE_PATH, content=str(png_filename)
        )

        output_message = load_image_bytes_to_message(input_message)
        assert id(output_message) != id(input_message)

        expected_output_message = Message(
            role=Role.USER, type=Type.IMAGE_BINARY, binary=png_bytes
        )
        assert output_message == expected_output_message
