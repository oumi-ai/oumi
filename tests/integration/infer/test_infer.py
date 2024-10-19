from pathlib import Path
from typing import Final

import pytest
import torch

from oumi import infer, infer_interactive
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.image_utils import load_image_png_bytes_from_path

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."

OUMI_ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent
TEST_IMAGE_DIR: Final[Path] = OUMI_ROOT_DIR / "tests" / "testdata" / "images"


def is_cuda_available_and_initialized():
    if not torch.cuda.is_available():
        return False
    if not torch.cuda.is_initialized():
        torch.cuda.init()
    return torch.cuda.is_initialized()


def test_infer_basic_interactive(monkeypatch: pytest.MonkeyPatch):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
            chat_template="gpt2",
            tokenizer_pad_token="<|endoftext|>",
        ),
        generation=GenerationParams(max_new_tokens=5, temperature=0.0, seed=42),
    )

    # Simulate the user entering "Hello world!" in the terminal folowed by Ctrl+D.
    input_iterator = iter([FIXED_PROMPT])

    def mock_input(_):
        try:
            return next(input_iterator)
        except StopIteration:
            raise EOFError  # Simulate Ctrl+D

    # Replace the built-in input function
    monkeypatch.setattr("builtins.input", mock_input)
    infer_interactive(config)


@pytest.mark.skipif(
    not is_cuda_available_and_initialized(),
    reason="CUDA is not available",
)
def test_infer_basic_interactive_with_images(monkeypatch: pytest.MonkeyPatch):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="llava-hf/llava-1.5-7b-hf",
            model_max_length=1024,
            trust_remote_code=True,
            chat_template="llava",
        ),
        generation=GenerationParams(max_new_tokens=16, temperature=0.0, seed=42),
    )

    png_image_bytes = load_image_png_bytes_from_path(TEST_IMAGE_DIR / "cambrian.png")

    # Simulate the user entering "Hello world!" in the terminal folowed by Ctrl+D.
    input_iterator = iter(["Describe the image!"])

    def mock_input(_):
        try:
            return next(input_iterator)
        except StopIteration:
            raise EOFError  # Simulate Ctrl+D

    # Replace the built-in input function
    monkeypatch.setattr("builtins.input", mock_input)
    infer_interactive(config, input_image_bytes=png_image_bytes)


@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_infer_basic_non_interactive(num_batches, batch_size):
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
    )
    generation_params = GenerationParams(
        max_new_tokens=5, temperature=0.0, seed=42, batch_size=batch_size
    )

    input = [FIXED_PROMPT] * (num_batches * batch_size)
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
    )

    conversation = Conversation(
        messages=(
            [
                Message(content=FIXED_PROMPT, role=Role.USER),
                Message(content=FIXED_RESPONSE, role=Role.ASSISTANT),
            ]
        )
    )
    expected_output = [conversation] * (num_batches * batch_size)
    assert output == expected_output


@pytest.mark.skipif(
    not is_cuda_available_and_initialized(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2)])
def test_infer_basic_non_interactive_with_images(num_batches, batch_size):
    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="llava",
    )
    generation_params = GenerationParams(
        max_new_tokens=10, temperature=0.0, seed=42, batch_size=batch_size
    )

    png_image_bytes = load_image_png_bytes_from_path(TEST_IMAGE_DIR / "cambrian.png")

    input = ["Describe the high-level theme of the image in few words!"] * (
        num_batches * batch_size
    )
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
        input_image_bytes=png_image_bytes,
    )

    conversation = Conversation(
        messages=(
            [
                Message(role=Role.USER, binary=png_image_bytes, type=Type.IMAGE_BINARY),
                Message(
                    role=Role.USER,
                    content="Describe the high-level theme of the image in few words!",
                    type=Type.TEXT,
                ),
                Message(
                    role=Role.ASSISTANT,
                    content="3D underwater scene with various sea creatures",
                    type=Type.TEXT,
                ),
            ]
        )
    )
    expected_output = [conversation] * (num_batches * batch_size)
    assert output == expected_output
