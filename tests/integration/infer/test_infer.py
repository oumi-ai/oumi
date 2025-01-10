from pathlib import Path
from typing import Final

import pytest

from oumi import infer, infer_interactive
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.image_utils import load_image_png_bytes_from_path
from oumi.utils.io_utils import get_oumi_root_directory
from tests.markers import requires_cuda_initialized, requires_gpus

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."

TEST_IMAGE_DIR: Final[Path] = (
    get_oumi_root_directory().parent.parent.resolve() / "tests" / "testdata" / "images"
)


@requires_cuda_initialized()
@requires_gpus()
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


@requires_cuda_initialized()
@requires_gpus()
@pytest.mark.skip(reason="TODO: this test takes too long to run")
def test_infer_basic_interactive_with_images(monkeypatch: pytest.MonkeyPatch):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            model_max_length=1024,
            trust_remote_code=True,
            chat_template="qwen2-vl-instruct",
        ),
        generation=GenerationParams(max_new_tokens=16, temperature=0.0, seed=42),
    )

    png_image_bytes = load_image_png_bytes_from_path(
        TEST_IMAGE_DIR / "the_great_wave_off_kanagawa.jpg"
    )

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


@requires_gpus()
@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2)])
def test_infer_basic_non_interactive_with_images(num_batches, batch_size):
    model_params = ModelParams(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="qwen2-vl-instruct",
        torch_dtype_str="float16",
    )
    generation_params = GenerationParams(
        max_new_tokens=10, temperature=0.0, seed=42, batch_size=batch_size
    )

    png_image_bytes = load_image_png_bytes_from_path(
        TEST_IMAGE_DIR / "the_great_wave_off_kanagawa.jpg"
    )

    test_prompt: str = "Generate a short, descriptive caption for this image!"

    input = [test_prompt] * (num_batches * batch_size)
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
        input_image_bytes=png_image_bytes,
    )

    valid_responses = [
        "A detailed Japanese print depicting a large wave crashing with",
        "A traditional Japanese painting of a large wave crashing with",
    ]

    def _create_conversation(response: str) -> Conversation:
        return Conversation(
            messages=(
                [
                    Message(
                        role=Role.USER,
                        content=[
                            ContentItem(binary=png_image_bytes, type=Type.IMAGE_BINARY),
                            ContentItem(
                                content=test_prompt,
                                type=Type.TEXT,
                            ),
                        ],
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=response,
                    ),
                ]
            )
        )

    # Check that each output conversation matches one of the valid responses
    assert len(output) == num_batches * batch_size
    for conv in output:
        assert any(
            conv == _create_conversation(response) for response in valid_responses
        ), f"Generated response '{conv.messages[-1].content}' not in valid responses"
