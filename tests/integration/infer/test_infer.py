from pathlib import Path
from typing import NamedTuple

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
from tests.integration.infer import get_default_device_map_for_inference
from tests.markers import requires_cuda_initialized, requires_gpus

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."


class InferTestSpec(NamedTuple):
    num_batches: int
    batch_size: int


def _get_infer_test_spec_id(x):
    assert isinstance(x, InferTestSpec)
    return f"batches={x.num_batches} bs={x.batch_size}"


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
def test_infer_basic_interactive_with_images(
    monkeypatch: pytest.MonkeyPatch, root_testdata_dir: Path
):
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
        root_testdata_dir / "images" / "the_great_wave_off_kanagawa.jpg"
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
    infer_interactive(config, input_image_bytes=[png_image_bytes])


@pytest.mark.parametrize(
    "test_spec",
    [
        InferTestSpec(num_batches=1, batch_size=1),
        InferTestSpec(num_batches=1, batch_size=2),
        InferTestSpec(num_batches=2, batch_size=1),
        InferTestSpec(num_batches=2, batch_size=2),
    ],
    ids=_get_infer_test_spec_id,
)
def test_infer_basic_non_interactive(test_spec: InferTestSpec):
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
        device_map=get_default_device_map_for_inference(),
    )
    generation_params = GenerationParams(
        max_new_tokens=5, temperature=0.0, seed=42, batch_size=test_spec.batch_size
    )

    input = [FIXED_PROMPT] * (test_spec.num_batches * test_spec.batch_size)
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
    expected_output = [conversation] * (test_spec.num_batches * test_spec.batch_size)
    assert output == expected_output


@requires_gpus()
@pytest.mark.parametrize(
    "test_spec",
    [
        InferTestSpec(num_batches=1, batch_size=1),
        InferTestSpec(num_batches=1, batch_size=2),
    ],
    ids=_get_infer_test_spec_id,
)
@pytest.mark.single_gpu
def test_infer_basic_non_interactive_with_images(
    test_spec: InferTestSpec, root_testdata_dir: Path
):
    model_params = ModelParams(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="qwen2-vl-instruct",
        torch_dtype_str="bfloat16",
        device_map=get_default_device_map_for_inference(),
    )
    generation_params = GenerationParams(
        max_new_tokens=10, temperature=0.0, seed=42, batch_size=test_spec.batch_size
    )

    png_image_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "the_great_wave_off_kanagawa.jpg"
    )

    test_prompt: str = "Generate a short, descriptive caption for this image!"

    input = [test_prompt] * (test_spec.num_batches * test_spec.batch_size)
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
        input_image_bytes=[png_image_bytes],
    )

    valid_responses = [
        "A detailed Japanese print depicting a large wave crashing with",
        "A traditional Japanese painting of a large wave crashing with",
        "A traditional Japanese ukiyo-e painting depicting a",
        "A detailed Japanese woodblock print depicting a large wave",
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
    assert len(output) == test_spec.num_batches * test_spec.batch_size
    for conv in output:
        assert any(
            conv == _create_conversation(response) for response in valid_responses
        ), f"Generated response '{conv.messages[-1].content}' not in valid responses"
