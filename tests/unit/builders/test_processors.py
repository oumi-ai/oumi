import base64
from typing import Any, Final, Optional

import numpy as np
import PIL.Image
import pytest
import torch
import transformers

from oumi.builders import build_chat_template, build_processor, build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import ContentItem, Message, Role, Type

_LLAVA_SYSTEM_PROMPT: Final[str] = (
    "A chat between a curious user and an artificial "
    "intelligence assistant. "
    "The assistant gives helpful, detailed, and "
    "polite answers to the user's questions."
)
_IMAGE_TOKEN: Final[str] = "<image>"
_IMAGE_TOKEN_ID: Final[int] = 32000

_SMALL_B64_IMAGE: Final[str] = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.parametrize(
    "trust_remote_code",
    [
        False,
        True,
    ],
)
def test_build_processor_empty_name(trust_remote_code, mock_tokenizer):
    with pytest.raises(ValueError, match="Empty model name"):
        build_processor("", mock_tokenizer, trust_remote_code=trust_remote_code)


@pytest.mark.parametrize(
    "processor_kwargs",
    [
        None,
        {},
    ],
)
def test_build_processor_basic_gpt2_success(
    processor_kwargs: Optional[dict[str, Any]], mock_tokenizer
):
    test_chat_template: Final[str] = build_chat_template(template_name="default")

    model_params = ModelParams(model_name="openai-community/gpt2")
    processor = build_processor(
        model_params.model_name,
        mock_tokenizer,
        trust_remote_code=False,
        processor_kwargs=processor_kwargs,
    )
    assert callable(processor)

    assert id(mock_tokenizer) == id(processor.tokenizer)
    processor.tokenizer = mock_tokenizer
    assert id(mock_tokenizer) == id(processor.tokenizer)
    assert processor.chat_template == test_chat_template

    processor.chat_template = test_chat_template + " "
    assert processor.chat_template == test_chat_template + " "
    processor.chat_template = test_chat_template
    assert processor.chat_template == test_chat_template

    assert processor.image_processor is None
    assert processor.image_token is None
    assert processor.image_token_id is None

    result = processor(text=["hello world"], padding=False)
    assert isinstance(result, transformers.BatchEncoding)
    assert len(result) == 2

    assert "input_ids" in result
    input_ids = result["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (1, 2)
    assert np.all(input_ids.numpy() == np.array([[31373, 995]]))

    assert "attention_mask" in result
    attention_mask = result["attention_mask"]
    assert isinstance(attention_mask, torch.Tensor)
    assert attention_mask.shape == (1, 2)
    assert np.all(attention_mask.numpy() == np.array([[1, 1]]))

    prompt = processor.apply_chat_template([Message(role=Role.USER, content="FooBazz")])
    assert isinstance(prompt, str)
    assert "FooBazz" in prompt
    assert prompt == "USER: FooBazz"

    prompt = processor.apply_chat_template(
        [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="How can I help?"),
            Message(role=Role.USER, content="Hmm"),
        ],
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str)
    assert prompt == "USER: Hello\nASSISTANT: How can I help?\nUSER: Hmm\nASSISTANT: "

    with pytest.raises(ValueError, match="Conversation includes non-text messages"):
        processor.apply_chat_template(
            [
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(type=Type.TEXT, content="Hello"),
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=base64.b64decode(_SMALL_B64_IMAGE),
                        ),
                    ],
                ),
                Message(role=Role.ASSISTANT, content="How can I help?"),
                Message(role=Role.USER, content="Hmm"),
            ]
        )


def test_build_processor_basic_multimodal_success():
    default_chat_template: Final[str] = build_chat_template(template_name="default")
    llava_chat_template: Final[str] = build_chat_template(template_name="llava")

    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="default"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=False
    )
    assert callable(processor)

    assert id(tokenizer) == id(processor.tokenizer)
    processor.tokenizer = tokenizer
    assert id(tokenizer) == id(processor.tokenizer)
    assert processor.chat_template
    assert processor.chat_template == default_chat_template
    processor.chat_template = llava_chat_template
    assert processor.chat_template == llava_chat_template

    assert processor.image_processor is not None
    assert processor.image_token == _IMAGE_TOKEN
    assert processor.image_token_id == _IMAGE_TOKEN_ID

    result = processor(text=["hello world"], padding=False)
    assert isinstance(result, transformers.BatchEncoding)
    assert len(result) == 2

    assert "input_ids" in result
    input_ids = result["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (1, 3)
    assert np.all(input_ids.numpy() == np.array([[1, 22172, 3186]]))

    assert "attention_mask" in result
    attention_mask = result["attention_mask"]
    assert isinstance(attention_mask, torch.Tensor)
    assert attention_mask.shape == (1, 3)
    assert np.all(attention_mask.numpy() == np.array([[1, 1, 1]]))

    prompt = processor.apply_chat_template([Message(role=Role.USER, content="FooBazz")])
    assert isinstance(prompt, str)
    assert "FooBazz" in prompt
    assert prompt == _LLAVA_SYSTEM_PROMPT + " USER: FooBazz "

    prompt = processor.apply_chat_template(
        [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="How can I help?"),
            Message(role=Role.USER, content="Hmm"),
        ],
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str)
    assert prompt == (
        _LLAVA_SYSTEM_PROMPT
        + " USER: Hello ASSISTANT: How can I help? </s>USER: Hmm ASSISTANT: "
    )

    test_image = PIL.Image.new(mode="RGB", size=(512, 256))
    result = processor(
        text=[prompt], images=[test_image], padding=True, return_tensors="pt"
    )
    assert isinstance(result, transformers.BatchEncoding)
    assert sorted(list(result.keys())) == [
        "attention_mask",
        "input_ids",
        "pixel_values",
    ]
    attention_mask = result["attention_mask"]
    assert isinstance(attention_mask, torch.Tensor)
    assert attention_mask.shape == (1, 57)

    input_ids = result["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (1, 57)

    pixel_values = result["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 3, 336, 336)

    image_proc_result = processor.image_processor(
        images=[test_image], return_tensors="pt"
    )
    assert isinstance(image_proc_result, transformers.BatchFeature)
    assert sorted(list(image_proc_result.keys())) == [
        "pixel_values",
    ]
    image_proc_pixel_values = result["pixel_values"]
    assert isinstance(image_proc_pixel_values, torch.Tensor)
    assert image_proc_pixel_values.shape == (1, 3, 336, 336)

    assert np.all(image_proc_pixel_values.numpy() == pixel_values.numpy())

    # Multiple prompts, Multiple images (different counts).
    result = processor(
        text=[prompt, prompt, prompt],
        images=[test_image, test_image],
        padding=True,
        return_tensors="pt",
    )
    assert isinstance(result, transformers.BatchEncoding)
    assert sorted(list(result.keys())) == [
        "attention_mask",
        "input_ids",
        "pixel_values",
    ]
    attention_mask = result["attention_mask"]
    assert isinstance(attention_mask, torch.Tensor)
    assert attention_mask.shape == (3, 57)

    input_ids = result["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (3, 57)

    pixel_values = result["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (2, 3, 336, 336)

    image_proc_result = processor.image_processor(
        images=[test_image, test_image], return_tensors="pt"
    )
    assert isinstance(image_proc_result, transformers.BatchFeature)
    assert sorted(list(image_proc_result.keys())) == [
        "pixel_values",
    ]
    image_proc_pixel_values = image_proc_result["pixel_values"]
    assert isinstance(image_proc_pixel_values, torch.Tensor)
    assert image_proc_pixel_values.shape == (2, 3, 336, 336)

    assert np.all(image_proc_pixel_values.numpy() == pixel_values.numpy())
