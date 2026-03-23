import base64
from typing import Any, Final

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
    processor_kwargs: dict[str, Any] | None, mock_tokenizer
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


def test_processor_converts_messages_to_dict_internally():
    """Test DefaultProcessor converts Message objects to dicts for HF compatibility.

    This test verifies the fix for transformers v5+ where apply_chat_template
    requires dict messages with .get() access, not Message objects.
    """
    from oumi.core.processors.default_processor import DefaultProcessor

    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=False
    )

    # Verify processor is a DefaultProcessor
    assert isinstance(processor, DefaultProcessor)

    # Test the internal conversion method
    messages = [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Hi there!"),
    ]
    converted = processor._convert_messages_to_dicts(messages)

    # Verify conversion produces dict format with .get() access
    assert len(converted) == 2
    assert all(isinstance(m, dict) for m in converted)
    assert converted[0].get("role") == "user"
    assert converted[0].get("content") == "Hello"
    assert converted[1].get("role") == "assistant"
    assert converted[1].get("content") == "Hi there!"


def test_processor_apply_chat_template_with_message_objects():
    """Test processor.apply_chat_template works with Message objects.

    This verifies the v5 compatibility fix works end-to-end.
    """
    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=False
    )

    # This should work on both transformers v4 and v5
    messages = [
        Message(role=Role.USER, content="Test message"),
    ]
    prompt = processor.apply_chat_template(messages)
    assert isinstance(prompt, str)
    assert "Test message" in prompt


def test_processor_apply_chat_template_multimodal_text_content():
    """Test processor handles multimodal Message content items."""
    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=False
    )

    # Message with ContentItem list (text only - images handled separately)
    messages = [
        Message(
            role=Role.USER,
            content=[
                ContentItem(type=Type.TEXT, content="Describe the following:"),
            ],
        ),
    ]
    prompt = processor.apply_chat_template(messages)
    assert isinstance(prompt, str)
    assert "Describe the following:" in prompt


def test_processor_saves_chat_template(tmp_path):
    """Test that processor.save_config saves the chat template.

    This is critical for 3rd party libraries like vLLM to pick up
    the correct chat template when loading a trained model.
    """
    import json

    from oumi.core.processors.default_processor import DefaultProcessor

    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=False
    )

    # Verify processor has the expected chat template
    assert isinstance(processor, DefaultProcessor)
    assert processor.chat_template is not None
    assert len(processor.chat_template) > 0

    # Save the processor config
    processor.save_config(tmp_path)

    # Verify chat template was saved (either in tokenizer_config.json or chat_template.jinja)
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    chat_template_jinja_path = tmp_path / "chat_template.jinja"

    chat_template_saved = False
    if chat_template_jinja_path.exists():
        # Newer transformers versions save chat template to separate file
        with open(chat_template_jinja_path) as f:
            saved_template = f.read()
        assert len(saved_template) > 0
        chat_template_saved = True
    elif tokenizer_config_path.exists():
        # Older versions may save it in tokenizer_config.json
        with open(tokenizer_config_path) as f:
            config = json.load(f)
        if "chat_template" in config:
            assert len(config["chat_template"]) > 0
            chat_template_saved = True

    assert chat_template_saved, (
        "Chat template was not saved. This will cause issues with 3rd party "
        "inference libraries like vLLM that rely on the saved chat template."
    )


def test_processor_saved_chat_template_can_be_reloaded(tmp_path):
    """Test that the saved chat template can be reloaded and produces identical output.

    This verifies end-to-end that a model trained with oumi can be loaded
    by other libraries and produce the same prompts.
    """
    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=False
    )

    # Create a test message
    messages = [
        Message(role=Role.USER, content="Hello, how are you?"),
        Message(role=Role.ASSISTANT, content="I am doing well!"),
    ]

    # Get the prompt from the original processor
    original_prompt = processor.apply_chat_template(messages)

    # Save the processor config
    processor.save_config(tmp_path)

    # Reload the tokenizer from the saved config
    reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(str(tmp_path))

    # Verify the chat template was preserved
    assert reloaded_tokenizer.chat_template is not None
    assert reloaded_tokenizer.chat_template == processor.chat_template

    # Verify the reloaded tokenizer produces the same output
    # Convert messages to dict format (as vLLM would do)
    messages_as_dicts = [
        {"role": msg.role.value, "content": msg.content} for msg in messages
    ]
    reloaded_prompt = reloaded_tokenizer.apply_chat_template(
        messages_as_dicts, tokenize=False, add_generation_prompt=False
    )

    assert reloaded_prompt == original_prompt


def test_oumi_chat_template_handles_both_text_formats():
    """Test that oumi's chat templates handle both 'text' and 'content' keys.

    Oumi's custom chat templates are designed to work with both:
    - transformers v5 format: {"type": "text", "text": "..."}
    - oumi internal format: {"type": "text", "content": "..."}

    This is important for compatibility with different inference backends.
    """
    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)

    # Test with oumi's format (content key)
    oumi_format_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": "Hello world"},
            ],
        }
    ]

    # Test with transformers v5 format (text key)
    transformers_format_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello world"},
            ],
        }
    ]

    oumi_prompt = tokenizer.apply_chat_template(
        oumi_format_messages, tokenize=False, add_generation_prompt=False
    )
    transformers_prompt = tokenizer.apply_chat_template(
        transformers_format_messages, tokenize=False, add_generation_prompt=False
    )

    # Both formats should produce identical prompts
    assert oumi_prompt == transformers_prompt
    assert "Hello world" in oumi_prompt


def test_oumi_chat_template_handles_image_type_variations():
    """Test that oumi's chat templates handle different image type values.

    Oumi uses 'image_url' and 'image_path' types, while transformers v5 uses 'image'.
    Oumi's templates use item['type'].startswith('image') to handle all variants.
    """
    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer: BaseTokenizer = build_tokenizer(model_params)

    # Test with oumi's image_url type
    oumi_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "content": "http://example.com/img.jpg"},
                {"type": "text", "content": "What is this?"},
            ],
        }
    ]

    # Test with transformers v5 image type
    transformers_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "http://example.com/img.jpg"},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]

    oumi_prompt = tokenizer.apply_chat_template(
        oumi_image_messages, tokenize=False, add_generation_prompt=False
    )
    transformers_prompt = tokenizer.apply_chat_template(
        transformers_image_messages, tokenize=False, add_generation_prompt=False
    )

    # Both should include the image placeholder and text
    assert "<image>" in oumi_prompt
    assert "What is this?" in oumi_prompt
    assert "<image>" in transformers_prompt
    assert "What is this?" in transformers_prompt
