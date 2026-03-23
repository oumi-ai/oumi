import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import jsonlines
import PIL.Image
import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type
from oumi.inference import VLLMInferenceEngine
from oumi.utils.conversation_utils import base64encode_content_item_image_bytes
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)
from oumi.utils.packaging import is_vllm_v0_12_or_later

try:
    vllm_import_failed = False
    from vllm.lora.request import LoRARequest  # type: ignore
    from vllm.outputs import (  # pyright: ignore[reportMissingImports]
        CompletionOutput,
        RequestOutput,
    )

    def _create_vllm_output(responses: list[str], output_id: str) -> RequestOutput:
        outputs = []
        for ind, response in enumerate(responses):
            outputs.append(
                CompletionOutput(
                    text=response,
                    index=ind,
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=None,
                )
            )
        return RequestOutput(
            request_id=output_id,
            outputs=outputs,
            prompt=None,
            prompt_token_ids=[],
            prompt_logprobs=None,
            finished=True,
        )
except ModuleNotFoundError:
    vllm_import_failed = True


@pytest.fixture
def mock_sampling_params():
    with patch("oumi.inference.vllm_inference_engine.SamplingParams") as mock:
        yield mock


#
# Fixtures
#
@pytest.fixture
def mock_vllm():
    with patch("oumi.inference.vllm_inference_engine.vllm") as mvllm:
        yield mvllm


@pytest.fixture
def mock_lora_request():
    with patch("oumi.inference.vllm_inference_engine.LoRARequest") as mlo:
        yield mlo


def _get_default_model_params(use_lora: bool = False) -> ModelParams:
    return ModelParams(
        model_name="MlpEncoder",
        adapter_model="/path/to/adapter" if use_lora else None,
        trust_remote_code=True,
        tokenizer_pad_token="<pad>",
        tokenizer_name="openai-community/gpt2",
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(generation=GenerationParams(max_new_tokens=5))


def _setup_input_conversations(filepath: str, conversations: list[Conversation]):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).touch()
    with jsonlines.open(filepath, mode="w") as writer:
        for conversation in conversations:
            json_obj = conversation.to_dict()
            writer.write(json_obj)
    # Add some empty lines into the file
    with open(filepath, "a") as f:
        f.write("\n\n\n")


def _create_test_pil_image() -> PIL.Image.Image:
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    return pil_image


def _create_test_png_image_bytes() -> bytes:
    return create_png_bytes_from_image(_create_test_pil_image())


def _create_test_png_image_base64_str() -> str:
    return base64encode_content_item_image_bytes(
        ContentItem(binary=_create_test_png_image_bytes(), type=Type.IMAGE_BINARY),
        add_mime_prefix=True,
    )


#
# Tests
#
@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [
        _create_vllm_output(["The first time I saw"], "123")
    ]

    engine = VLLMInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="You're a good assistant!",
                role=Role.SYSTEM,
            ),
            Message(
                content="Hi there",
                role=Role.USER,
            ),
            Message(
                content="Hello again!",
                role=Role.USER,
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )
    expected_result = [
        Conversation(
            messages=[
                *conversation.messages,
                Message(
                    content="The first time I saw",
                    role=Role.ASSISTANT,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
    ]
    result = engine.infer([conversation], _get_default_inference_config())
    assert expected_result == result
    mock_vllm_instance.chat.assert_called_once()
    assert isinstance(mock_vllm_instance.chat.call_args_list[0][0][0], list)
    assert mock_vllm_instance.chat.call_args_list[0][0][0] == [
        [
            {
                "content": "You're a good assistant!",
                "role": "system",
            },
            {
                "content": [
                    {
                        "text": "Hi there",
                        "type": "text",
                    },
                    {
                        "text": "Hello again!",
                        "type": "text",
                    },
                ],
                "role": "user",
            },
        ]
    ]


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_multimodal(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [
        _create_vllm_output(["The first time I saw"], "123")
    ]

    engine = VLLMInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(
                        type=Type.IMAGE_BINARY, binary=_create_test_png_image_bytes()
                    ),
                    ContentItem(type=Type.TEXT, content="Describe this image!"),
                ],
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )
    expected_result = [
        Conversation(
            messages=[
                *conversation.messages,
                Message(
                    content="The first time I saw",
                    role=Role.ASSISTANT,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
    ]
    result = engine.infer([conversation], _get_default_inference_config())
    assert expected_result == result
    mock_vllm_instance.chat.assert_called_once()
    assert isinstance(mock_vllm_instance.chat.call_args_list[0][0][0], list)
    assert mock_vllm_instance.chat.call_args_list[0][0][0] == [
        [
            {
                "content": [
                    {
                        "image_url": {"url": _create_test_png_image_base64_str()},
                        "type": "image_url",
                    },
                    {
                        "text": "Describe this image!",
                        "type": "text",
                    },
                ],
                "role": "user",
            }
        ]
    ]


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_lora(mock_vllm, mock_lora_request):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [
        _create_vllm_output(["The first time I saw"], "123")
    ]

    lora_request = LoRARequest(
        lora_name="oumi_lora_adapter",
        lora_int_id=1,
        lora_path="/path/to/adapter",
    )
    mock_lora_request.return_value = lora_request

    with patch("oumi.inference.vllm_inference_engine.get_lora_rank", return_value=32):
        engine = VLLMInferenceEngine(_get_default_model_params(use_lora=True))
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
            Message(
                content="Hello again!",
                role=Role.USER,
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )
    expected_result = [
        Conversation(
            messages=[
                *conversation.messages,
                Message(
                    content="The first time I saw",
                    role=Role.ASSISTANT,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
    ]
    result = engine.infer([conversation], _get_default_inference_config())
    assert expected_result == result

    mock_lora_request.assert_called_once_with(
        lora_name="oumi_lora_adapter",
        lora_int_id=1,
        lora_path="/path/to/adapter",
    )
    mock_vllm_instance.chat.assert_called_once_with(
        ANY,
        sampling_params=ANY,
        lora_request=lora_request,
        use_tqdm=False,
        chat_template=None,
        chat_template_content_format="auto",
        chat_template_kwargs=None,
    )


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_empty(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(_get_default_model_params())
    result = engine.infer([], _get_default_inference_config())
    assert [] == result
    mock_vllm_instance.chat.assert_not_called()


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_to_file(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.side_effect = [
        [
            _create_vllm_output(["The first time I saw"], "123"),
            _create_vllm_output(["The U.S."], "123"),
        ]
    ]
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = VLLMInferenceEngine(_get_default_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)
        result = engine.infer(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_from_file(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [
        _create_vllm_output(["The first time I saw"], "123")
    ]
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = VLLMInferenceEngine(_get_default_model_params())
        conversation = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation])
        expected_result = [
            Conversation(
                messages=[
                    *conversation.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            )
        ]
        inference_config = _get_default_inference_config()
        inference_config.input_path = str(input_path)
        infer_result = engine.infer(
            inference_config=inference_config,
        )
        assert expected_result == infer_result


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_from_file_empty(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = VLLMInferenceEngine(_get_default_model_params())
        inference_config = _get_default_inference_config()
        inference_config.input_path = str(input_path)
        result = engine.infer(inference_config=inference_config)
        assert [] == result
        mock_vllm_instance.chat.assert_not_called()


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_from_file_to_file(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.side_effect = [
        [
            _create_vllm_output(["The first time I saw"], "123"),
            _create_vllm_output(["The U.S."], "123"),
        ]
    ]
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = VLLMInferenceEngine(_get_default_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1, conversation_2])
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)
        result = engine.infer(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        # Ensure the final output is in order.
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_guided_decoding_json(
    mock_vllm, single_turn_conversation, mock_sampling_params
):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    }
    config = InferenceConfig(
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            tokenizer_pad_token="<eos>",
        ),
        generation=GenerationParams(guided_decoding=GuidedDecodingParams(json=schema)),
    )

    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(config.model)
    engine._llm.chat = MagicMock()
    engine._llm.chat.return_value = [
        _create_vllm_output(["The first time I saw"], "123")
    ]
    result = engine._infer([single_turn_conversation], config)

    # Verify SamplingParams was called with guided_decoding/structured_outputs
    assert result is not None
    mock_sampling_params.assert_called_once()
    call_kwargs = mock_sampling_params.call_args[1]
    guided_key = "structured_outputs" if is_vllm_v0_12_or_later() else "guided_decoding"
    assert guided_key in call_kwargs
    assert call_kwargs[guided_key].json == schema


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_guided_decoding_regex(mock_vllm, mock_sampling_params):
    pattern = r"\d{3}-\d{2}-\d{4}"

    config = InferenceConfig(
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            tokenizer_pad_token="<eos>",
        ),
        generation=GenerationParams(
            guided_decoding=GuidedDecodingParams(regex=pattern)
        ),
    )

    conversation = Conversation(
        messages=[Message(content="Is this a SSN?", role=Role.USER)]
    )

    # Mock the VLLM response
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(config.model)

    engine._llm = MagicMock()
    engine._llm.chat.return_value = [MagicMock(outputs=[MagicMock(text="123-45-6789")])]

    result = engine._infer([conversation], config)

    # Verify SamplingParams was called with guided_decoding/structured_outputs
    assert result is not None
    mock_sampling_params.assert_called_once()
    call_kwargs = mock_sampling_params.call_args[1]
    guided_key = "structured_outputs" if is_vllm_v0_12_or_later() else "guided_decoding"
    assert guided_key in call_kwargs
    assert call_kwargs[guided_key].regex == pattern


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_guided_decoding_choice(mock_vllm, mock_sampling_params):
    choices = ["option1", "option2"]
    config = InferenceConfig(
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="openai-community/gpt2",
            tokenizer_pad_token="<eos>",
        ),
        generation=GenerationParams(
            guided_decoding=GuidedDecodingParams(choice=choices)
        ),
    )

    conversation = Conversation(
        messages=[Message(content="What is your favorite color?", role=Role.USER)]
    )

    # Mock the VLLM response
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(config.model)

    engine._llm = MagicMock()
    engine._llm.chat.return_value = [MagicMock(outputs=[MagicMock(text="option1")])]

    result = engine._infer([conversation], config)

    # Verify SamplingParams was called with guided_decoding/structured_outputs
    assert result is not None
    mock_sampling_params.assert_called_once()
    call_kwargs = mock_sampling_params.call_args[1]
    guided_key = "structured_outputs" if is_vllm_v0_12_or_later() else "guided_decoding"
    assert guided_key in call_kwargs
    assert call_kwargs[guided_key].choice == choices


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_chat_template_kwargs_enable_thinking_false(mock_vllm):
    convo = Conversation(messages=[Message(content="hi", role=Role.USER)])
    model_params = _get_default_model_params()
    model_params.chat_template_kwargs = {"enable_thinking": False}

    engine = VLLMInferenceEngine(model_params)
    engine._llm = MagicMock()
    engine._llm.chat.return_value = [MagicMock(outputs=[MagicMock(text="response")])]

    inference_config = _get_default_inference_config()
    inference_config.model = model_params

    engine._infer([convo], inference_config=inference_config)

    # Assert the parameter was passed
    call_kwargs = engine._llm.chat.call_args.kwargs
    assert "chat_template_kwargs" in call_kwargs
    assert call_kwargs["chat_template_kwargs"].get("enable_thinking") is False


# =============================================================================
# Tests for vLLM message format compatibility with oumi chat templates
# =============================================================================


def test_vllm_message_format_uses_text_key_for_text_content():
    """Verify that vLLM message conversion uses 'text' key, not 'content'.

    vLLM expects OpenAI API format: {"type": "text", "text": "..."}
    This test ensures oumi's conversion produces this format.
    """
    from oumi.utils.conversation_utils import convert_message_content_item_to_json_dict

    text_item = ContentItem(type=Type.TEXT, content="Hello world")
    result = convert_message_content_item_to_json_dict(text_item)

    # Must use 'text' key for vLLM/OpenAI compatibility
    assert result["type"] == "text"
    assert "text" in result
    assert result["text"] == "Hello world"
    # Should NOT have 'content' key for text items
    assert "content" not in result


def test_vllm_message_format_uses_image_url_structure():
    """Verify that vLLM message conversion uses OpenAI image_url structure.

    vLLM expects: {"type": "image_url", "image_url": {"url": "..."}}
    This nested structure is different from transformers v5 format.
    """
    from oumi.utils.conversation_utils import convert_message_content_item_to_json_dict

    image_item = ContentItem(type=Type.IMAGE_URL, content="http://example.com/img.jpg")
    result = convert_message_content_item_to_json_dict(image_item)

    # Must use OpenAI's nested image_url structure
    assert result["type"] == "image_url"
    assert "image_url" in result
    assert isinstance(result["image_url"], dict)
    assert result["image_url"]["url"] == "http://example.com/img.jpg"


def test_vllm_multimodal_message_conversion_end_to_end():
    """Test complete multimodal message conversion for vLLM.

    This verifies that a multimodal conversation is correctly converted
    to the format vLLM expects for its chat API.
    """
    from oumi.utils.conversation_utils import create_list_of_message_json_dicts

    messages = [
        Message(
            role=Role.USER,
            content=[
                ContentItem(type=Type.IMAGE_URL, content="http://example.com/img.jpg"),
                ContentItem(type=Type.TEXT, content="What is in this image?"),
            ],
        ),
    ]

    result = create_list_of_message_json_dicts(
        messages, group_adjacent_same_role_turns=True
    )

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Image item should use OpenAI format
    image_item = result[0]["content"][0]
    assert image_item["type"] == "image_url"
    assert "image_url" in image_item
    assert image_item["image_url"]["url"] == "http://example.com/img.jpg"

    # Text item should use 'text' key
    text_item = result[0]["content"][1]
    assert text_item["type"] == "text"
    assert text_item["text"] == "What is in this image?"


def test_vllm_format_compatible_with_oumi_chat_template():
    """Test that vLLM's message format works with oumi's custom chat templates.

    This is a critical compatibility test: when vLLM loads a model trained
    with oumi and applies the saved chat template to vLLM-formatted messages,
    it should produce correct prompts.

    vLLM uses: {"type": "text", "text": "..."} and
               {"type": "image_url", "image_url": {"url": "..."}}

    Oumi's templates handle these via:
    - Text: (item['text'] if 'text' in item else item['content'])
    - Images: item['type'].startswith('image')
    """
    from oumi.builders import build_tokenizer
    from oumi.core.configs import ModelParams

    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer = build_tokenizer(model_params)

    # This is the format vLLM would pass to the chat template
    vllm_format_messages = [
        {
            "role": "user",
            "content": [
                # vLLM uses image_url with nested structure
                {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                # vLLM uses 'text' key
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ]

    # Apply the oumi chat template (which would be saved with the model)
    prompt = tokenizer.apply_chat_template(
        vllm_format_messages, tokenize=False, add_generation_prompt=True
    )

    # The template should correctly:
    # 1. Insert image placeholder for image_url type (startswith('image'))
    # 2. Extract text from 'text' key
    assert "<image>" in prompt
    assert "What do you see?" in prompt
    assert "USER:" in prompt
    assert "ASSISTANT:" in prompt


def test_vllm_text_only_message_format():
    """Test that simple text messages work with vLLM format.

    For text-only messages, vLLM may pass content as a string directly,
    which oumi's templates should handle.
    """
    from oumi.builders import build_tokenizer
    from oumi.core.configs import ModelParams

    model_params = ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf", chat_template="llava"
    )
    tokenizer = build_tokenizer(model_params)

    # Simple text message (content as string, not list)
    simple_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well!"},
    ]

    prompt = tokenizer.apply_chat_template(
        simple_messages, tokenize=False, add_generation_prompt=False
    )

    assert "Hello, how are you?" in prompt
    assert "I am doing well!" in prompt
    assert "USER:" in prompt
    assert "ASSISTANT:" in prompt
