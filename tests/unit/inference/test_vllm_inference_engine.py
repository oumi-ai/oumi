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
        tokenizer_name="gpt2",
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
    result = engine.infer_online([conversation], _get_default_inference_config())
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
    result = engine.infer_online([conversation], _get_default_inference_config())
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
    result = engine.infer_online([conversation], _get_default_inference_config())
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
    )


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_empty(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(_get_default_model_params())
    result = engine.infer_online([], _get_default_inference_config())
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
        result = engine.infer_online(
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
        result = engine.infer_from_file(str(input_path), inference_config)
        assert expected_result == result
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
        result = engine.infer_from_file(str(input_path), inference_config)
        assert [] == result
        inference_config.input_path = str(input_path)
        infer_result = engine.infer(
            inference_config=inference_config,
        )
        assert [] == infer_result
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
        result = engine.infer_online(
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
            model_name="MlpEncoder", tokenizer_name="gpt2", tokenizer_pad_token="<eos>"
        ),
        generation=GenerationParams(guided_decoding=GuidedDecodingParams(json=schema)),
    )

    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(config.model)

    engine._llm.chat.return_value = [
        _create_vllm_output(["The first time I saw"], "123")
    ]
    result = engine._infer([single_turn_conversation], config)

    # Verify SamplingParams was called with guided_decoding
    assert result is not None
    mock_sampling_params.assert_called_once()
    call_kwargs = mock_sampling_params.call_args[1]
    assert "guided_decoding" in call_kwargs
    assert call_kwargs["guided_decoding"].json == schema


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_guided_decoding_regex(mock_vllm, mock_sampling_params):
    pattern = r"\d{3}-\d{2}-\d{4}"

    config = InferenceConfig(
        model=ModelParams(
            model_name="MlpEncoder", tokenizer_name="gpt2", tokenizer_pad_token="<eos>"
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

    # Verify SamplingParams was called with guided_decoding
    assert result is not None
    mock_sampling_params.assert_called_once()
    call_kwargs = mock_sampling_params.call_args[1]
    assert "guided_decoding" in call_kwargs
    assert call_kwargs["guided_decoding"].regex == pattern


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_guided_decoding_choice(mock_vllm, mock_sampling_params):
    choices = ["option1", "option2"]
    config = InferenceConfig(
        model=ModelParams(
            model_name="MlpEncoder", tokenizer_name="gpt2", tokenizer_pad_token="<eos>"
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

    # Verify SamplingParams was called with guided_decoding
    assert result is not None
    mock_sampling_params.assert_called_once()
    call_kwargs = mock_sampling_params.call_args[1]
    assert "guided_decoding" in call_kwargs
    assert call_kwargs["guided_decoding"].choice == choices
