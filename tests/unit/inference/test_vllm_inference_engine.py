import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Final
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


#
# Tests for vLLM passthrough kwargs
#
@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_language_model_only_passed_to_vllm(mock_vllm):
    """Verify language_model_only from model_kwargs is passed to vllm.LLM."""
    mock_vllm.LLM.return_value = Mock()

    params = _get_default_model_params()
    params.model_kwargs = {"language_model_only": True}
    VLLMInferenceEngine(params)

    call_kwargs = mock_vllm.LLM.call_args[1]
    assert call_kwargs.get("language_model_only") is True


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_hf_config_path_passed_to_vllm(mock_vllm):
    """Verify hf_config_path from model_kwargs is passed to vllm.LLM."""
    mock_vllm.LLM.return_value = Mock()

    params = _get_default_model_params()
    params.model_kwargs = {
        "language_model_only": True,
        "hf_config_path": "Qwen/Qwen3.5-0.8B",
    }
    VLLMInferenceEngine(params)

    call_kwargs = mock_vllm.LLM.call_args[1]
    assert call_kwargs.get("hf_config_path") == "Qwen/Qwen3.5-0.8B"
    assert call_kwargs.get("language_model_only") is True


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_no_passthrough_kwargs_by_default(mock_vllm):
    """Verify passthrough kwargs are not set when not in model_kwargs."""
    mock_vllm.LLM.return_value = Mock()

    VLLMInferenceEngine(_get_default_model_params())

    call_kwargs = mock_vllm.LLM.call_args[1]
    assert "language_model_only" not in call_kwargs
    assert "hf_config_path" not in call_kwargs


#
# Tool-calling tests
#
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_CALENDAR_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
}


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_forwards_tools_kwarg(mock_vllm):
    """Conversation.tools is forwarded to LLM.chat(tools=...)."""
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [_create_vllm_output(["ok"], "1")]

    engine = VLLMInferenceEngine(_get_default_model_params())
    conv = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[Message(role=Role.USER, content="weather in Tokyo?")],
        conversation_id="1",
    )
    engine.infer([conv], _get_default_inference_config())

    mock_vllm_instance.chat.assert_called_once()
    call_kwargs = mock_vllm_instance.chat.call_args.kwargs
    assert call_kwargs["tools"] == [_WEATHER_TOOL]


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_fast_path_when_no_tools(mock_vllm):
    """When no conversation has tools, chat() is called once without `tools`."""
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [
        _create_vllm_output(["a"], "1"),
        _create_vllm_output(["b"], "2"),
    ]

    engine = VLLMInferenceEngine(_get_default_model_params())
    convs = [
        Conversation(
            messages=[Message(role=Role.USER, content="hi")], conversation_id="1"
        ),
        Conversation(
            messages=[Message(role=Role.USER, content="bye")], conversation_id="2"
        ),
    ]
    engine.infer(convs, _get_default_inference_config())

    assert mock_vllm_instance.chat.call_count == 1
    assert "tools" not in mock_vllm_instance.chat.call_args.kwargs


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_groups_conversations_by_tools(mock_vllm):
    """Heterogeneous tool sets dispatch as separate chat() calls in input order."""
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance

    # Two groups: T1 used by indices [0, 2], T2 used by [1]. Each group's
    # response is tagged so the stitched result can be verified to preserve
    # original (A, B, C) input order.
    def chat_side_effect(messages, **kwargs):
        tools = kwargs.get("tools")
        prefix = "weather" if tools == [_WEATHER_TOOL] else "calendar"
        return [_create_vllm_output([f"{prefix}-out"], f"{prefix}-id")] * len(messages)

    mock_vllm_instance.chat.side_effect = chat_side_effect

    engine = VLLMInferenceEngine(_get_default_model_params())
    convs = [
        Conversation(
            tools=[_WEATHER_TOOL],
            messages=[Message(role=Role.USER, content="A")],
            conversation_id="A",
        ),
        Conversation(
            tools=[_CALENDAR_TOOL],
            messages=[Message(role=Role.USER, content="B")],
            conversation_id="B",
        ),
        Conversation(
            tools=[_WEATHER_TOOL],
            messages=[Message(role=Role.USER, content="C")],
            conversation_id="C",
        ),
    ]
    results = engine.infer(convs, _get_default_inference_config())

    assert mock_vllm_instance.chat.call_count == 2
    tools_per_call = [
        call.kwargs.get("tools") for call in mock_vllm_instance.chat.call_args_list
    ]
    assert [_WEATHER_TOOL] in tools_per_call
    assert [_CALENDAR_TOOL] in tools_per_call

    # Original input order is preserved on the way out.
    assert [r.conversation_id for r in results] == ["A", "B", "C"]
    # And each conversation got the response for its own tool group.
    assert results[0].messages[-1].content == "weather-out"
    assert results[1].messages[-1].content == "calendar-out"
    assert results[2].messages[-1].content == "weather-out"


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_input_preserves_tool_calls_and_tool_call_id(mock_vllm):
    """Multi-turn tool-use messages round-trip into the chat() input."""
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [_create_vllm_output(["ok"], "1")]

    engine = VLLMInferenceEngine(_get_default_model_params())
    tool_call = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
    }
    conv = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(role=Role.USER, content="What's the weather in Tokyo?"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call]),
            Message(role=Role.TOOL, content="22C, sunny", tool_call_id="call_abc"),
        ],
        conversation_id="1",
    )
    engine.infer([conv], _get_default_inference_config())

    sent = mock_vllm_instance.chat.call_args.args[0][0]
    # User message unchanged.
    assert sent[0]["role"] == "user"
    # Assistant tool-call message: content=None, tool_calls forwarded.
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] is None
    assert sent[1]["tool_calls"] == [tool_call]
    # Tool response: content + tool_call_id forwarded.
    assert sent[2]["role"] == "tool"
    assert sent[2]["content"] == "22C, sunny"
    assert sent[2]["tool_call_id"] == "call_abc"


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_tool_call_parser_populates_message_tool_calls(mock_vllm):
    """When tool_call_parser is set, parsed tool calls land on Message.tool_calls."""
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [
        _create_vllm_output(["<tool_call>...</tool_call>"], "1")
    ]

    # Stub tool call object exposing `.model_dump()`.
    fake_call = Mock()
    fake_call.model_dump.return_value = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
    }
    fake_extracted = SimpleNamespace(
        tools_called=True,
        tool_calls=[fake_call],
        content=None,
    )
    parser_instance = Mock()
    parser_instance.extract_tool_calls.return_value = fake_extracted
    parser_cls = Mock(return_value=parser_instance)
    manager = Mock()
    manager.get_tool_parser.return_value = parser_cls

    with (
        patch("oumi.inference.vllm_inference_engine.ToolParserManager", manager),
        patch(
            "oumi.inference.vllm_inference_engine._VLLM_TOOL_PARSERS_AVAILABLE", True
        ),
    ):
        engine = VLLMInferenceEngine(
            _get_default_model_params(), tool_call_parser="hermes"
        )

    conv = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[Message(role=Role.USER, content="weather?")],
        conversation_id="1",
    )
    results = engine.infer([conv], _get_default_inference_config())

    assistant = results[0].messages[-1]
    assert assistant.tool_calls == [fake_call.model_dump.return_value]
    assert assistant.content is None
    assert results[0].metadata["finish_reason"] == "tool_calls"


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_tool_call_parser_error_falls_back_to_text(mock_vllm):
    """A parser exception leaves Message.tool_calls=None and keeps raw text."""
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.return_value = [_create_vllm_output(["raw"], "1")]

    parser_instance = Mock()
    parser_instance.extract_tool_calls.side_effect = RuntimeError("boom")
    parser_cls = Mock(return_value=parser_instance)
    manager = Mock()
    manager.get_tool_parser.return_value = parser_cls

    with (
        patch("oumi.inference.vllm_inference_engine.ToolParserManager", manager),
        patch(
            "oumi.inference.vllm_inference_engine._VLLM_TOOL_PARSERS_AVAILABLE", True
        ),
    ):
        engine = VLLMInferenceEngine(
            _get_default_model_params(), tool_call_parser="hermes"
        )

    conv = Conversation(
        messages=[Message(role=Role.USER, content="hi")], conversation_id="1"
    )
    results = engine.infer([conv], _get_default_inference_config())

    assistant = results[0].messages[-1]
    assert assistant.tool_calls is None
    assert assistant.content == "raw"
    # Falls back to the raw vLLM finish reason (not overridden to "tool_calls").
    assert results[0].metadata.get("finish_reason") != "tool_calls"


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_tool_call_parser_unknown_name_raises(mock_vllm):
    """An unknown parser name raises a clear ValueError."""
    mock_vllm.LLM.return_value = Mock()

    manager = Mock()
    manager.get_tool_parser.side_effect = KeyError("not-a-parser")

    with (
        patch("oumi.inference.vllm_inference_engine.ToolParserManager", manager),
        patch(
            "oumi.inference.vllm_inference_engine._VLLM_TOOL_PARSERS_AVAILABLE", True
        ),
    ):
        with pytest.raises(ValueError, match="Unknown vLLM tool_call_parser"):
            VLLMInferenceEngine(
                _get_default_model_params(), tool_call_parser="not-a-parser"
            )


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_tool_call_parser_from_model_params(mock_vllm):
    """`model_params.tool_call_parser` is used when no kwarg is supplied."""
    mock_vllm.LLM.return_value = Mock()

    manager = Mock()
    parser_cls = Mock(return_value=Mock())
    manager.get_tool_parser.return_value = parser_cls

    params = _get_default_model_params()
    params.tool_call_parser = "hermes"

    with (
        patch("oumi.inference.vllm_inference_engine.ToolParserManager", manager),
        patch(
            "oumi.inference.vllm_inference_engine._VLLM_TOOL_PARSERS_AVAILABLE", True
        ),
    ):
        VLLMInferenceEngine(params)

    manager.get_tool_parser.assert_called_once_with("hermes")
