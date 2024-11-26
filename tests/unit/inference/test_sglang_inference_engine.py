import functools
from unittest.mock import patch

import PIL.Image
import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference.sglang_inference_engine import SGLangInferenceEngine
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)


def create_test_remote_params():
    return RemoteParams(api_key="test_api_key", api_url="<placeholder>")


def create_test_vision_language_engine() -> SGLangInferenceEngine:
    return SGLangInferenceEngine(
        model_params=ModelParams(
            model_name="llava-hf/llava-1.5-7b-hf",
            torch_dtype_str="bfloat16",
            model_max_length=1024,
            chat_template="llama3-instruct",
            trust_remote_code=True,
        ),
        remote_params=create_test_remote_params(),
    )


def create_test_text_only_engine() -> SGLangInferenceEngine:
    return SGLangInferenceEngine(
        model_params=ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="bfloat16",
            model_max_length=1024,
            chat_template="llama3-instruct",
            trust_remote_code=True,
            tokenizer_pad_token="<|endoftext|>",
        ),
        remote_params=create_test_remote_params(),
    )


@functools.cache
def _generate_all_engines() -> list[SGLangInferenceEngine]:
    return [create_test_vision_language_engine(), create_test_text_only_engine()]


@pytest.mark.parametrize(
    "engine",
    _generate_all_engines(),
)
def test_convert_conversation_to_api_input(engine: SGLangInferenceEngine):
    is_vision_language: bool = "llava" in engine._model.lower()

    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    conversation = Conversation(
        messages=(
            [Message(role=Role.SYSTEM, content="System message")]
            + (
                [Message(role=Role.USER, binary=png_bytes, type=Type.IMAGE_BINARY)]
                if is_vision_language
                else []
            )
            + [
                Message(role=Role.USER, content="User message"),
                Message(role=Role.ASSISTANT, content="Assistant message"),
            ]
        ),
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    generation_params = GenerationParams(
        max_new_tokens=100,
        temperature=0.2,
        top_p=0.8,
        min_p=0.1,
        frequency_penalty=0.3,
        presence_penalty=0.4,
        stop_strings=["stop it"],
        stop_token_ids=[32000],
    )

    result = engine._convert_conversation_to_api_input(conversation, generation_params)

    expected_prompt = (
        "\n\n".join(
            [
                engine._tokenizer.bos_token
                + "<|start_header_id|>system<|end_header_id|>",
                "System message<|eot_id|><|start_header_id|>user<|end_header_id|>",
            ]
            + (
                ["<|image|><|eot_id|><|start_header_id|>user<|end_header_id|>"]
                if is_vision_language
                else []
            )
            + [
                "User message<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                (
                    "Assistant message<|eot_id|><|start_header_id|>assistant"
                    "<|end_header_id|>"
                ),
            ]
        )
        + "\n\n"
    )

    assert "text" in result, result
    assert result["text"] == expected_prompt, result
    if is_vision_language:
        assert "image_data" in result, result
        assert result["image_data"].startswith("data:image/png;base64,"), result
    else:
        assert "image_data" not in result, result
    assert "sampling_params" in result, result
    assert result["sampling_params"]["max_new_tokens"] == 100, result
    assert result["sampling_params"]["temperature"] == 0.2, result
    assert result["sampling_params"]["top_p"] == 0.8, result
    assert result["sampling_params"]["min_p"] == 0.1, result
    assert result["sampling_params"]["frequency_penalty"] == 0.3, result
    assert result["sampling_params"]["presence_penalty"] == 0.4, result
    assert result["sampling_params"]["stop"] == ["stop it"], result
    assert result["sampling_params"]["stop_token_ids"] == [32000], result


@pytest.mark.parametrize(
    "engine",
    _generate_all_engines(),
)
def test_convert_api_output_to_conversation(engine):
    original_conversation = Conversation(
        messages=[
            Message(content="User message", role=Role.USER),
        ],
        metadata={"key": "value"},
        conversation_id="test_id",
    )
    api_response = {"text": "Assistant response"}

    result = engine._convert_api_output_to_conversation(
        api_response, original_conversation
    )

    assert len(result.messages) == 2
    assert result.messages[0].content == "User message"
    assert result.messages[1].content == "Assistant response"
    assert result.messages[1].role == Role.ASSISTANT
    assert result.messages[1].type == Type.TEXT
    assert result.metadata == {"key": "value"}
    assert result.conversation_id == "test_id"


@pytest.mark.parametrize(
    "engine",
    _generate_all_engines(),
)
def test_get_request_headers(engine):
    remote_params = RemoteParams(api_key="test_api_key", api_url="<placeholder>")

    with patch.object(
        SGLangInferenceEngine,
        "_get_api_key",
        return_value="test_api_key",
    ):
        result = engine._get_request_headers(remote_params)

    assert result["Content-Type"] == "application/json"


@pytest.mark.parametrize(
    "engine",
    _generate_all_engines(),
)
def test_get_supported_params(engine):
    assert engine.get_supported_params() == set(
        {
            "frequency_penalty",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
        }
    )
