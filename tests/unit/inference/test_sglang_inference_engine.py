import functools
import itertools
import json
from typing import Optional
from unittest.mock import patch

import PIL.Image
import pydantic
import pytest

from oumi.core.configs import (
    GenerationParams,
    GuidedDecodingParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.inference.sglang_inference_engine import SGLangInferenceEngine
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)
from oumi.utils.logging import logger


class SamplePydanticType(pydantic.BaseModel):
    name: str
    score: float


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
    "engine,guided_decoding,num_images",
    list(
        itertools.product(
            _generate_all_engines(),
            [
                None,
                GuidedDecodingParams(choice=["apple", "pear"]),
                GuidedDecodingParams(json={"enum": ["apple", "pear"]}),
                GuidedDecodingParams(json=SamplePydanticType(name="hey", score=0.7)),
                GuidedDecodingParams(json=SamplePydanticType),
                GuidedDecodingParams(regex="(apple|pear)"),
            ],
            [None],  # num_images
        )
    )
    + [
        (create_test_vision_language_engine(), None, 2),
    ],
)
def test_convert_conversation_to_api_input(
    engine: SGLangInferenceEngine,
    guided_decoding: Optional[GuidedDecodingParams],
    num_images: Optional[int],
):
    is_vision_language: bool = "llava" in engine._model_params.model_name.lower()
    num_images = num_images or (1 if is_vision_language else 0)

    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    conversation = Conversation(
        messages=(
            [Message(role=Role.SYSTEM, content="System message")]
            + (
                [
                    Message(
                        role=Role.USER,
                        content=(
                            [ContentItem(binary=png_bytes, type=Type.IMAGE_BINARY)]
                            * num_images
                            + [ContentItem(type=Type.TEXT, content="User message")]
                        ),
                    )
                ]
                if is_vision_language
                else [Message(role=Role.USER, content="User message")]
            )
            + [
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
        guided_decoding=guided_decoding,
    )

    if num_images > 1 and is_vision_language and not engine._supports_multiple_images:
        with pytest.raises(ValueError, match="A conversation contains too many images"):
            engine._convert_conversation_to_api_input(
                conversation, generation_params, ModelParams()
            )

        return

    result = engine._convert_conversation_to_api_input(
        conversation, generation_params, ModelParams()
    )

    assert isinstance(engine._tokenizer.bos_token, str), (
        "bos_token: {engine._tokenizer.bos_token}"
    )
    expected_prompt = (
        "\n\n".join(
            [
                (
                    engine._tokenizer.bos_token
                    + "<|start_header_id|>system<|end_header_id|>"
                ),
                "System message<|eot_id|><|start_header_id|>user<|end_header_id|>",
            ]
            + [
                (
                    ("<|image|>" if is_vision_language else "")
                    + "User message<|eot_id|>"
                    + "<|start_header_id|>assistant<|end_header_id|>"
                ),
                (
                    "Assistant message<|eot_id|><|start_header_id|>assistant"
                    "<|end_header_id|>"
                ),
            ]
        )
        + "\n\n"
    )

    assert "text" in result, result
    logger.info(f"result['text']:\n{result['text']}\n\n")
    logger.info(f"expected_prompt:\n{expected_prompt}\n\n")
    assert result["text"] == expected_prompt, result
    if is_vision_language:
        assert num_images >= 1
        assert "image_data" in result, result
        if num_images > 1:
            assert isinstance(result["image_data"], list)
            assert len(result["image_data"]) == num_images
            for idx in range(num_images):
                assert result["image_data"][idx].startswith("data:image/png;base64,"), (
                    result
                )

        else:
            assert isinstance(result["image_data"], str)
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

    expect_valid_regex = False
    expect_valid_json_schema = False
    if guided_decoding is not None:
        if guided_decoding.regex is not None:
            assert "regex" in result["sampling_params"]
            assert result["sampling_params"]["regex"] == guided_decoding.regex
            expect_valid_regex = True
        elif guided_decoding.json is not None:
            assert "json_schema" in result["sampling_params"]
            if isinstance(guided_decoding.json, pydantic.BaseModel) or (
                isinstance(guided_decoding.json, type)
                and issubclass(guided_decoding.json, pydantic.BaseModel)
            ):
                assert (
                    json.loads(result["sampling_params"]["json_schema"])
                    == guided_decoding.json.model_json_schema()
                )

            else:
                assert (
                    json.loads(result["sampling_params"]["json_schema"])
                    == guided_decoding.json
                )
            expect_valid_json_schema = True
        elif guided_decoding.choice is not None:
            assert "json_schema" in result["sampling_params"]
            assert json.loads(result["sampling_params"]["json_schema"]) == {
                "enum": list(guided_decoding.choice)
            }
            expect_valid_json_schema = True

    if not expect_valid_regex:
        assert "regex" in result["sampling_params"]
        assert result["sampling_params"]["regex"] is None
    if not expect_valid_json_schema:
        assert "json_schema" in result["sampling_params"]
        assert result["sampling_params"]["json_schema"] is None


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
            "guided_decoding",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
        }
    )
