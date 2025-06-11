import itertools
import tempfile
from pathlib import Path

import jsonlines
import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.inference import NativeTextInferenceEngine
from oumi.utils.image_utils import load_image_png_bytes_from_path
from tests.integration.infer import get_default_device_map_for_inference
from tests.markers import requires_cuda_initialized


def _get_default_text_model_params() -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
        device_map=get_default_device_map_for_inference(),
    )


def _get_default_image_model_params() -> ModelParams:
    return ModelParams(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="qwen2-vl-instruct",
        device_map=get_default_device_map_for_inference(),
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=5, use_sampling=False, temperature=0.0, min_p=0.0, seed=42
        )
    )


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


#
# Tests
#
def test_infer_online():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
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


def test_infer_online_empty():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
    result = engine.infer_online([], _get_default_inference_config())
    assert [] == result


def test_infer_online_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
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


def test_infer_from_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
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
        result = engine.infer_from_file(
            str(input_path),
            _get_default_inference_config(),
        )
        assert expected_result == result
        inference_config = _get_default_inference_config()
        inference_config.input_path = str(input_path)
        infer_result = engine.infer(inference_config=inference_config)
        assert expected_result == infer_result


def test_infer_from_file_empty():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        inference_config = _get_default_inference_config()
        result = engine.infer_from_file(str(input_path), inference_config)
        assert [] == result
        inference_config.input_path = str(input_path)
        infer_result = engine.infer(inference_config=inference_config)
        assert [] == infer_result


def test_infer_from_file_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
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
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


@requires_cuda_initialized()
@pytest.mark.single_gpu
def test_infer_from_file_to_file_with_images(root_testdata_dir: Path):
    png_image_bytes_great_wave = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "the_great_wave_off_kanagawa.jpg"
    )
    png_image_bytes_logo = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_dark.png"
    )

    test_prompt: str = "Generate a short, descriptive caption for this image!"

    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_image_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=png_image_bytes_great_wave,
                        ),
                        ContentItem(
                            type=Type.TEXT,
                            content=test_prompt,
                        ),
                    ],
                )
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=png_image_bytes_logo,
                        ),
                        ContentItem(
                            type=Type.TEXT,
                            content=test_prompt,
                        ),
                    ],
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1, conversation_2])

        expected_results = []
        for response1, response2 in itertools.product(
            [
                "A traditional Japanese painting of",
                "A detailed Japanese print depicting",
                "A Japanese print depicting a",
            ],
            ["The image features a black"],
        ):
            expected_results.append(
                [
                    Conversation(
                        messages=[
                            *conversation_1.messages,
                            Message(
                                content=response1,
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
                                content=response2,
                                role=Role.ASSISTANT,
                            ),
                        ],
                        metadata={"umi": "bar"},
                        conversation_id="133",
                    ),
                ]
            )

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)

        result = engine.infer_online(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result in expected_results
        idx = expected_results.index(result)
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_results[idx] == parsed_conversations


def test_unsupported_model_raises_error():
    model_params = ModelParams(
        model_name="MlpEncoder",
        tokenizer_name="gpt2",
        tokenizer_pad_token="<|endoftext|>",
        load_pretrained_weights=False,
    )
    with pytest.raises(ValueError, match="requires a generation config"):
        NativeTextInferenceEngine(model_params)
