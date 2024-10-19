import pathlib
import tempfile
from pathlib import Path
from typing import Final, List

import jsonlines

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference import NativeTextInferenceEngine
from oumi.utils.image_utils import load_image_png_bytes_from_path

OUMI_ROOT_DIR: Final[Path] = (
    pathlib.Path(__file__).resolve().parent.parent.parent.parent
)
TEST_IMAGE_DIR: Final[Path] = OUMI_ROOT_DIR / "tests" / "testdata" / "images"


def _get_default_text_model_params() -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
    )


def _get_default_image_model_params() -> ModelParams:
    return ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="llava",
    )


def _setup_input_conversations(filepath: str, conversations: List[Conversation]):
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
    result = engine.infer_online(
        [conversation], GenerationParams(max_new_tokens=5, temperature=0.0, seed=42)
    )
    assert expected_result == result


def test_infer_online_empty():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
    result = engine.infer_online(
        [], GenerationParams(max_new_tokens=5, temperature=0.0, seed=42)
    )
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
        result = engine.infer_online(
            [conversation_1, conversation_2],
            GenerationParams(
                max_new_tokens=5,
                temperature=0.0,
                seed=42,
                output_filepath=str(output_path),
            ),
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
            GenerationParams(max_new_tokens=5, temperature=0.0, seed=42),
        )
        assert expected_result == result
        infer_result = engine.infer(
            generation_params=GenerationParams(
                max_new_tokens=5,
                temperature=0.0,
                seed=42,
                input_filepath=str(input_path),
            )
        )
        assert expected_result == infer_result


def test_infer_from_file_empty():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        result = engine.infer_from_file(
            str(input_path), GenerationParams(max_new_tokens=5)
        )
        assert [] == result
        infer_result = engine.infer(
            generation_params=GenerationParams(
                max_new_tokens=5, input_filepath=str(input_path)
            )
        )
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
        result = engine.infer_online(
            [conversation_1, conversation_2],
            GenerationParams(
                max_new_tokens=5,
                output_filepath=str(output_path),
                temperature=0.0,
                seed=42,
            ),
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


def test_infer_from_file_to_file_with_images():
    png_image_bytes_cambrian = load_image_png_bytes_from_path(
        TEST_IMAGE_DIR / "cambrian.png"
    )
    png_image_bytes_math = load_image_png_bytes_from_path(TEST_IMAGE_DIR / "math.png")

    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_image_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    type=Type.IMAGE_BINARY,
                    binary=png_image_bytes_cambrian,
                ),
                Message(
                    content="Describe the high-level theme of the image in few words!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    type=Type.IMAGE_BINARY,
                    binary=png_image_bytes_math,
                ),
                Message(
                    content="Describe the high-level theme of the image in few words!",
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
                        content="3D underwater scene",
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
                        content="25 is the value",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        result = engine.infer_online(
            [conversation_1, conversation_2],
            GenerationParams(
                max_new_tokens=5,
                output_filepath=str(output_path),
                temperature=0.0,
                seed=42,
            ),
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations
