import tempfile
from pathlib import Path
from typing import List

import jsonlines
import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import NativeTextInferenceEngine


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        trust_remote_code=True,
    )


def _setup_input_conversations(filepath: str, conversations: List[Conversation]):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).touch()
    with jsonlines.open(filepath, mode="w") as writer:
        for conversation in conversations:
            json_obj = conversation.model_dump()
            writer.write(json_obj)
    # Add some empty lines into the file
    with open(filepath, "a") as f:
        f.write("\n\n\n")


@pytest.fixture
def conversation_1():
    return Conversation(
        messages=[
            Message(
                content="Hello!",
                role=Role.USER,
            ),
            Message(
                content="Hi, how can I help you?",
                role=Role.ASSISTANT,
            ),
            Message(
                content="What is the capital of the France ?",
                role=Role.USER,
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )


@pytest.fixture
def conversation_2():
    return Conversation(
        messages=[
            Message(
                content="Where is Seattle ?",
                role=Role.USER,
            ),
        ],
        metadata={"umi": "bar"},
        conversation_id="133",
    )


@pytest.fixture
def sample_conversations(conversation_1, conversation_2):
    return [conversation_1, conversation_2]


@pytest.fixture
def expected_result_1(conversation_1):
    return Conversation(
        messages=[
            *conversation_1.messages,
            Message(
                content="The capital of France is",
                role=Role.ASSISTANT,
            ),
        ],
        metadata=conversation_1.metadata,
        conversation_id=conversation_1.conversation_id,
    )


@pytest.fixture
def expected_result_2(conversation_2):
    return Conversation(
        messages=[
            *conversation_2.messages,
            Message(
                content="\nSeattle, located",
                role=Role.ASSISTANT,
            ),
        ],
        metadata=conversation_2.metadata,
        conversation_id=conversation_2.conversation_id,
    )


@pytest.fixture
def expected_results(expected_result_1, expected_result_2):
    return [expected_result_1, expected_result_2]


#
# Tests
#
def test_infer_online(conversation_1, expected_result_1):
    engine = NativeTextInferenceEngine(_get_default_model_params())
    result = engine.infer_online(
        [conversation_1],
        GenerationParams(
            max_new_tokens=5,
            temperature=0.0,
            seed=42,
        ),
    )
    assert [expected_result_1] == result


def test_infer_online_empty():
    engine = NativeTextInferenceEngine(_get_default_model_params())
    result = engine.infer_online(
        [], GenerationParams(max_new_tokens=5, temperature=0.0, seed=42)
    )
    assert [] == result


def test_infer_online_to_file(sample_conversations, expected_results):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_model_params())

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        result = engine.infer_online(
            sample_conversations,
            GenerationParams(
                max_new_tokens=5,
                temperature=0.0,
                seed=42,
                output_filepath=str(output_path),
            ),
        )
        assert result == expected_results
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.model_validate_json(line))
            assert expected_results == parsed_conversations


def test_infer_from_file(conversation_1, expected_result_1):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_model_params())
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1])
        result = engine.infer_from_file(
            str(input_path),
            GenerationParams(max_new_tokens=5, temperature=0.0, seed=42),
        )
        assert [expected_result_1] == result
        infer_result = engine.infer(
            generation_params=GenerationParams(
                max_new_tokens=5,
                temperature=0.0,
                seed=42,
                input_filepath=str(input_path),
            )
        )
        assert [expected_result_1] == infer_result


def test_infer_from_file_empty():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = NativeTextInferenceEngine(_get_default_model_params())
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


def test_infer_from_file_to_file(sample_conversations, expected_results):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_model_params())
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), sample_conversations)

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        result = engine.infer_online(
            sample_conversations,
            GenerationParams(
                max_new_tokens=5,
                output_filepath=str(output_path),
                temperature=0.0,
                seed=42,
            ),
        )
        assert result == expected_results
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.model_validate_json(line))
            assert expected_results == parsed_conversations
