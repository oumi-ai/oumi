import tempfile
from pathlib import Path
from typing import List
from unittest.mock import ANY, Mock, patch

import jsonlines
import pytest

from oumi.core.configs import GenerationConfig, ModelParams
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import VLLMInferenceEngine

try:
    vllm_import_failed = False
    from vllm.lora.request import LoRARequest  # type: ignore
    from vllm.outputs import (  # pyright: ignore[reportMissingImports]
        CompletionOutput,
        RequestOutput,
    )

    def _create_vllm_output(responses: List[str], output_id: str) -> RequestOutput:
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


#
# Fixtures
#
@pytest.fixture
def mock_vllm():
    with patch("oumi.inference.vllm_inference_engine.vllm") as mvllm:
        yield mvllm


def _get_default_model_params(use_lora: bool = False) -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        adapter_model="/path/to/adapter" if use_lora else None,
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
    result = engine.infer_online([conversation], GenerationConfig(max_new_tokens=5))
    assert expected_result == result
    mock_vllm_instance.chat.assert_called_once()


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_lora(mock_vllm):
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
    mock_vllm.lora.request.LoRARequest.return_value = lora_request
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
    result = engine.infer_online([conversation], GenerationConfig(max_new_tokens=5))
    assert expected_result == result

    mock_vllm.lora.request.LoRARequest.assert_called_once_with(
        lora_name="oumi_lora_adapter",
        lora_int_id=1,
        lora_path="/path/to/adapter",
    )
    mock_vllm_instance.chat.assert_called_once_with(
        ANY,
        sampling_params=ANY,
        lora_request=lora_request,
    )


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_empty(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    engine = VLLMInferenceEngine(_get_default_model_params())
    result = engine.infer_online([], GenerationConfig(max_new_tokens=5))
    assert [] == result
    mock_vllm_instance.chat.assert_not_called()


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_online_to_file(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.side_effect = [
        [_create_vllm_output(["The first time I saw"], "123")],
        [_create_vllm_output(["The U.S."], "123")],
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
        result = engine.infer_online(
            [conversation_1, conversation_2],
            GenerationConfig(
                max_new_tokens=5,
                output_filepath=str(output_path),
            ),
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.model_validate_json(line))
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
        result = engine.infer_from_file(
            str(input_path), GenerationConfig(max_new_tokens=5)
        )
        assert expected_result == result
        infer_result = engine.infer(
            generation_config=GenerationConfig(
                max_new_tokens=5, input_filepath=str(input_path)
            )
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
        result = engine.infer_from_file(
            str(input_path), GenerationConfig(max_new_tokens=5)
        )
        assert [] == result
        infer_result = engine.infer(
            generation_config=GenerationConfig(
                max_new_tokens=5, input_filepath=str(input_path)
            )
        )
        assert [] == infer_result
        mock_vllm_instance.chat.assert_not_called()


@pytest.mark.skipif(vllm_import_failed, reason="vLLM not available")
def test_infer_from_file_to_file(mock_vllm):
    mock_vllm_instance = Mock()
    mock_vllm.LLM.return_value = mock_vllm_instance
    mock_vllm_instance.chat.side_effect = [
        [_create_vllm_output(["The first time I saw"], "123")],
        [_create_vllm_output(["The U.S."], "123")],
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
        result = engine.infer_online(
            [conversation_1, conversation_2],
            GenerationConfig(
                max_new_tokens=5,
                output_filepath=str(output_path),
            ),
        )
        assert result == expected_result
        # Ensure the final output is in order.
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.model_validate_json(line))
            assert expected_result == parsed_conversations
