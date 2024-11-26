from importlib.util import find_spec
from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.llama_cpp_inference_engine import LlamaCppInferenceEngine

llama_cpp_import_failed = find_spec("llama_cpp") is None


@pytest.fixture
def mock_llama():
    with patch("oumi.inference.llama_cpp_inference_engine.Llama") as mock:
        yield mock


@pytest.fixture
def inference_engine(mock_llama):
    model_params = ModelParams(
        model_name="test_model.gguf",
        model_max_length=2048,
        model_kwargs={"n_gpu_layers": -1, "n_threads": 4},
    )
    return LlamaCppInferenceEngine(model_params)


@pytest.mark.skipif(llama_cpp_import_failed, reason="llama_cpp not available")
def test_initialization(mock_llama):
    model_params = ModelParams(
        model_name="test_model.gguf",
        model_max_length=2048,
        model_kwargs={"n_gpu_layers": -1, "n_threads": 4},
    )

    with patch("pathlib.Path.exists", return_value=True):
        LlamaCppInferenceEngine(model_params)

    mock_llama.assert_called_once_with(
        model_path="test_model.gguf",
        n_ctx=2048,
        verbose=False,
        n_gpu_layers=-1,
        n_threads=4,
        flash_attn=True,
    )


@pytest.mark.skipif(llama_cpp_import_failed, reason="llama_cpp not available")
def test_convert_conversation_to_llama_input(inference_engine):
    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ]
    )

    result = inference_engine._convert_conversation_to_llama_input(conversation)

    expected = [
        {"content": "Hello", "role": "user"},
        {"content": "Hi there!", "role": "assistant"},
        {"content": "How are you?", "role": "user"},
    ]
    assert result == expected


@pytest.mark.skipif(llama_cpp_import_failed, reason="llama_cpp not available")
def test_infer_online(inference_engine):
    with patch.object(inference_engine, "_infer") as mock_infer:
        mock_infer.return_value = [
            Conversation(messages=[Message(content="Response", role=Role.ASSISTANT)])
        ]

        input_conversations = [
            Conversation(messages=[Message(content="Hello", role=Role.USER)])
        ]
        generation_params = GenerationParams(max_new_tokens=50)

        result = inference_engine.infer_online(input_conversations, generation_params)

        mock_infer.assert_called_once_with(input_conversations, generation_params)
        assert result == mock_infer.return_value


@pytest.mark.skipif(llama_cpp_import_failed, reason="llama_cpp not available")
def test_infer_from_file(inference_engine):
    with (
        patch.object(inference_engine, "_read_conversations") as mock_read,
        patch.object(inference_engine, "_infer") as mock_infer,
    ):
        mock_read.return_value = [
            Conversation(messages=[Message(content="Hello", role=Role.USER)])
        ]
        mock_infer.return_value = [
            Conversation(
                messages=[
                    Message(content="Hello", role=Role.USER),
                    Message(content="Response", role=Role.ASSISTANT),
                ]
            )
        ]

        inference_config = InferenceConfig(
            generation=GenerationParams(max_new_tokens=50), output_path="output.json"
        )

        result = inference_engine.infer_from_file("input.json", inference_config)

        mock_read.assert_called_once_with("input.json")
        mock_infer.assert_called_once()
        assert result == mock_infer.return_value
