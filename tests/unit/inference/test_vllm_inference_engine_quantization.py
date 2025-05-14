from unittest.mock import ANY, patch

import pytest

from oumi.core.configs import ModelParams
from oumi.inference import VLLMInferenceEngine


def _mock_gguf(repo_id: str, filename: str) -> str:
    return "MOCK[{repo_id}|{filename}]"


def test_gguf():
    model_name = "my_model_name"
    model_kwargs = {"filename": "my_model_filename.gguf"}
    tokenizer_name = "gpt2"

    expected_model_name = _mock_gguf(model_name, model_kwargs["filename"])
    expected_tokenizer_name = tokenizer_name
    expected_quantization = "gguf"

    with (
        patch("oumi.inference.vllm_inference_engine.vllm.LLM") as mock_LLM,
        patch(
            "oumi.inference.vllm_inference_engine.get_local_filepath_for_gguf",
            _mock_gguf,
        ),
    ):
        VLLMInferenceEngine(
            ModelParams(
                model_name=model_name,
                model_kwargs=model_kwargs,
                tokenizer_name=tokenizer_name,
            )
        )
        mock_LLM.assert_called_once_with(
            model=expected_model_name,
            tokenizer=expected_tokenizer_name,
            quantization=expected_quantization,
            # Params not relevant to this test.
            trust_remote_code=ANY,
            dtype=ANY,
            tensor_parallel_size=ANY,
            enable_prefix_caching=ANY,
            enable_lora=ANY,
            max_model_len=ANY,
            gpu_memory_utilization=ANY,
            enforce_eager=ANY,
        )


def test_gguf_no_tokenizer():
    model_name = "my_model_name"
    model_kwargs = {"filename": "my_model_filename.gguf"}

    with patch(
        "oumi.inference.vllm_inference_engine.get_local_filepath_for_gguf",
        _mock_gguf,
    ):
        with pytest.raises(
            ValueError,
            match="GGUF quantization with the VLLM engine requires that you "
            "explicitly set the `tokenizer_name` in `model_params`.",
        ):
            VLLMInferenceEngine(
                ModelParams(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_name=None,
                )
            )


def test_bnb():
    model_name = "my_model_name"
    model_kwargs = {"load_in_8bit": True}
    tokenizer_name = "gpt2"

    expected_model_name = model_name
    expected_tokenizer_name = tokenizer_name
    expected_quantization = "bitsandbytes"
    expected_load_format = "bitsandbytes"

    with patch("oumi.inference.vllm_inference_engine.vllm.LLM") as mock_LLM:
        VLLMInferenceEngine(
            ModelParams(
                model_name=model_name,
                model_kwargs=model_kwargs,
                tokenizer_name=tokenizer_name,
            )
        )
        mock_LLM.assert_called_once_with(
            model=expected_model_name,
            tokenizer=expected_tokenizer_name,
            quantization=expected_quantization,
            load_format=expected_load_format,
            # Params not relevant to this test.
            trust_remote_code=ANY,
            dtype=ANY,
            tensor_parallel_size=ANY,
            enable_prefix_caching=ANY,
            enable_lora=ANY,
            max_model_len=ANY,
            gpu_memory_utilization=ANY,
            enforce_eager=ANY,
        )
