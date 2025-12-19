# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end integration tests for quantization.

These tests actually quantize a small model, load it, and verify inference works.
They require a GPU and the llmcompressor library to be installed.
"""

import importlib.util
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

# Skip all tests if requirements are not met
llmcompressor_available = importlib.util.find_spec("llmcompressor") is not None
gpu_available = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(
        not llmcompressor_available, reason="llmcompressor is not installed"
    ),
    pytest.mark.skipif(not gpu_available, reason="GPU is required for quantization"),
    pytest.mark.integration,
]

# Use a small model for fast testing
TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture
def output_dir():
    """Create a temporary directory for quantized model output."""
    temp_dir = tempfile.mkdtemp(prefix="oumi_quant_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestLlmCompressorQuantizationE2E:
    """End-to-end tests for llm_compressor quantization."""

    def test_quantize_and_load_w4a16(self, output_dir):
        """Test quantizing a model with W4A16 and loading it for inference."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from oumi.core.configs import ModelParams, QuantizationConfig
        from oumi.quantize import quantize

        output_path = str(Path(output_dir) / "smollm2_w4a16")

        # Configure quantization
        config = QuantizationConfig(
            model=ModelParams(model_name=TEST_MODEL),
            method="llmc_W4A16_ASYM",
            output_path=output_path,
            calibration_samples=8,  # Minimal samples for fast testing
            max_seq_length=512,
        )

        # Run quantization
        result = quantize(config)

        # Verify quantization succeeded
        assert result.quantized_size_bytes > 0
        assert Path(result.output_path).exists()
        assert result.quantization_method == "llmc_W4A16_ASYM"

        # Verify model files were created
        model_files = list(Path(output_path).glob("*.safetensors"))
        assert len(model_files) > 0, "No safetensors files created"

        # Load the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            output_path,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(output_path)

        # Test inference produces reasonable output
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Verify we got a non-empty response that extends the prompt
        assert len(response) > len(prompt), "Model did not generate any new tokens"
        assert response.startswith(prompt), "Response should start with the prompt"

        # Verify the response is coherent (contains actual words, not garbage)
        generated_text = response[len(prompt) :].strip()
        assert len(generated_text) > 0, "No text was generated"
        # Check that it's not just special characters or numbers
        assert any(c.isalpha() for c in generated_text), (
            "Generated text contains no letters"
        )

    def test_quantized_model_size_reduction(self, output_dir):
        """Test that quantization actually reduces model size."""
        from transformers import AutoModelForCausalLM

        from oumi.core.configs import ModelParams, QuantizationConfig
        from oumi.quantize import quantize

        output_path = str(Path(output_dir) / "smollm2_size_test")

        # Get original model size (approximate)
        original_model = AutoModelForCausalLM.from_pretrained(
            TEST_MODEL, torch_dtype=torch.float16
        )
        original_params = sum(p.numel() * 2 for p in original_model.parameters())
        del original_model
        torch.cuda.empty_cache()

        # Quantize
        config = QuantizationConfig(
            model=ModelParams(model_name=TEST_MODEL),
            method="llmc_W4A16_ASYM",
            output_path=output_path,
            calibration_samples=8,
            max_seq_length=512,
        )
        result = quantize(config)

        # 4-bit quantization should reduce size by roughly 4x compared to fp16
        # Allow for some overhead (config files, etc.)
        expected_max_size = original_params * 0.5  # Should be much smaller than 50%
        assert result.quantized_size_bytes < expected_max_size, (
            f"Quantized model ({result.quantized_size_bytes} bytes) is not "
            f"significantly smaller than original ({original_params} bytes)"
        )
