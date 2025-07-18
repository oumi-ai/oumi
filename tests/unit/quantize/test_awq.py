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

"""Unit tests for AWQ quantization functionality."""

import tempfile
from pathlib import Path

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.awq_quantizer import AwqQuantization


class TestValidateAwqRequirements:
    """Test AWQ dependency validation."""

    def test_validate_awq_requirements_function_exists(self):
        """Test that validate_awq_requirements method exists and is callable."""
        quantizer = AwqQuantization()
        assert callable(quantizer.validate_requirements)

    def test_validate_awq_requirements_returns_bool(self):
        """Test that method returns boolean."""
        quantizer = AwqQuantization()
        result = quantizer.validate_requirements()
        # Should return bool
        assert isinstance(result, bool)


class TestSimulateAwqQuantization:
    """Test AWQ quantization simulation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = QuantizationConfig(
            model=ModelParams(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            method="awq_q4_0",
            output_path=f"{self.temp_dir}/test_output.gguf",
            output_format="gguf",
        )

    def test_simulate_awq_small_model(self):
        """Test simulation with small model."""
        config = QuantizationConfig(
            model=ModelParams(model_name="small-model/test"),
            method="awq_q4_0",
            output_path=f"{self.temp_dir}/small_output.gguf",
            output_format="gguf",
        )

        quantizer = AwqQuantization()
        result = quantizer._simulate_quantization(config)

        # Verify result structure
        assert "quantization_method" in result
        assert "simulation_mode" in result
        assert "awq_dependencies_missing" in result
        assert result["simulation_mode"] is True
        assert result["awq_dependencies_missing"] is True

        # Verify output file was created
        output_file = Path(config.output_path)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_simulate_awq_7b_model_q4(self):
        """Test simulation with 7B model using Q4."""
        config = QuantizationConfig(
            model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
            method="awq_q4_0",
            output_path=f"{self.temp_dir}/llama7b_q4_output.gguf",
            output_format="gguf",
        )

        quantizer = AwqQuantization()
        result = quantizer._simulate_quantization(config)

        # Verify larger mock size for 7B model
        output_file = Path(config.output_path)
        assert output_file.exists()
        # Should be larger than small model
        assert output_file.stat().st_size > 100 * 1024 * 1024  # > 100MB

    def test_simulate_awq_7b_model_q8(self):
        """Test simulation with 7B model using Q8."""
        config = QuantizationConfig(
            model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
            method="awq_q8_0",
            output_path=f"{self.temp_dir}/llama7b_q8_output.gguf",
            output_format="gguf",
        )

        quantizer = AwqQuantization()
        result = quantizer._simulate_quantization(config)

        # Verify Q8 is larger than Q4 for same model
        output_file = Path(config.output_path)
        assert output_file.exists()
        # Q8 should be larger than Q4
        assert output_file.stat().st_size > 200 * 1024 * 1024  # > 200MB

    def test_simulate_creates_parent_directories(self):
        """Test that simulation creates parent directories."""
        nested_path = f"{self.temp_dir}/nested/deep/path/output.gguf"
        config = QuantizationConfig(
            model=ModelParams(model_name="test-model"),
            method="awq_q4_0",
            output_path=nested_path,
            output_format="gguf",
        )

        quantizer = AwqQuantization()
        result = quantizer._simulate_quantization(config)

        # Verify nested directories were created
        output_file = Path(nested_path)
        assert output_file.exists()
        assert output_file.parent.exists()

    def test_simulate_result_format(self):
        """Test that simulation result has correct format."""
        quantizer = AwqQuantization()
        result = quantizer._simulate_quantization(self.test_config)

        # Verify all required keys are present
        required_keys = [
            "quantization_method",
            "quantized_size",
            "quantized_size_bytes",
            "output_path",
            "simulation_mode",
            "awq_dependencies_missing",
        ]

        for key in required_keys:
            assert key in result

        # Verify types
        assert isinstance(result["quantized_size"], str)
        assert isinstance(result["quantized_size_bytes"], int)
        assert isinstance(result["output_path"], str)
        assert isinstance(result["simulation_mode"], bool)
        assert isinstance(result["awq_dependencies_missing"], bool)

        # Verify quantization method mentions simulation
        assert "SIMULATED" in result["quantization_method"]
        assert self.test_config.method in result["quantization_method"]
