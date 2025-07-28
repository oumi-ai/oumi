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

"""Unit tests for base quantization functionality."""

import pytest  # type: ignore

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult


class TestQuantization(BaseQuantization):
    """Test implementation of BaseQuantization."""

    supported_methods = ["awq_q4_0"]  # Use a real supported method
    supported_formats = ["safetensors", "pytorch"]

    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Test implementation of quantize."""
        return QuantizationResult(
            quantized_size_bytes=1000,
            output_path="/fake/path",
            quantization_method="awq_q4_0",
            format_type="safetensors",
            additional_info={},
        )

    def raise_if_requirements_not_met(self) -> None:
        """Test implementation."""
        pass


class TestBaseQuantization:
    """Test cases for BaseQuantization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_quantizer = TestQuantization()
        self.valid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_path="test_output.safetensors",
            output_format="safetensors",
        )

    def test_supports_method_true(self):
        """Test supports_method returns True for supported method."""
        assert self.test_quantizer.supports_method("awq_q4_0") is True

    def test_supports_method_false(self):
        """Test supports_method returns False for unsupported method."""
        assert self.test_quantizer.supports_method("unsupported") is False

    def test_supports_format_true(self):
        """Test supports_format returns True for supported format."""
        assert self.test_quantizer.supports_format("safetensors") is True
        assert self.test_quantizer.supports_format("pytorch") is True

    def test_supports_format_false(self):
        """Test supports_format returns False for unsupported format."""
        assert self.test_quantizer.supports_format("gguf") is False

    def test_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        # Should not raise any exception
        self.test_quantizer.validate_config(self.valid_config)

    def test_validate_config_invalid_method(self):
        """Test validate_config with invalid method."""
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",  # Valid method but not supported by TestQuantization
            output_path="test.safetensors",
            output_format="safetensors",
        )

        with pytest.raises(ValueError, match="not supported by"):
            self.test_quantizer.validate_config(config)

    def test_quantize_implementation(self):
        """Test the quantize method implementation."""
        result = self.test_quantizer.quantize(self.valid_config)

        assert isinstance(result, QuantizationResult)
        assert result.quantized_size_bytes == 1000
        assert result.output_path == "/fake/path"
        assert result.quantization_method == "awq_q4_0"
        assert result.format_type == "safetensors"


class TestQuantizationResult:
    """Test cases for QuantizationResult dataclass."""

    def test_quantization_result_creation(self):
        """Test creating a QuantizationResult instance."""
        result = QuantizationResult(
            quantized_size_bytes=2048,
            output_path="/path/to/model",
            quantization_method="awq_q4_0",
            format_type="safetensors",
            additional_info={"compression_ratio": 0.25},
        )

        assert result.quantized_size_bytes == 2048
        assert result.output_path == "/path/to/model"
        assert result.quantization_method == "awq_q4_0"
        assert result.format_type == "safetensors"
        assert result.additional_info["compression_ratio"] == 0.25

    def test_quantization_result_default_additional_info(self):
        """Test QuantizationResult with default additional_info."""
        result = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path",
            quantization_method="method",
            format_type="format",
        )

        assert result.additional_info == {}
