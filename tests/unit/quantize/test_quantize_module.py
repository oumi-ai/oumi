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

"""Unit tests for main quantize module functionality."""

from unittest.mock import MagicMock, patch

import pytest  # type: ignore

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize import quantize
from oumi.quantize.base import QuantizationResult


class TestQuantizeModule:
    """Test cases for the main quantize function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.valid_config = QuantizationConfig(
            model=ModelParams(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            method="awq_q4_0",
            output_path="test_model.pytorch",
            output_format="pytorch",
        )

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_awq_success(self, mock_build_quantizer):
        """Test successful quantization with AWQ method."""
        # Mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.return_value = None
        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path/to/model",
            quantization_method="awq_q4_0",
            format_type="pytorch",
            additional_info={"test": "info"},
        )
        mock_build_quantizer.return_value = mock_quantizer

        # Run quantization
        result = quantize(self.valid_config)

        # Verify
        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == "awq_q4_0"
        assert result.format_type == "pytorch"
        assert result.quantized_size_bytes == 1024
        assert result.additional_info["test"] == "info"

        # Verify calls
        mock_build_quantizer.assert_called_once_with("awq_q4_0")
        mock_quantizer.raise_if_requirements_not_met.assert_called_once()
        mock_quantizer.quantize.assert_called_once_with(self.valid_config)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_bnb_success(self, mock_build_quantizer):
        """Test successful quantization with BitsAndBytes method."""
        # Create BNB config
        bnb_config = QuantizationConfig(
            model=ModelParams(model_name="gpt2"),
            method="bnb_4bit",
            output_path="test_model.pytorch",
            output_format="pytorch",
        )

        # Mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.return_value = None
        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=512,
            output_path="/path/to/bnb_model",
            quantization_method="bnb_4bit",
            format_type="pytorch",
        )
        mock_build_quantizer.return_value = mock_quantizer

        # Run quantization
        result = quantize(bnb_config)

        # Verify
        assert result.quantization_method == "bnb_4bit"
        assert result.quantized_size_bytes == 512

        # Verify builder was called with correct method
        mock_build_quantizer.assert_called_once_with("bnb_4bit")

    def test_quantize_invalid_config_type(self):
        """Test quantize with invalid config type."""
        with pytest.raises(ValueError, match="Expected QuantizationConfig"):
            quantize("not a config")  # type: ignore

    def test_quantize_unsupported_method(self):
        """Test quantization with unsupported method."""
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="invalid_method",
                output_path="test.pytorch",
                output_format="pytorch",
            )

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_requirements_not_met(self, mock_build_quantizer):
        """Test quantization when requirements are not met."""
        # Mock quantizer that fails requirements check
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.side_effect = ImportError(
            "Missing required package"
        )
        mock_build_quantizer.return_value = mock_quantizer

        with pytest.raises(ImportError, match="Missing required package"):
            quantize(self.valid_config)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_quantizer_failure(self, mock_build_quantizer):
        """Test when quantizer.quantize() fails."""
        # Mock quantizer that fails during quantization
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.return_value = None
        mock_quantizer.quantize.side_effect = RuntimeError("Quantization failed")
        mock_build_quantizer.return_value = mock_quantizer

        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize(self.valid_config)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_different_output_formats(self, mock_build_quantizer):
        """Test quantization with different output formats."""
        # Mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.return_value = None

        # Test safetensors format
        safetensors_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_path="test.safetensors",
            output_format="safetensors",
        )

        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=2048,
            output_path="/path/to/model.safetensors",
            quantization_method="awq_q4_0",
            format_type="safetensors",
        )
        mock_build_quantizer.return_value = mock_quantizer

        result = quantize(safetensors_config)

        assert result.format_type == "safetensors"
        assert result.output_path.endswith(".safetensors")

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_return_type(self, mock_build_quantizer):
        """Test that quantize always returns QuantizationResult."""
        # Mock quantizer
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.return_value = None
        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=100,
            output_path="/test",
            quantization_method="test",
            format_type="test",
        )
        mock_build_quantizer.return_value = mock_quantizer

        result = quantize(self.valid_config)

        # Verify return type
        assert isinstance(result, QuantizationResult)
        assert hasattr(result, "quantized_size_bytes")
        assert hasattr(result, "output_path")
        assert hasattr(result, "quantization_method")
        assert hasattr(result, "format_type")
        assert hasattr(result, "additional_info")
