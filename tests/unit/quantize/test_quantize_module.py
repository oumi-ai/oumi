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

"""Unit tests for quantize module main function."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize import quantize
from oumi.quantize.base import QuantizationResult


class TestQuantizeModule:
    """Test cases for the main quantize function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        self.valid_awq_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="awq_4bit",
            output_path=os.path.join(self.temp_dir, "model_awq"),
            output_format="safetensors",
        )

        self.valid_bnb_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="bnb_4bit",
            output_path=os.path.join(self.temp_dir, "model_bnb"),
            output_format="safetensors",
        )

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_awq_success(self, mock_build_quantizer):
        """Test successful AWQ quantization through main function."""
        # Setup mock quantizer
        mock_quantizer = Mock()
        mock_result = QuantizationResult(
            quantized_size_bytes=125000000,
            output_path=self.valid_awq_config.output_path,
            quantization_method="awq_4bit",
            format_type="safetensors",
            additional_info={},
        )
        mock_quantizer.quantize.return_value = mock_result
        mock_build_quantizer.return_value = mock_quantizer

        # Run quantization
        result = quantize(self.valid_awq_config)

        # Verify build_quantizer was called with correct method
        mock_build_quantizer.assert_called_once_with("awq_4bit")

        # Verify quantizer.quantize was called with config
        mock_quantizer.quantize.assert_called_once_with(self.valid_awq_config)

        # Verify result is returned correctly
        assert result == mock_result
        assert result.quantization_method == "awq_4bit"
        assert result.quantized_size_bytes == 125000000

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_bnb_success(self, mock_build_quantizer):
        """Test successful BnB quantization through main function."""
        # Setup mock quantizer
        mock_quantizer = Mock()
        mock_result = QuantizationResult(
            quantized_size_bytes=62500000,
            output_path=self.valid_bnb_config.output_path,
            quantization_method="bnb_4bit",
            format_type="safetensors",
            additional_info={},
        )
        mock_quantizer.quantize.return_value = mock_result
        mock_build_quantizer.return_value = mock_quantizer

        # Run quantization
        result = quantize(self.valid_bnb_config)

        # Verify build_quantizer was called with correct method
        mock_build_quantizer.assert_called_once_with("bnb_4bit")

        # Verify quantizer.quantize was called with config
        mock_quantizer.quantize.assert_called_once_with(self.valid_bnb_config)

        # Verify result is returned correctly
        assert result == mock_result
        assert result.quantization_method == "bnb_4bit"
        assert result.quantized_size_bytes == 62500000

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_unsupported_method(self, mock_build_quantizer):
        """Test quantization with unsupported method."""
        # Setup config with unsupported method
        unsupported_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="unsupported_method",
            output_path=os.path.join(self.temp_dir, "model"),
            output_format="safetensors",
        )

        # Setup mock to raise ValueError for unsupported method
        mock_build_quantizer.side_effect = ValueError(
            "Unsupported quantization method: unsupported_method"
        )

        # Run quantization and expect failure
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            quantize(unsupported_config)

        # Verify build_quantizer was called
        mock_build_quantizer.assert_called_once_with("unsupported_method")

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_quantizer_failure(self, mock_build_quantizer):
        """Test quantization failure from quantizer."""
        # Setup mock quantizer to raise exception
        mock_quantizer = Mock()
        mock_quantizer.quantize.side_effect = RuntimeError("Quantization failed")
        mock_build_quantizer.return_value = mock_quantizer

        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize(self.valid_awq_config)

        # Verify build_quantizer was called
        mock_build_quantizer.assert_called_once_with("awq_4bit")

        # Verify quantizer.quantize was called
        mock_quantizer.quantize.assert_called_once_with(self.valid_awq_config)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_builder_failure(self, mock_build_quantizer):
        """Test quantization failure from builder."""
        # Setup mock to raise exception during builder creation
        mock_build_quantizer.side_effect = ImportError("Failed to import quantizer")

        # Run quantization and expect failure
        with pytest.raises(ImportError, match="Failed to import quantizer"):
            quantize(self.valid_awq_config)

        # Verify build_quantizer was called
        mock_build_quantizer.assert_called_once_with("awq_4bit")

    def test_quantize_invalid_config_type(self):
        """Test quantization with invalid config type."""
        # Pass invalid config (not QuantizationConfig)
        with pytest.raises(AttributeError):
            quantize("invalid_config")

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_with_different_configs(self, mock_build_quantizer):
        """Test quantization with various configuration options."""
        test_configs = [
            # AWQ with trust_remote_code
            QuantizationConfig(
                model=ModelParams(model_name="custom/model", trust_remote_code=True),
                method="awq_q4_0",
                output_path=os.path.join(self.temp_dir, "model1"),
                output_format="safetensors",
            ),
            # BnB 8-bit
            QuantizationConfig(
                model=ModelParams(model_name="microsoft/DialoGPT-medium"),
                method="bnb_8bit",
                output_path=os.path.join(self.temp_dir, "model2"),
                output_format="safetensors",
            ),
        ]

        for config in test_configs:
            # Setup mock quantizer
            mock_quantizer = Mock()
            mock_result = QuantizationResult(
                quantized_size_bytes=100000000,
                output_path=config.output_path,
                quantization_method=config.method,
                format_type="safetensors",
                additional_info={},
            )
            mock_quantizer.quantize.return_value = mock_result
            mock_build_quantizer.return_value = mock_quantizer

            # Run quantization
            result = quantize(config)

            # Verify correct method was requested
            mock_build_quantizer.assert_called_with(config.method)

            # Verify quantizer was called with correct config
            mock_quantizer.quantize.assert_called_with(config)

            # Verify result
            assert result.quantization_method == config.method
            assert result.output_path == config.output_path

            # Reset mock for next iteration
            mock_build_quantizer.reset_mock()

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_return_type(self, mock_build_quantizer):
        """Test that quantize function returns correct type."""
        # Setup mock quantizer
        mock_quantizer = Mock()
        mock_result = QuantizationResult(
            quantized_size_bytes=125000000,
            output_path=self.valid_awq_config.output_path,
            quantization_method="awq_4bit",
            format_type="safetensors",
            additional_info={"compression_ratio": 0.25},
        )
        mock_quantizer.quantize.return_value = mock_result
        mock_build_quantizer.return_value = mock_quantizer

        # Run quantization
        result = quantize(self.valid_awq_config)

        # Verify return type
        assert isinstance(result, QuantizationResult)
        assert hasattr(result, "quantized_size_bytes")
        assert hasattr(result, "output_path")
        assert hasattr(result, "quantization_method")
        assert hasattr(result, "format_type")
        assert hasattr(result, "additional_info")

        # Verify values
        assert result.additional_info["compression_ratio"] == 0.25
