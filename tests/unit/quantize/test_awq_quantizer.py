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

"""Unit tests for AWQ quantization."""

from unittest.mock import MagicMock, patch

import pytest  # type: ignore

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.awq_quantizer import AwqQuantization


class TestAwqQuantization:
    """Test cases for AWQ quantization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.awq_quantizer = AwqQuantization()
        self.valid_config = QuantizationConfig(
            model=ModelParams(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            method="awq_q4_0",
            output_path="test_model",
        )

    def test_supported_methods(self):
        """Test AWQ supported methods."""
        expected_methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
        assert self.awq_quantizer.supported_methods == expected_methods

    def test_supported_formats(self):
        """Test AWQ supported output formats."""
        expected_formats = ["safetensors"]
        assert self.awq_quantizer.supported_formats == expected_formats

    def test_supports_method_valid(self):
        """Test supports_method for valid AWQ methods."""
        assert self.awq_quantizer.supports_method("awq_q4_0") is True
        assert self.awq_quantizer.supports_method("awq_q4_1") is True
        assert self.awq_quantizer.supports_method("awq_q8_0") is True
        assert self.awq_quantizer.supports_method("awq_f16") is True

    def test_supports_method_invalid(self):
        """Test supports_method for invalid methods."""
        assert self.awq_quantizer.supports_method("bnb_4bit") is False
        assert self.awq_quantizer.supports_method("q4_0") is False
        assert self.awq_quantizer.supports_method("invalid") is False

    def test_validate_config_valid(self):
        """Test validate_config with valid AWQ configuration."""
        # Should not raise any exception
        self.awq_quantizer.validate_config(self.valid_config)

    def test_validate_config_invalid_method(self):
        """Test validate_config with non-AWQ method."""
        invalid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_path="test",
        )

        with pytest.raises(ValueError, match="not supported by"):
            self.awq_quantizer.validate_config(invalid_config)

    def test_validate_config_invalid_format(self):
        """Test validate_config with invalid format."""

        with pytest.raises(ValueError, match="Unsupported output format"):
            QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="awq_q4_0",
                output_path="test",
                output_format="unknown",
            )

    def test_str_representation(self):
        """Test string representation of AWQ quantizer."""
        assert self.awq_quantizer.__class__.__name__ == "AwqQuantization"

    def test_raise_if_requirements_not_met_missing_awq(self):
        """Test requirements check when AWQ is not installed."""
        # Set _awq to None to simulate missing AWQ
        self.awq_quantizer._awq = None

        with pytest.raises(
            RuntimeError, match="AWQ quantization requires autoawq library"
        ):
            self.awq_quantizer.raise_if_requirements_not_met()

    def test_raise_if_requirements_not_met_no_gpu(self):
        """Test requirements check when no GPU is available."""
        # Set _awq to a mock to simulate AWQ is installed
        self.awq_quantizer._awq = MagicMock()

        with patch("torch.cuda.is_available") as mock_cuda:
            mock_cuda.return_value = False

            with pytest.raises(RuntimeError, match="AWQ quantization requires a GPU"):
                self.awq_quantizer.raise_if_requirements_not_met()


class TestAwqQuantizationSimple:
    """Additional simplified test cases for AWQ quantization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.awq_quantizer = AwqQuantization()

    def test_supports_method(self):
        """Test supports_method for AWQ methods."""
        assert self.awq_quantizer.supports_method("awq_q4_0") is True
        assert self.awq_quantizer.supports_method("awq_q8_0") is True
        assert self.awq_quantizer.supports_method("bnb_4bit") is False

    def test_validate_config_valid_simple(self):
        """Test validate_config with valid configuration."""
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_path="test",
        )
        # Should not raise
        self.awq_quantizer.validate_config(config)

    def test_raise_if_requirements_not_met_no_awq_simple(self):
        """Test requirements check when AWQ is not available."""
        self.awq_quantizer._awq = None
        with pytest.raises(RuntimeError, match="requires autoawq library"):
            self.awq_quantizer.raise_if_requirements_not_met()

    @patch("torch.cuda.is_available")
    def test_raise_if_requirements_not_met_no_gpu_simple(self, mock_cuda):
        """Test requirements check when no GPU is available."""
        mock_cuda.return_value = False
        self.awq_quantizer._awq = MagicMock()  # AWQ is available
        with pytest.raises(RuntimeError, match="requires a GPU"):
            self.awq_quantizer.raise_if_requirements_not_met()
