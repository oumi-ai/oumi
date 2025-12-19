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

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult


class MockQuantization(BaseQuantization):
    """Mock implementation of BaseQuantization for testing."""

    supported_methods = ["llmc_W4A16_ASYM"]
    supported_formats = ["safetensors"]

    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        return QuantizationResult(
            quantized_size_bytes=1000,
            output_path=config.output_path,
            quantization_method=config.method,
            format_type=config.output_format,
        )

    def raise_if_requirements_not_met(self) -> None:
        pass


class TestBaseQuantization:
    """Test cases for BaseQuantization functionality."""

    def test_validate_config_rejects_unsupported_method(self):
        """Test that validate_config rejects methods not in supported_methods."""
        quantizer = MockQuantization()
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_path="test",
        )
        with pytest.raises(ValueError, match="not supported by"):
            quantizer.validate_config(config)

    def test_supports_method(self):
        """Test supports_method returns correct boolean."""
        quantizer = MockQuantization()
        assert quantizer.supports_method("llmc_W4A16_ASYM") is True
        assert quantizer.supports_method("unsupported") is False


class TestQuantizationConfig:
    """Test cases for QuantizationConfig validation."""

    def test_rejects_unsupported_method(self):
        """Test that config rejects unsupported quantization methods."""
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="invalid_method",
                output_path="test",
            )

    def test_rejects_unsupported_format(self):
        """Test that config rejects unsupported output formats."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="llmc_W4A16_ASYM",
                output_path="test",
                output_format="unknown_format",
            )
