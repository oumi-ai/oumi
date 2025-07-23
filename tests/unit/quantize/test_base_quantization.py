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

"""Tests for base quantization classes."""

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult


class MockQuantization(BaseQuantization):
    """Mock quantization class for testing."""

    supported_methods = ["mock_4bit", "mock_8bit"]
    supported_formats = ["pytorch", "safetensors"]

    def raise_if_requirements_not_met(self) -> None:
        """Mock implementation - always passes."""
        pass

    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Mock quantization implementation."""
        return QuantizationResult(
            quantized_size_bytes=1000,
            output_path="/mock/path",
            quantization_method=config.method,
            format_type=config.output_format,
        )


class TestQuantizationResult:
    """Test QuantizationResult dataclass."""

    def test_quantization_result_creation(self):
        """Test creating a QuantizationResult instance."""
        result = QuantizationResult(
            quantized_size_bytes=2048,
            output_path="/test/path",
            quantization_method="test_method",
            format_type="pytorch",
        )

        assert result.quantized_size_bytes == 2048
        assert result.output_path == "/test/path"
        assert result.quantization_method == "test_method"
        assert result.format_type == "pytorch"
        assert result.additional_info == {}

    def test_quantization_result_with_additional_info(self):
        """Test QuantizationResult with additional info."""
        additional_info = {"note": "test note", "metrics": {"accuracy": 0.95}}
        result = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/test/path2",
            quantization_method="test_method2",
            format_type="safetensors",
            additional_info=additional_info,
        )

        assert result.additional_info == additional_info
        assert result.additional_info["note"] == "test note"
        assert result.additional_info["metrics"]["accuracy"] == 0.95


class TestBaseQuantization:
    """Test BaseQuantization abstract base class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = MockQuantization()
        # Use a valid method/format that exists in constants to avoid validation errors
        self.config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",  # Use valid method
            output_format="pytorch",
            output_path="/test/output",
        )

    def test_get_supported_methods(self):
        """Test getting supported methods."""
        methods = self.quantizer.get_supported_methods()
        assert methods == ["mock_4bit", "mock_8bit"]
        # Ensure it returns a copy
        methods.append("new_method")
        assert self.quantizer.get_supported_methods() == ["mock_4bit", "mock_8bit"]

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = self.quantizer.get_supported_formats()
        assert formats == ["pytorch", "safetensors"]
        # Ensure it returns a copy
        formats.append("new_format")
        assert self.quantizer.get_supported_formats() == ["pytorch", "safetensors"]

    def test_supports_method(self):
        """Test method support checking."""
        assert self.quantizer.supports_method("mock_4bit") is True
        assert self.quantizer.supports_method("mock_8bit") is True
        assert self.quantizer.supports_method("unsupported_method") is False

    def test_supports_format(self):
        """Test format support checking."""
        assert self.quantizer.supports_format("pytorch") is True
        assert self.quantizer.supports_format("safetensors") is True
        assert self.quantizer.supports_format("unsupported_format") is False

    def test_validate_config_valid_method_and_format(self):
        """Test config validation with valid method and format."""
        # Should not raise any exception
        self.quantizer.validate_config(self.config)

    def test_validate_config_invalid_method(self):
        """Test config validation with invalid method."""
        invalid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="invalid_method",
            output_format="pytorch",
            output_path="/test/output",
        )

        with pytest.raises(ValueError, match="Method 'invalid_method' not supported"):
            self.quantizer.validate_config(invalid_config)

    def test_validate_config_invalid_format(self):
        """Test config validation with invalid format."""
        invalid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="mock_4bit",
            output_format="invalid_format",
            output_path="/test/output",
        )

        with pytest.raises(ValueError, match="Format 'invalid_format' not supported"):
            self.quantizer.validate_config(invalid_config)

    def test_validate_requirements_success(self):
        """Test successful requirements validation."""
        assert self.quantizer.validate_requirements() is True

    def test_quantize_returns_result(self):
        """Test that quantize returns a QuantizationResult."""
        result = self.quantizer.quantize(self.config)

        assert isinstance(result, QuantizationResult)
        assert result.quantized_size_bytes == 1000
        assert result.output_path == "/mock/path"
        assert result.quantization_method == "bnb_4bit"  # Updated to match config
        assert result.format_type == "pytorch"


class TestBaseQuantizationAbstract:
    """Test that abstract methods raise NotImplementedError."""

    def test_abstract_methods_not_implemented(self):
        """Test that BaseQuantization cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseQuantization()  # type: ignore
