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
from unittest.mock import Mock, patch
from pathlib import Path

from oumi.core.configs import QuantizationConfig, ModelParams, GenerationParams
from oumi.quantize.base import BaseQuantization, QuantizationResult


class TestQuantization(BaseQuantization):
    """Test implementation of BaseQuantization."""
    
    supported_methods = ["awq_q4_0"]
    supported_formats = ["safetensors"]
    
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Test implementation of quantize."""
        return QuantizationResult(
            quantized_size_bytes=1000,
            output_path="/fake/path",
            quantization_method="awq_q4_0",
            format_type="safetensors",
            additional_info={}
        )
    
    def raise_if_requirements_not_met(self) -> None:
        """Test implementation - no requirements to check."""
        pass


class TestBaseQuantization:
    """Test cases for BaseQuantization class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_quantizer = TestQuantization()
        self.valid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_path="/tmp/output",
            output_format="safetensors"
        )
    
    def test_supports_method_true(self):
        """Test that supports_method returns True for supported methods."""
        assert self.test_quantizer.supports_method("awq_q4_0") is True
    
    def test_supports_method_false(self):
        """Test that supports_method returns False for unsupported methods."""
        assert self.test_quantizer.supports_method("unsupported_method") is False
    
    def test_supports_format_true(self):
        """Test that supports_format returns True for supported formats."""
        assert self.test_quantizer.supports_format("safetensors") is True
    
    def test_supports_format_false(self):
        """Test that supports_format returns False for unsupported formats."""
        assert self.test_quantizer.supports_format("unsupported_format") is False
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise any exception
        self.test_quantizer.validate_config(self.valid_config)
    
    def test_validate_config_invalid_method(self):
        """Test config validation with invalid method."""
        # We can't easily test invalid method validation because QuantizationConfig
        # itself validates methods. Instead test unsupported method for our test quantizer
        # by creating a config that passes QuantizationConfig validation but not our quantizer
        valid_config_different_method = QuantizationConfig(
            model=ModelParams(model_name="test/model"), 
            method="bnb_4bit",
            output_path="/tmp/output",
            output_format="safetensors"
        )
        
        with pytest.raises(ValueError, match="Method 'bnb_4bit' not supported"):
            self.test_quantizer.validate_config(valid_config_different_method)
    
    def test_quantize_implementation(self):
        """Test the quantize method implementation."""
        result = self.test_quantizer.quantize(self.valid_config)
        
        assert isinstance(result, QuantizationResult)
        assert result.quantized_size_bytes == 1000
        assert result.output_path == "/fake/path"
        assert result.quantization_method == "awq_q4_0"
        assert result.format_type == "safetensors"
        assert result.additional_info == {}
    
    @patch('oumi.quantize.base.Path.mkdir')
    def test_ensure_output_directory_exists(self, mock_mkdir):
        """Test that output directory is created if it doesn't exist."""
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_path="/tmp/new_dir/output.bin",
            output_format="safetensors"
        )
        
        self.test_quantizer.validate_config(config)
        
        # Verify mkdir was called with parents=True and exist_ok=True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestQuantizationResult:
    """Test cases for QuantizationResult dataclass."""
    
    def test_quantization_result_creation(self):
        """Test creating a QuantizationResult instance."""
        result = QuantizationResult(
            quantized_size_bytes=2048,
            output_path="/path/to/model",
            quantization_method="awq_4bit",
            format_type="safetensors",
            additional_info={"compression_ratio": 0.25}
        )
        
        assert result.quantized_size_bytes == 2048
        assert result.output_path == "/path/to/model"
        assert result.quantization_method == "awq_4bit"
        assert result.format_type == "safetensors"
        assert result.additional_info["compression_ratio"] == 0.25
    
    def test_quantization_result_equality(self):
        """Test QuantizationResult equality comparison."""
        result1 = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path",
            quantization_method="method",
            format_type="format",
            additional_info={}
        )
        
        result2 = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path",
            quantization_method="method",
            format_type="format",
            additional_info={}
        )
        
        assert result1 == result2
    
    def test_quantization_result_inequality(self):
        """Test QuantizationResult inequality comparison."""
        result1 = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path",
            quantization_method="method1",
            format_type="format",
            additional_info={}
        )
        
        result2 = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path",
            quantization_method="method2",
            format_type="format",
            additional_info={}
        )
        
        assert result1 != result2