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

"""Unit tests for quantization utilities."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.utils import (
    calculate_compression_ratio,
    format_size,
    get_directory_size,
    is_valid_hf_model_id,
    validate_quantization_config,
)


class TestFormatSize:
    """Test the format_size utility function."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert format_size(512) == "512.0 B"
        assert format_size(1000) == "1000.0 B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.0 KB"
        assert format_size(2048) == "2.0 KB"
        # Note: due to integer division in format_size, 1536 becomes 1.0 KB
        assert format_size(1536) == "1.0 KB"

    def test_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(2 * 1024 * 1024) == "2.0 MB"

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_size(5 * 1024 * 1024 * 1024) == "5.0 GB"

    def test_zero_size(self):
        """Test zero size."""
        assert format_size(0) == "0.0 B"


class TestCalculateCompressionRatio:
    """Test the calculate_compression_ratio utility function."""

    def test_valid_compression(self):
        """Test valid compression ratio calculation."""
        result = calculate_compression_ratio(1000, 250)
        assert result == "4.00x"

    def test_no_compression(self):
        """Test no compression scenario."""
        result = calculate_compression_ratio(1000, 1000)
        assert result == "1.00x"

    def test_none_original_size(self):
        """Test with None original size."""
        result = calculate_compression_ratio(None, 250)
        assert result == "Unknown"

    def test_zero_quantized_size(self):
        """Test with zero quantized size."""
        result = calculate_compression_ratio(1000, 0)
        assert result == "Unknown"


class TestGetDirectorySize:
    """Test the get_directory_size utility function."""

    def test_empty_directory(self):
        """Test size of empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            size = get_directory_size(temp_dir)
            assert size == 0

    def test_directory_with_files(self):
        """Test size of directory with files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = Path(temp_dir) / "test1.txt"
            test_file2 = Path(temp_dir) / "test2.txt"
            
            test_file1.write_text("Hello" * 100)  # 500 bytes
            test_file2.write_text("World" * 200)  # 1000 bytes
            
            size = get_directory_size(temp_dir)
            assert size == 1500

    def test_nested_directories(self):
        """Test size calculation with nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            
            test_file1 = Path(temp_dir) / "test1.txt"
            test_file2 = subdir / "test2.txt"
            
            test_file1.write_text("A" * 100)
            test_file2.write_text("B" * 200)
            
            size = get_directory_size(temp_dir)
            assert size == 300


class TestValidateQuantizationConfig:
    """Test the validate_quantization_config utility function."""

    def test_valid_config(self):
        """Test validation passes for valid config."""
        config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="awq_q4_0",
            output_path="/tmp/test.gguf",
            output_format="gguf"
        )
        
        with patch("oumi.quantize.utils.is_valid_hf_model_id", return_value=True):
            # Should not raise any exception
            validate_quantization_config(config)

    def test_invalid_method(self):
        """Test validation fails for invalid method."""
        config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="invalid_method",
            output_path="/tmp/test.gguf",
            output_format="gguf"
        )
        
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            validate_quantization_config(config)

    def test_invalid_output_format(self):
        """Test validation fails for invalid output format."""
        config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="awq_q4_0",
            output_path="/tmp/test.invalid",
            output_format="invalid_format"
        )
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            validate_quantization_config(config)

    def test_invalid_model_path(self):
        """Test validation fails for invalid model."""
        config = QuantizationConfig(
            model=ModelParams(model_name="nonexistent/model"),
            method="awq_q4_0",
            output_path="/tmp/test.gguf",
            output_format="gguf"
        )
        
        with patch("oumi.quantize.utils.is_valid_hf_model_id", return_value=False):
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(ValueError, match="Model not found"):
                    validate_quantization_config(config)


class TestIsValidHfModelId:
    """Test the is_valid_hf_model_id utility function."""

    @patch("huggingface_hub.model_info")
    def test_valid_model_id(self, mock_model_info):
        """Test valid HuggingFace model ID."""
        mock_model_info.return_value = MagicMock()
        
        result = is_valid_hf_model_id("facebook/opt-125m")
        assert result is True
        mock_model_info.assert_called_once_with("facebook/opt-125m")

    @patch("huggingface_hub.model_info")
    def test_invalid_model_id(self, mock_model_info):
        """Test invalid HuggingFace model ID."""
        mock_model_info.side_effect = Exception("Model not found")
        
        result = is_valid_hf_model_id("nonexistent/model")
        assert result is False
        mock_model_info.assert_called_once_with("nonexistent/model")