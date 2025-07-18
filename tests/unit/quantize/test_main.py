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

"""Unit tests for main quantization logic with new class-based architecture."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.main import quantize
from oumi.quantize.factory import QuantizationFactory


class TestQuantizeFunction:
    """Test the main quantize function with new architecture."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="awq_q4_0",
            output_path=f"{self.temp_dir}/test_output.gguf",
            output_format="gguf"
        )

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.awq_quantizer.AwqQuantization.quantize")
    @patch("oumi.quantize.awq_quantizer.AwqQuantization.validate_requirements")
    def test_awq_simulation_mode(
        self, 
        mock_validate_requirements,
        mock_quantize,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test AWQ quantization in simulation mode."""
        # Setup mocks
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_validate_requirements.return_value = False  # Simulation mode
        mock_quantize.return_value = {
            "quantization_method": "SIMULATED: AWQ → PyTorch (awq_q4_0)",
            "quantized_size": "250.0 MB",
            "quantized_size_bytes": 250000000,
            "simulation_mode": True
        }
        
        # Run quantization
        result = quantize(self.test_config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(self.test_config)
        mock_validate_requirements.assert_called_once()
        mock_quantize.assert_called_once_with(self.test_config)
        
        # Verify result
        assert "quantization_method" in result
        assert "simulation_mode" in result
        assert result["simulation_mode"] is True
        assert "compression_ratio" in result


    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.bnb_quantizer.BitsAndBytesQuantization.quantize")
    @patch("oumi.quantize.bnb_quantizer.BitsAndBytesQuantization.validate_requirements")
    def test_bitsandbytes_quantization(
        self,
        mock_validate_requirements,
        mock_quantize,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test BitsAndBytes quantization."""
        # Setup config and mocks
        config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="bnb_4bit",
            output_path=f"{self.temp_dir}/test_output.pytorch",
            output_format="pytorch"
        )
        
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_validate_requirements.return_value = True
        mock_quantize.return_value = {
            "quantization_method": "BitsAndBytes 4-bit",
            "quantized_size": "250.0 MB", 
            "quantized_size_bytes": 250000000
        }
        
        # Run quantization
        result = quantize(config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(config)
        mock_validate_requirements.assert_called_once()
        mock_quantize.assert_called_once_with(config)
        
        # Verify result
        assert result["quantization_method"] == "BitsAndBytes 4-bit"

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.gguf_quantizer.GgufQuantization.quantize")
    @patch("oumi.quantize.gguf_quantizer.GgufQuantization.validate_requirements")
    def test_gguf_quantization(
        self,
        mock_validate_requirements,
        mock_quantize,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test GGUF quantization."""
        # Setup config and mocks
        config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="q4_0",
            output_path=f"{self.temp_dir}/test_output.gguf",
            output_format="gguf"
        )
        
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_validate_requirements.return_value = True
        mock_quantize.return_value = {
            "quantized_size": "250.0 MB",
            "quantized_size_bytes": 250000000
        }
        
        # Run quantization
        result = quantize(config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(config)
        mock_validate_requirements.assert_called_once()
        mock_quantize.assert_called_once_with(config)
        
        # Verify result has expected keys
        assert "quantized_size" in result

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.factory.QuantizationFactory.create_quantizer")
    def test_quantization_error_handling(
        self, 
        mock_create_quantizer,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test error handling in quantization."""
        # Setup mocks - early calls pass but quantization fails
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_quantizer = MagicMock()
        mock_quantizer.validate_requirements.return_value = True
        mock_quantizer.quantize.side_effect = Exception("Quantization failed")
        mock_create_quantizer.return_value = mock_quantizer
        
        # Verify exception is raised and wrapped in RuntimeError
        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize(self.test_config)

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.main.calculate_compression_ratio")
    @patch("oumi.quantize.factory.QuantizationFactory.create_quantizer")
    def test_compression_ratio_calculation(
        self,
        mock_create_quantizer,
        mock_calc_ratio,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test compression ratio calculation."""
        # Setup mocks
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_calc_ratio.return_value = "4.00x"
        
        # Setup quantizer mock
        mock_quantizer = MagicMock()
        mock_quantizer.validate_requirements.return_value = True
        mock_quantizer.quantize.return_value = {
            "quantized_size_bytes": 250000000,
            "simulation_mode": True
        }
        mock_create_quantizer.return_value = mock_quantizer
        
        result = quantize(self.test_config)
        
        # Verify compression ratio was calculated and added
        mock_calc_ratio.assert_called_once_with(1000000000, 250000000)
        assert result["compression_ratio"] == "4.00x"


class TestQuantizationFactory:
    """Test the QuantizationFactory class."""

    def test_create_awq_quantizer(self):
        """Test creating AWQ quantizer."""
        quantizer = QuantizationFactory.create_quantizer("awq_q4_0")
        from oumi.quantize.awq_quantizer import AwqQuantization
        assert isinstance(quantizer, AwqQuantization)

    def test_create_bnb_quantizer(self):
        """Test creating BitsAndBytes quantizer."""
        quantizer = QuantizationFactory.create_quantizer("bnb_4bit")
        from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
        assert isinstance(quantizer, BitsAndBytesQuantization)

    def test_create_gguf_quantizer(self):
        """Test creating GGUF quantizer."""
        quantizer = QuantizationFactory.create_quantizer("q4_0")
        from oumi.quantize.gguf_quantizer import GgufQuantization
        assert isinstance(quantizer, GgufQuantization)

    def test_unsupported_method(self):
        """Test creating quantizer with unsupported method."""
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            QuantizationFactory.create_quantizer("invalid_method")

    def test_get_available_methods(self):
        """Test getting available methods."""
        methods = QuantizationFactory.get_available_methods()
        assert "AWQ" in methods
        assert "BitsAndBytes" in methods
        assert "GGUF" in methods
        assert "awq_q4_0" in methods["AWQ"]
        assert "bnb_4bit" in methods["BitsAndBytes"]
        assert "q4_0" in methods["GGUF"]

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = QuantizationFactory.get_supported_formats()
        assert "gguf" in formats
        assert "pytorch" in formats
        assert "safetensors" in formats