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

"""Unit tests for main quantization logic."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.main import quantize


class TestQuantizeFunction:
    """Test the main quantize function."""

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
    @patch("oumi.quantize.main.validate_awq_requirements")
    @patch("oumi.quantize.main.simulate_awq_quantization")
    def test_awq_simulation_mode(
        self, 
        mock_simulate,
        mock_validate_awq,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test AWQ quantization in simulation mode."""
        # Setup mocks
        mock_validate_awq.return_value = False  # No AWQ available
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_simulate.return_value = {
            "quantization_method": "SIMULATED: AWQ → PyTorch (awq_q4_0)",
            "quantized_size": "250.0 MB",
            "quantized_size_bytes": 250000000,
            "simulation_mode": True
        }
        
        # Run quantization
        result = quantize(self.test_config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(self.test_config)
        mock_validate_awq.assert_called_once()
        mock_simulate.assert_called_once_with(self.test_config)
        
        # Verify result
        assert "quantization_method" in result
        assert "simulation_mode" in result
        assert result["simulation_mode"] is True
        assert "compression_ratio" in result

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.main.validate_awq_requirements")
    @patch("oumi.quantize.main.quantize_awq_fallback_to_pytorch")
    def test_awq_fallback_mode(
        self,
        mock_fallback,
        mock_validate_awq,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test AWQ quantization with BitsAndBytes fallback."""
        # Setup mocks
        mock_validate_awq.return_value = "bitsandbytes"  # BitsAndBytes available
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_fallback.return_value = {
            "quantization_method": "BitsAndBytes (awq_q4_0)",
            "quantized_size": "250.0 MB",
            "quantized_size_bytes": 250000000,
            "fallback_mode": True
        }
        
        # Run quantization
        result = quantize(self.test_config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(self.test_config)
        mock_validate_awq.assert_called_once()
        mock_fallback.assert_called_once_with(self.test_config)
        
        # Verify result
        assert "quantization_method" in result
        assert "fallback_mode" in result
        assert result["fallback_mode"] is True

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info") 
    @patch("oumi.quantize.main.quantize_with_bitsandbytes")
    def test_bitsandbytes_quantization(
        self,
        mock_bnb_quantize,
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
        mock_bnb_quantize.return_value = {
            "quantization_method": "BitsAndBytes 4-bit",
            "quantized_size": "250.0 MB", 
            "quantized_size_bytes": 250000000
        }
        
        # Run quantization
        result = quantize(config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(config)
        mock_bnb_quantize.assert_called_once_with(config)
        
        # Verify result
        assert result["quantization_method"] == "BitsAndBytes 4-bit"

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.main.quantize_to_gguf")
    def test_gguf_quantization(
        self,
        mock_gguf_quantize,
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
        mock_gguf_quantize.return_value = {
            "quantized_size": "250.0 MB",
            "quantized_size_bytes": 250000000
        }
        
        # Run quantization
        result = quantize(config)
        
        # Verify calls
        mock_validate_config.assert_called_once_with(config)
        mock_gguf_quantize.assert_called_once_with(config)
        
        # Verify result has expected keys
        assert "quantized_size" in result

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.main.validate_awq_requirements")
    @patch("oumi.quantize.main.simulate_awq_quantization")
    def test_quantization_error_handling(
        self, 
        mock_simulate,
        mock_validate_awq,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test error handling in quantization."""
        # Setup mocks - early calls pass but quantization fails
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_validate_awq.return_value = False  # Simulation mode
        mock_simulate.side_effect = Exception("Simulation failed")
        
        # Verify exception is raised and wrapped in RuntimeError
        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize(self.test_config)

    @patch("oumi.quantize.main.validate_quantization_config")
    @patch("oumi.quantize.main.get_model_size_info")
    @patch("oumi.quantize.main.calculate_compression_ratio")
    def test_compression_ratio_calculation(
        self,
        mock_calc_ratio,
        mock_get_size_info,
        mock_validate_config
    ):
        """Test compression ratio calculation."""
        # Setup mocks
        mock_get_size_info.return_value = ({"original_size": "1.0 GB"}, 1000000000)
        mock_calc_ratio.return_value = "4.00x"
        
        with patch("oumi.quantize.main.validate_awq_requirements", return_value=False):
            with patch("oumi.quantize.main.simulate_awq_quantization") as mock_simulate:
                mock_simulate.return_value = {
                    "quantized_size_bytes": 250000000,
                    "simulation_mode": True
                }
                
                result = quantize(self.test_config)
                
                # Verify compression ratio was calculated and added
                mock_calc_ratio.assert_called_once_with(1000000000, 250000000)
                assert result["compression_ratio"] == "4.00x"