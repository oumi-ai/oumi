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

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from oumi.core.configs import QuantizationConfig, ModelParams, GenerationParams
from oumi.quantize.awq_quantizer import AwqQuantization
from oumi.quantize.base import QuantizationResult


class TestAwqQuantization:
    """Test cases for AWQ quantization class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = AwqQuantization()
        self.temp_dir = tempfile.mkdtemp()
        
        self.valid_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="awq_4bit",
            output_path=os.path.join(self.temp_dir, "model_awq"),
            output_format="safetensors"
        )
    
    def test_supported_methods(self):
        """Test that correct methods are supported."""
        expected_methods = [
            "awq_4bit", "awq_q4_0", "awq_q4_1", "awq_q8_0",
            "awq_q5_0", "awq_q5_1", "awq_q2_k", "awq_q3_k_s",
            "awq_q3_k_m", "awq_q3_k_l", "awq_q4_k_s", "awq_q4_k_m",
            "awq_q5_k_s", "awq_q5_k_m", "awq_q6_k"
        ]
        assert self.quantizer.supported_methods == expected_methods
    
    def test_supported_formats(self):
        """Test that correct formats are supported."""
        expected_formats = ["safetensors", "pytorch"]
        assert self.quantizer.supported_formats == expected_formats
    
    def test_supports_method_valid(self):
        """Test that valid AWQ methods are supported."""
        assert self.quantizer.supports_method("awq_4bit") is True
        assert self.quantizer.supports_method("awq_q4_0") is True
        assert self.quantizer.supports_method("awq_q8_0") is True
    
    def test_supports_method_invalid(self):
        """Test that invalid methods are not supported."""
        assert self.quantizer.supports_method("bnb_4bit") is False
        assert self.quantizer.supports_method("awq_invalid") is False
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        # Should not raise any exception
        self.quantizer.validate_config(self.valid_config)
    
    def test_validate_config_invalid_method(self):
        """Test config validation with invalid method."""
        invalid_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="invalid_method",
            output_path=os.path.join(self.temp_dir, "model"),
            output_format="safetensors"
        )
        
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            self.quantizer.validate_config(invalid_config)
    
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_success(self, mock_awq_model):
        """Test successful AWQ quantization."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.get_model_size.return_value = 125000000  # 125MB
        mock_awq_model.from_pretrained.return_value = mock_model_instance
        
        # Run quantization
        result = self.quantizer.quantize(self.valid_config)
        
        # Verify result
        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == "awq_4bit"
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 125000000
        assert result.output_path == self.valid_config.output_path
        
        # Verify model was loaded correctly
        mock_awq_model.from_pretrained.assert_called_once_with(
            "facebook/opt-125m",
            device_map="auto",
            trust_remote_code=False,
            safetensors=True
        )
        
        # Verify model was quantized
        mock_model_instance.quantize.assert_called_once_with(
            tokenizer=None,
            quant_config={
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM"
            }
        )
        
        # Verify model was saved
        mock_model_instance.save_quantized.assert_called_once_with(
            self.valid_config.output_path,
            safetensors=True
        )
    
    @patch('oumi.quantize.awq_quantizer.AutoTokenizer')
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_with_tokenizer(self, mock_awq_model, mock_tokenizer):
        """Test AWQ quantization with tokenizer loading."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.get_model_size.return_value = 125000000
        mock_awq_model.from_pretrained.return_value = mock_model_instance
        
        # Run quantization
        result = self.quantizer.quantize(self.valid_config)
        
        # Verify tokenizer was loaded
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "facebook/opt-125m",
            trust_remote_code=False
        )
        
        # Verify model was quantized with tokenizer
        mock_model_instance.quantize.assert_called_once_with(
            tokenizer=mock_tokenizer_instance,
            quant_config={
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM"
            }
        )
        
        # Verify tokenizer was saved
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(
            self.valid_config.output_path
        )
    
    @patch('oumi.quantize.awq_quantizer.AutoTokenizer')
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_with_trust_remote_code(self, mock_awq_model, mock_tokenizer):
        """Test quantization with trust_remote_code=True."""
        # Modify config to include trust_remote_code
        config_with_trust = QuantizationConfig(
            model=ModelParams(
                model_name="facebook/opt-125m",
                trust_remote_code=True
            ),
            method="awq_4bit",
            output_path=os.path.join(self.temp_dir, "model_trust"),
            output_format="safetensors"
        )
        
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.get_model_size.return_value = 125000000
        mock_awq_model.from_pretrained.return_value = mock_model_instance
        
        # Run quantization
        result = self.quantizer.quantize(config_with_trust)
        
        # Verify model was loaded with trust_remote_code=True
        mock_awq_model.from_pretrained.assert_called_once_with(
            "facebook/opt-125m",
            device_map="auto",
            trust_remote_code=True,
            safetensors=True
        )
        
        # Verify tokenizer was loaded with trust_remote_code=True
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "facebook/opt-125m",
            trust_remote_code=True
        )
    
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_different_methods(self, mock_awq_model):
        """Test quantization with different AWQ methods."""
        test_cases = [
            ("awq_q4_0", {"zero_point": False, "q_group_size": 32, "w_bit": 4, "version": "GEMM"}),
            ("awq_q8_0", {"zero_point": False, "q_group_size": 32, "w_bit": 8, "version": "GEMM"}),
            ("awq_q5_0", {"zero_point": False, "q_group_size": 32, "w_bit": 5, "version": "GEMM"}),
        ]
        
        for method, expected_config in test_cases:
            # Setup mock
            mock_model_instance = Mock()
            mock_model_instance.get_model_size.return_value = 125000000
            mock_awq_model.from_pretrained.return_value = mock_model_instance
            
            # Create config for this method
            config = QuantizationConfig(
                model=ModelParams(model_name="facebook/opt-125m"),
                method=method,
                output_path=os.path.join(self.temp_dir, f"model_{method}"),
                output_format="safetensors"
            )
            
            # Run quantization
            result = self.quantizer.quantize(config)
            
            # Verify result
            assert result.quantization_method == method
            
            # Verify correct quantization config was used
            mock_model_instance.quantize.assert_called_with(
                tokenizer=None,
                quant_config=expected_config
            )
            
            # Reset mock for next iteration
            mock_awq_model.reset_mock()
    
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_model_loading_failure(self, mock_awq_model):
        """Test quantization failure during model loading."""
        # Setup mock to raise exception
        mock_awq_model.from_pretrained.side_effect = Exception("Model loading failed")
        
        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Failed to quantize model"):
            self.quantizer.quantize(self.valid_config)
    
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_quantization_failure(self, mock_awq_model):
        """Test quantization failure during quantization step."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.quantize.side_effect = Exception("Quantization failed")
        mock_awq_model.from_pretrained.return_value = mock_model_instance
        
        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Failed to quantize model"):
            self.quantizer.quantize(self.valid_config)
    
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_saving_failure(self, mock_awq_model):
        """Test quantization failure during model saving."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.get_model_size.return_value = 125000000
        mock_model_instance.save_quantized.side_effect = Exception("Save failed")
        mock_awq_model.from_pretrained.return_value = mock_model_instance
        
        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Failed to quantize model"):
            self.quantizer.quantize(self.valid_config)
    
    @patch('oumi.quantize.awq_quantizer.AutoTokenizer')
    @patch('oumi.quantize.awq_quantizer.AWQModel')
    def test_quantize_tokenizer_failure(self, mock_awq_model, mock_tokenizer):
        """Test quantization continues even with tokenizer loading failure."""
        # Setup mocks
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer failed")
        
        mock_model_instance = Mock()
        mock_model_instance.get_model_size.return_value = 125000000
        mock_awq_model.from_pretrained.return_value = mock_model_instance
        
        # Run quantization - should succeed without tokenizer
        result = self.quantizer.quantize(self.valid_config)
        
        # Verify quantization succeeded
        assert isinstance(result, QuantizationResult)
        
        # Verify model was quantized without tokenizer
        mock_model_instance.quantize.assert_called_once_with(
            tokenizer=None,
            quant_config={
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM"
            }
        )
    
    def test_get_quant_config_awq_4bit(self):
        """Test getting quantization config for awq_4bit."""
        config = self.quantizer._get_quant_config("awq_4bit")
        expected = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }
        assert config == expected
    
    def test_get_quant_config_awq_q4_k_s(self):
        """Test getting quantization config for awq_q4_k_s."""
        config = self.quantizer._get_quant_config("awq_q4_k_s")
        expected = {
            "zero_point": True,
            "q_group_size": 32,
            "w_bit": 4,
            "version": "GEMV_FAST"
        }
        assert config == expected
    
    def test_get_quant_config_unknown_method(self):
        """Test getting quantization config for unknown method."""
        with pytest.raises(ValueError, match="Unsupported AWQ quantization method"):
            self.quantizer._get_quant_config("unknown_method")
    
    def test_str_representation(self):
        """Test string representation of the quantizer."""
        str_repr = str(self.quantizer)
        assert "AwqQuantization" in str_repr
        assert "awq_4bit" in str_repr