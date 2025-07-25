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

"""Unit tests for BitsAndBytes quantization."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import QuantizationResult
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization


class TestBitsAndBytesQuantization:
    """Test cases for BitsAndBytesQuantization class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = BitsAndBytesQuantization()
        self.temp_dir = tempfile.mkdtemp()

        self.valid_4bit_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="bnb_4bit",
            output_path=os.path.join(self.temp_dir, "model_4bit"),
            output_format="safetensors",
        )

        self.valid_8bit_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="bnb_8bit",
            output_path=os.path.join(self.temp_dir, "model_8bit"),
            output_format="safetensors",
        )

    def test_supported_methods(self):
        """Test that correct methods are supported."""
        expected_methods = ["bnb_4bit", "bnb_8bit"]
        assert self.quantizer.supported_methods == expected_methods

    def test_supported_formats(self):
        """Test that correct formats are supported."""
        expected_formats = ["pytorch", "safetensors"]
        assert self.quantizer.supported_formats == expected_formats

    def test_supports_method_4bit(self):
        """Test that bnb_4bit method is supported."""
        assert self.quantizer.supports_method("bnb_4bit") is True

    def test_supports_method_8bit(self):
        """Test that bnb_8bit method is supported."""
        assert self.quantizer.supports_method("bnb_8bit") is True

    def test_supports_method_invalid(self):
        """Test that invalid methods are not supported."""
        assert self.quantizer.supports_method("bnb_16bit") is False
        assert self.quantizer.supports_method("awq_4bit") is False

    def test_validate_config_valid_4bit(self):
        """Test config validation with valid 4bit config."""
        # Should not raise any exception
        self.quantizer.validate_config(self.valid_4bit_config)

    def test_validate_config_valid_8bit(self):
        """Test config validation with valid 8bit config."""
        # Should not raise any exception
        self.quantizer.validate_config(self.valid_8bit_config)

    def test_validate_config_invalid_method(self):
        """Test config validation with invalid method."""
        invalid_config = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m"),
            method="invalid_method",
            output_path=os.path.join(self.temp_dir, "model"),
            output_format="safetensors",
        )

        with pytest.raises(ValueError, match="Unsupported quantization method"):
            self.quantizer.validate_config(invalid_config)

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_4bit_success(self, mock_bnb_config, mock_model, mock_tokenizer):
        """Test successful 4bit quantization."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.get_memory_footprint.return_value = 500000000  # 500MB
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_bnb_config_instance = Mock()
        mock_bnb_config.return_value = mock_bnb_config_instance

        # Run quantization
        result = self.quantizer.quantize(self.valid_4bit_config)

        # Verify result
        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == "bnb_4bit"
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 500000000
        assert result.output_path == self.valid_4bit_config.output_path

        # Verify BitsAndBytesConfig was created with correct parameters
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="auto",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Verify model was loaded with quantization config
        mock_model.from_pretrained.assert_called_once_with(
            "facebook/opt-125m",
            quantization_config=mock_bnb_config_instance,
            device_map="auto",
            trust_remote_code=False,
        )

        # Verify model was saved
        mock_model_instance.save_pretrained.assert_called_once_with(
            self.valid_4bit_config.output_path, safe_serialization=True
        )

        # Verify tokenizer was saved
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(
            self.valid_4bit_config.output_path
        )

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_8bit_success(self, mock_bnb_config, mock_model, mock_tokenizer):
        """Test successful 8bit quantization."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.get_memory_footprint.return_value = 250000000  # 250MB
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_bnb_config_instance = Mock()
        mock_bnb_config.return_value = mock_bnb_config_instance

        # Run quantization
        result = self.quantizer.quantize(self.valid_8bit_config)

        # Verify result
        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == "bnb_8bit"
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 250000000
        assert result.output_path == self.valid_8bit_config.output_path

        # Verify BitsAndBytesConfig was created with correct parameters
        mock_bnb_config.assert_called_once_with(load_in_8bit=True)

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_with_trust_remote_code(
        self, mock_bnb_config, mock_model, mock_tokenizer
    ):
        """Test quantization with trust_remote_code=True."""
        # Modify config to include trust_remote_code
        config_with_trust = QuantizationConfig(
            model=ModelParams(model_name="facebook/opt-125m", trust_remote_code=True),
            method="bnb_4bit",
            output_path=os.path.join(self.temp_dir, "model_trust"),
            output_format="safetensors",
        )

        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.get_memory_footprint.return_value = 500000000
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_bnb_config_instance = Mock()
        mock_bnb_config.return_value = mock_bnb_config_instance

        # Run quantization
        self.quantizer.quantize(config_with_trust)

        # Verify model was loaded with trust_remote_code=True
        mock_model.from_pretrained.assert_called_once_with(
            "facebook/opt-125m",
            quantization_config=mock_bnb_config_instance,
            device_map="auto",
            trust_remote_code=True,
        )

        # Verify tokenizer was loaded with trust_remote_code=True
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "facebook/opt-125m", trust_remote_code=True
        )

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_model_loading_failure(
        self, mock_bnb_config, mock_model, mock_tokenizer
    ):
        """Test quantization failure during model loading."""
        # Setup mocks to raise exception
        mock_model.from_pretrained.side_effect = Exception("Model loading failed")

        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Failed to quantize model"):
            self.quantizer.quantize(self.valid_4bit_config)

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_tokenizer_loading_failure(
        self, mock_bnb_config, mock_model, mock_tokenizer
    ):
        """Test quantization failure during tokenizer loading."""
        # Setup mocks
        mock_tokenizer.from_pretrained.side_effect = Exception(
            "Tokenizer loading failed"
        )

        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Failed to quantize model"):
            self.quantizer.quantize(self.valid_4bit_config)

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_saving_failure(self, mock_bnb_config, mock_model, mock_tokenizer):
        """Test quantization failure during model saving."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.get_memory_footprint.return_value = 500000000
        mock_model_instance.save_pretrained.side_effect = Exception("Save failed")
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_bnb_config_instance = Mock()
        mock_bnb_config.return_value = mock_bnb_config_instance

        # Run quantization and expect failure
        with pytest.raises(RuntimeError, match="Failed to quantize model"):
            self.quantizer.quantize(self.valid_4bit_config)

    def test_str_representation(self):
        """Test string representation of the quantizer."""
        str_repr = str(self.quantizer)
        assert "BitsAndBytesQuantization" in str_repr
        assert "bnb_4bit" in str_repr
        assert "bnb_8bit" in str_repr

