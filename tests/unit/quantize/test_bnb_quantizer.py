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

"""Tests for BitsAndBytes quantization."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import QuantizationResult
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization


class TestBitsAndBytesQuantization:
    """Test BitsAndBytes quantization implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = BitsAndBytesQuantization()

    def test_supported_methods(self):
        """Test supported quantization methods."""
        expected_methods = ["bnb_4bit", "bnb_8bit"]
        assert self.quantizer.supported_methods == expected_methods

    def test_supported_formats(self):
        """Test supported output formats."""
        expected_formats = ["pytorch", "safetensors"]
        assert self.quantizer.supported_formats == expected_formats

    @patch("oumi.quantize.bnb_quantizer.importlib.util.find_spec")
    def test_raise_if_requirements_not_met_missing_library(self, mock_find_spec):
        """Test requirements check when bitsandbytes library is missing."""
        mock_find_spec.return_value = None
        quantizer = BitsAndBytesQuantization()

        with pytest.raises(RuntimeError, match="BitsAndBytes requires bitsandbytes"):
            quantizer.raise_if_requirements_not_met()

    @patch("oumi.quantize.bnb_quantizer.importlib.util.find_spec")
    def test_raise_if_requirements_not_met_library_present(self, mock_find_spec):
        """Test requirements check when bitsandbytes library is present."""
        mock_find_spec.return_value = Mock()
        quantizer = BitsAndBytesQuantization()

        with patch("builtins.__import__") as mock_import:
            mock_bitsandbytes = Mock()
            mock_bitsandbytes.__version__ = "0.45.0"
            mock_import.return_value = mock_bitsandbytes

            # Should not raise any exception
            quantizer.raise_if_requirements_not_met()

    @patch("oumi.quantize.bnb_quantizer.BitsAndBytesConfig")
    def test_get_quantization_config_4bit(self, mock_bnb_config):
        """Test 4-bit quantization config generation."""
        mock_config_instance = Mock()
        mock_bnb_config.return_value = mock_config_instance

        config = self.quantizer._get_quantization_config("bnb_4bit")

        # Verify BitsAndBytesConfig was called with correct parameters
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        assert config == mock_config_instance

    @patch("oumi.quantize.bnb_quantizer.BitsAndBytesConfig")
    def test_get_quantization_config_8bit(self, mock_bnb_config):
        """Test 8-bit quantization config generation."""
        mock_config_instance = Mock()
        mock_bnb_config.return_value = mock_config_instance

        config = self.quantizer._get_quantization_config("bnb_8bit")

        # Verify BitsAndBytesConfig was called with correct parameters
        mock_bnb_config.assert_called_once_with(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        assert config == mock_config_instance

    def test_get_quantization_config_invalid(self):
        """Test invalid quantization method."""
        with pytest.raises(ValueError, match="Unsupported BitsAndBytes method"):
            self.quantizer._get_quantization_config("invalid_method")

    @patch("oumi.quantize.bnb_quantizer.get_directory_size")
    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    def test_quantize_4bit_pytorch_format(
        self, mock_model_class, mock_tokenizer_class, mock_get_size
    ):
        """Test 4-bit quantization with PyTorch format."""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_get_size.return_value = 1024 * 1024  # 1MB

        with tempfile.TemporaryDirectory() as temp_dir:
            config = QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="bnb_4bit",
                output_format="pytorch",
                output_path=temp_dir,
            )

            with patch.object(self.quantizer, "raise_if_requirements_not_met"):
                result = self.quantizer.quantize(config)

            # Verify result
            assert isinstance(result, QuantizationResult)
            assert result.quantization_method == "bnb_4bit"
            assert result.quantized_size_bytes == 1024 * 1024
            assert result.format_type == "pytorch"
            assert temp_dir in result.output_path

            # Verify model was saved without safe serialization (PyTorch format)
            mock_model.save_pretrained.assert_called_once()
            call_args = mock_model.save_pretrained.call_args
            assert call_args[1]["safe_serialization"] is False

    @patch("oumi.quantize.bnb_quantizer.get_directory_size")
    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    def test_quantize_8bit_safetensors_format(
        self, mock_model_class, mock_tokenizer_class, mock_get_size
    ):
        """Test 8-bit quantization with safetensors format."""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_get_size.return_value = 2048 * 1024  # 2MB

        with tempfile.TemporaryDirectory() as temp_dir:
            config = QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="bnb_8bit",
                output_format="safetensors",
                output_path=temp_dir,
            )

            with patch.object(self.quantizer, "raise_if_requirements_not_met"):
                result = self.quantizer.quantize(config)

            # Verify result
            assert isinstance(result, QuantizationResult)
            assert result.quantization_method == "bnb_8bit"
            assert result.quantized_size_bytes == 2048 * 1024
            assert result.format_type == "safetensors"

            # Verify model was saved with safe serialization (safetensors format)
            mock_model.save_pretrained.assert_called_once()
            call_args = mock_model.save_pretrained.call_args
            assert call_args[1]["safe_serialization"] is True

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    def test_quantize_with_custom_tokenizer(
        self, mock_model_class, mock_tokenizer_class
    ):
        """Test quantization with custom tokenizer name."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as temp_dir:
            config = QuantizationConfig(
                model=ModelParams(
                    model_name="test/model", tokenizer_name="custom/tokenizer"
                ),
                method="bnb_4bit",
                output_format="pytorch",
                output_path=temp_dir,
            )

            with patch.object(self.quantizer, "raise_if_requirements_not_met"):
                with patch(
                    "oumi.quantize.bnb_quantizer.get_directory_size", return_value=1024
                ):
                    self.quantizer.quantize(config)

            # Verify tokenizer was loaded with custom name
            mock_tokenizer_class.from_pretrained.assert_called_once_with(
                "custom/tokenizer", trust_remote_code=True
            )

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    def test_quantize_with_model_kwargs(self, mock_model_class, mock_tokenizer_class):
        """Test quantization with additional model kwargs."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model_kwargs = {"attn_implementation": "flash_attention_2"}

        with tempfile.TemporaryDirectory() as temp_dir:
            config = QuantizationConfig(
                model=ModelParams(model_name="test/model", model_kwargs=model_kwargs),
                method="bnb_4bit",
                output_format="pytorch",
                output_path=temp_dir,
            )

            with patch.object(self.quantizer, "raise_if_requirements_not_met"):
                with patch(
                    "oumi.quantize.bnb_quantizer.get_directory_size", return_value=1024
                ):
                    self.quantizer.quantize(config)

            # Verify model was loaded with additional kwargs
            call_args = mock_model_class.from_pretrained.call_args[1]
            assert "attn_implementation" in call_args
            assert call_args["attn_implementation"] == "flash_attention_2"

    def test_save_model_with_file_extension(self):
        """Test _save_model with output path that has file extension."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Path with .bin extension
            output_path_with_ext = str(Path(temp_dir) / "model.bin")
            config = QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="bnb_4bit",
                output_format="pytorch",
                output_path=output_path_with_ext,
            )

            result_path = self.quantizer._save_model(mock_model, mock_tokenizer, config)

            # Should save to parent directory, not to the file itself
            expected_dir = str(Path(temp_dir))
            assert result_path == expected_dir

            # Verify directory was created
            assert Path(temp_dir).exists()

    def test_validate_config_integration(self):
        """Test config validation with actual supported methods/formats."""
        # Valid config should not raise
        valid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_format="pytorch",
            output_path="/test/output",
        )
        self.quantizer.validate_config(valid_config)

        # Invalid method should raise
        invalid_method_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="invalid_method",
            output_format="pytorch",
            output_path="/test/output",
        )
        with pytest.raises(
            ValueError, match="not supported by BitsAndBytesQuantization"
        ):
            self.quantizer.validate_config(invalid_method_config)

        # Invalid format should raise
        invalid_format_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_format="invalid_format",
            output_path="/test/output",
        )
        with pytest.raises(
            ValueError, match="not supported by BitsAndBytesQuantization"
        ):
            self.quantizer.validate_config(invalid_format_config)
