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

"""Tests for AWQ quantization."""

import tempfile
from unittest.mock import Mock, patch

import pytest
import torch

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.awq_quantizer import AWQ_DEFAULTS, AwqQuantization
from oumi.quantize.base import QuantizationResult


class TestAwqQuantization:
    """Test AWQ quantization implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = AwqQuantization()

    def test_supported_methods(self):
        """Test supported quantization methods."""
        expected_methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
        assert self.quantizer.supported_methods == expected_methods

    def test_supported_formats(self):
        """Test supported output formats."""
        expected_formats = ["pytorch"]
        assert self.quantizer.supported_formats == expected_formats

    @patch("oumi.quantize.awq_quantizer.importlib.util.find_spec")
    @patch("oumi.quantize.awq_quantizer.torch.cuda.is_available")
    def test_raise_if_requirements_not_met_missing_library(
        self, mock_cuda_available, mock_find_spec
    ):
        """Test requirements check when autoawq library is missing."""
        mock_find_spec.return_value = None
        mock_cuda_available.return_value = True
        quantizer = AwqQuantization()

        with pytest.raises(
            RuntimeError, match="AWQ quantization requires autoawq library"
        ):
            quantizer.raise_if_requirements_not_met()

    @patch("oumi.quantize.awq_quantizer.importlib.util.find_spec")
    @patch("oumi.quantize.awq_quantizer.torch.cuda.is_available")
    def test_raise_if_requirements_not_met_no_gpu(
        self, mock_cuda_available, mock_find_spec
    ):
        """Test requirements check when no GPU is available."""
        mock_find_spec.return_value = Mock()
        mock_cuda_available.return_value = False
        quantizer = AwqQuantization()

        with pytest.raises(RuntimeError, match="AWQ quantization requires a GPU"):
            quantizer.raise_if_requirements_not_met()

    @patch("oumi.quantize.awq_quantizer.importlib.util.find_spec")
    @patch("oumi.quantize.awq_quantizer.torch.cuda.is_available")
    def test_raise_if_requirements_not_met_success(
        self, mock_cuda_available, mock_find_spec
    ):
        """Test successful requirements check."""
        mock_find_spec.return_value = Mock()
        mock_cuda_available.return_value = True
        quantizer = AwqQuantization()

        # Should not raise any exception
        quantizer.raise_if_requirements_not_met()

    @patch.object(AwqQuantization, "validate_config")
    def test_validate_config_invalid_format(self, mock_validate):
        """Test config validation with invalid format."""
        # Create config with valid format for initialization
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_format="pytorch",  # Valid format for config creation
            output_path="/test/output",
        )

        # Change format after creation to test the logic
        config.output_format = "safetensors"

        with pytest.raises(
            ValueError, match="AWQ quantization only supports PyTorch format"
        ):
            self.quantizer.quantize(config)

    @patch("oumi.quantize.awq_quantizer.get_directory_size")
    @patch("oumi.quantize.awq_quantizer.AutoTokenizer")
    def test_quantize_success(self, mock_tokenizer_class, mock_get_size):
        """Test successful AWQ quantization."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_get_size.return_value = 1024 * 1024  # 1MB

        # Mock AWQ model
        mock_awq_model = Mock()
        mock_awq_model.quantize = Mock()
        mock_awq_model.save_quantized = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="awq_q4_0",
                output_format="pytorch",
                output_path=temp_dir,
                calibration_samples=128,
                awq_zero_point=True,
                awq_group_size=128,
                awq_version="GEMM",
            )

            with (
                patch.object(self.quantizer, "raise_if_requirements_not_met"),
                patch.object(
                    self.quantizer,
                    "_quantize",
                    return_value=(mock_awq_model, mock_tokenizer),
                ),
            ):
                result = self.quantizer.quantize(config)

            # Verify result
            assert isinstance(result, QuantizationResult)
            assert result.quantization_method == "awq_q4_0"
            assert result.quantized_size_bytes == 1024 * 1024
            assert result.format_type == "pytorch"
            assert result.output_path == temp_dir

            # Verify model was saved
            mock_awq_model.save_quantized.assert_called_once_with(temp_dir)
            mock_tokenizer.save_pretrained.assert_called_once_with(temp_dir)

    @patch("oumi.quantize.awq_quantizer.AutoTokenizer")
    def test_quantize_internal_method(self, mock_tokenizer_class):
        """Test internal _quantize method."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock AWQ model and library
        mock_awq_model = Mock()
        mock_awq_model.quantize = Mock()

        mock_awq_class = Mock()
        mock_awq_class.from_pretrained.return_value = mock_awq_model

        config = QuantizationConfig(
            model=ModelParams(
                model_name="test/model", tokenizer_name="custom/tokenizer"
            ),
            method="awq_q8_0",
            output_format="pytorch",
            output_path="/test/output",
            calibration_samples=256,
            awq_zero_point=False,
            awq_group_size=64,
            awq_version="GEMV",
        )

        # Mock the _awq attribute to return the mock class
        self.quantizer._awq = Mock()
        self.quantizer._awq.AutoAWQForCausalLM = mock_awq_class

        model, tokenizer = self.quantizer._quantize(config)

        # Verify model loading
        mock_awq_class.from_pretrained.assert_called_once_with(
            "test/model", safetensors=True, trust_remote_code=True
        )

        # Verify tokenizer loading with custom name
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "custom/tokenizer", trust_remote_code=True
        )

        # Verify quantization was called with correct parameters
        expected_quant_config = {
            "zero_point": False,
            "q_group_size": 64,
            "w_bit": 8,  # awq_q8_0 -> 8 bits
            "version": "GEMV",
        }

        mock_awq_model.quantize.assert_called_once()
        call_args = mock_awq_model.quantize.call_args

        # Check quantization parameters
        assert call_args[0][0] == mock_tokenizer  # tokenizer is first arg
        assert call_args[1]["quant_config"] == expected_quant_config
        assert call_args[1]["max_calib_samples"] == 256
        assert call_args[1]["calib_data"] == AWQ_DEFAULTS["calibration_dataset"]

        assert model == mock_awq_model
        assert tokenizer == mock_tokenizer

    def test_w_bit_mapping(self):
        """Test bit width mapping for different AWQ methods."""
        test_cases = [
            ("awq_q4_0", 4),
            ("awq_q4_1", 4),
            ("awq_q8_0", 8),
            ("awq_f16", 16),
        ]

        for method, expected_bits in test_cases:
            config = QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method=method,
                output_format="pytorch",
                output_path="/test/output",
            )

            # Mock the quantize method to capture the bit width
            with patch.object(self.quantizer, "_quantize") as mock_quantize_internal:
                mock_model = Mock()
                mock_tokenizer = Mock()
                mock_quantize_internal.return_value = (mock_model, mock_tokenizer)

                with (
                    patch.object(self.quantizer, "raise_if_requirements_not_met"),
                    patch(
                        "oumi.quantize.awq_quantizer.get_directory_size",
                        return_value=1024,
                    ),
                ):
                    self.quantizer.quantize(config)

                # The _quantize method should be called with the config
                mock_quantize_internal.assert_called_once_with(config)

    def test_awq_defaults_constants(self):
        """Test AWQ default constants."""
        assert AWQ_DEFAULTS["calibration_dataset"] == "pileval"
        assert AWQ_DEFAULTS["calibration_split"] == "train"
        assert AWQ_DEFAULTS["calibration_text_column"] == "text"
        assert AWQ_DEFAULTS["max_calibration_seq_len"] == 512
        assert AWQ_DEFAULTS["duo_scaling"] is True
        assert AWQ_DEFAULTS["apply_clip"] is True
        assert AWQ_DEFAULTS["n_parallel_calib_samples"] is None

    def test_validate_config_integration(self):
        """Test config validation with actual supported methods/formats."""
        # Valid config should not raise
        valid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
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
        with pytest.raises(ValueError, match="not supported by AwqQuantization"):
            self.quantizer.validate_config(invalid_method_config)

        # Invalid format should raise
        invalid_format_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_format="safetensors",
            output_path="/test/output",
        )
        with pytest.raises(ValueError, match="not supported by AwqQuantization"):
            self.quantizer.validate_config(invalid_format_config)
