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

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import QuantizationResult
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
from oumi.quantize.constants import BNB_METHODS, QuantizationMethod


def _make_config(method=QuantizationMethod.BNB_4BIT, **overrides):
    kwargs = dict(
        model=ModelParams(model_name="test/model"),
        method=method,
        output_path="test",
    )
    kwargs.update(overrides)
    return QuantizationConfig(**kwargs)


class TestBitsAndBytesSupportsAndValidation:

    def setup_method(self):
        self.quantizer = BitsAndBytesQuantization()

    def test_supported_methods(self):
        assert self.quantizer.supported_methods == BNB_METHODS

    def test_supported_formats(self):
        assert self.quantizer.supported_formats == ["safetensors"]

    @pytest.mark.parametrize(
        "method",
        [QuantizationMethod.BNB_4BIT, QuantizationMethod.BNB_8BIT],
    )
    def test_supports_method_valid(self, method):
        assert self.quantizer.supports_method(method) is True

    def test_supports_method_invalid(self):
        assert self.quantizer.supports_method(QuantizationMethod.FP8_DYNAMIC) is False

    @pytest.mark.parametrize(
        "method",
        [QuantizationMethod.BNB_4BIT, QuantizationMethod.BNB_8BIT],
    )
    def test_validate_config_valid(self, method):
        self.quantizer.validate_config(_make_config(method))

    def test_validate_config_wrong_method(self):
        with pytest.raises(ValueError, match="not supported by"):
            self.quantizer.validate_config(_make_config(QuantizationMethod.FP8_DYNAMIC))

    def test_validate_config_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported output format"):
            _make_config(output_format="unknown")

    def test_str_representation(self):
        assert self.quantizer.__class__.__name__ == "BitsAndBytesQuantization"


class TestBitsAndBytesRequirements:

    def test_missing_bitsandbytes(self):
        quantizer = BitsAndBytesQuantization()
        quantizer._bitsandbytes = None
        with pytest.raises(
            RuntimeError,
            match="BitsAndBytes quantization requires bitsandbytes library",
        ):
            quantizer.raise_if_requirements_not_met()


class TestGetQuantizationConfig:

    def setup_method(self):
        self.quantizer = BitsAndBytesQuantization()

    @patch("transformers.BitsAndBytesConfig")
    def test_4bit_config(self, mock_bnb_config_cls):
        self.quantizer._get_quantization_config(QuantizationMethod.BNB_4BIT)
        mock_bnb_config_cls.assert_called_once()
        call_kwargs = mock_bnb_config_cls.call_args.kwargs
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["bnb_4bit_quant_type"] == "nf4"
        assert call_kwargs["bnb_4bit_use_double_quant"] is True

    @patch("transformers.BitsAndBytesConfig")
    def test_8bit_config(self, mock_bnb_config_cls):
        self.quantizer._get_quantization_config(QuantizationMethod.BNB_8BIT)
        mock_bnb_config_cls.assert_called_once()
        call_kwargs = mock_bnb_config_cls.call_args.kwargs
        assert call_kwargs["load_in_8bit"] is True
        assert call_kwargs["llm_int8_threshold"] == 6.0

    def test_unsupported_method(self):
        with pytest.raises(ValueError, match="Unsupported BitsAndBytes method"):
            self.quantizer._get_quantization_config(QuantizationMethod.FP8_DYNAMIC)


class TestBitsAndBytesQuantize:

    def setup_method(self):
        self.quantizer = BitsAndBytesQuantization()

    @patch("oumi.quantize.bnb_quantizer.get_directory_size", return_value=2048)
    @patch("oumi.quantize.bnb_quantizer.Path")
    def test_quantize_delegates_to_quantize_model(self, _mock_path, _mock_size):
        config = _make_config(output_path="/tmp/bnb_test_output")

        with patch.object(
            self.quantizer,
            "_quantize_model",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_qm:
            self.quantizer.quantize(config)
            mock_qm.assert_called_once_with(config)

    @patch("oumi.quantize.bnb_quantizer.get_directory_size", return_value=2048)
    @patch("oumi.quantize.bnb_quantizer.Path")
    def test_quantize_saves_model_and_tokenizer(self, _mock_path, _mock_size):
        config = _make_config()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch.object(
            self.quantizer,
            "_quantize_model",
            return_value=(mock_model, mock_tokenizer),
        ):
            self.quantizer.quantize(config)

        mock_model.save_pretrained.assert_called_once()
        _, save_kwargs = mock_model.save_pretrained.call_args
        assert save_kwargs.get("safe_serialization") is True
        mock_tokenizer.save_pretrained.assert_called_once()

    @patch("oumi.quantize.bnb_quantizer.get_directory_size", return_value=4096)
    @patch("oumi.quantize.bnb_quantizer.Path")
    def test_quantize_returns_quantization_result(self, _mock_path, _mock_size):
        config = _make_config(output_path="/tmp/bnb_test_output")

        with patch.object(
            self.quantizer,
            "_quantize_model",
            return_value=(MagicMock(), MagicMock()),
        ):
            result = self.quantizer.quantize(config)

        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == QuantizationMethod.BNB_4BIT
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 4096


class TestQuantizeModelInternal:
    """Tests for _quantize_model calling from_pretrained with quantization_config."""

    def setup_method(self):
        self.quantizer = BitsAndBytesQuantization()

    @patch("oumi.quantize.bnb_quantizer.AutoTokenizer")
    @patch("oumi.quantize.bnb_quantizer.AutoModelForCausalLM")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_model_passes_quantization_config(
        self, mock_bnb_config_cls, mock_auto_model, mock_auto_tok
    ):
        config = _make_config()
        mock_bnb_cfg = MagicMock()
        mock_bnb_config_cls.return_value = mock_bnb_cfg
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_auto_tok.from_pretrained.return_value = MagicMock()

        self.quantizer._quantize_model(config)

        mock_auto_model.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert call_kwargs["quantization_config"] is mock_bnb_cfg
