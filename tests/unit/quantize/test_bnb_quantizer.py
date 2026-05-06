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

"""Unit tests for BitsAndBytes quantization backend."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError
from oumi.quantize.base import QuantizationResult
from oumi.quantize.bnb import _BNB_KWARGS, BitsAndBytesQuantization


def _make_config(
    scheme: QuantizationScheme = QuantizationScheme.BNB_NF4,
    **overrides: Any,
) -> QuantizationConfig:
    defaults: dict[str, Any] = {
        "model": ModelParams(model_name="test/model"),
        "scheme": scheme,
        "output_path": "test",
    }
    defaults.update(overrides)
    return QuantizationConfig(**defaults)


class TestBitsAndBytesSchemeMetadata:
    def test_backend_identity(self):
        assert BitsAndBytesQuantization.backend is QuantizationBackend.BNB

    def test_output_format(self):
        assert BitsAndBytesQuantization.output_format == "safetensors"

    @pytest.mark.parametrize(
        "scheme",
        [
            QuantizationScheme.BNB_NF4,
            QuantizationScheme.BNB_FP4,
            QuantizationScheme.BNB_INT8,
        ],
    )
    def test_owns(self, scheme):
        assert BitsAndBytesQuantization.owns(scheme) is True

    def test_does_not_own_llmc_scheme(self):
        assert (
            BitsAndBytesQuantization.owns(QuantizationScheme.FP8_DYNAMIC) is False
        )

    @pytest.mark.parametrize(
        "scheme",
        [
            QuantizationScheme.BNB_NF4,
            QuantizationScheme.BNB_FP4,
            QuantizationScheme.BNB_INT8,
        ],
    )
    def test_only_bnb_algorithm_allowed(self, scheme):
        spec = BitsAndBytesQuantization.schemes[scheme]
        assert spec.allowed_algorithms == (QuantizationAlgorithm.BNB,)
        assert spec.default_algorithm is QuantizationAlgorithm.BNB
        assert spec.needs_calibration_default is False

    def test_config_rejects_non_bnb_algorithm(self):
        with pytest.raises(OumiConfigError, match="not allowed"):
            _make_config(scheme=QuantizationScheme.BNB_NF4, algorithm="gptq")


class TestBnbKwargs:
    def test_nf4_kwargs(self):
        kwargs = _BNB_KWARGS[QuantizationScheme.BNB_NF4]
        assert kwargs["load_in_4bit"] is True
        assert kwargs["bnb_4bit_quant_type"] == "nf4"
        assert kwargs["bnb_4bit_use_double_quant"] is True

    def test_fp4_kwargs(self):
        kwargs = _BNB_KWARGS[QuantizationScheme.BNB_FP4]
        assert kwargs["load_in_4bit"] is True
        assert kwargs["bnb_4bit_quant_type"] == "fp4"

    def test_int8_kwargs(self):
        kwargs = _BNB_KWARGS[QuantizationScheme.BNB_INT8]
        assert kwargs["load_in_8bit"] is True
        assert kwargs["llm_int8_threshold"] == 6.0


class TestBitsAndBytesRequirements:
    def test_missing_bitsandbytes(self):
        quantizer = BitsAndBytesQuantization()
        quantizer._bnb_available = False
        with pytest.raises(
            RuntimeError,
            match="BitsAndBytes quantization requires bitsandbytes library",
        ):
            quantizer.raise_if_requirements_not_met()


class TestBitsAndBytesQuantize:
    def setup_method(self):
        self.quantizer = BitsAndBytesQuantization()

    @patch("oumi.quantize.bnb.warn_if_local_gpu_below_inference_capability")
    @patch("oumi.quantize.bnb.assert_output_path_writable")
    @patch("oumi.quantize.bnb.get_directory_size", return_value=4096)
    @patch("oumi.quantize.bnb.load_model_and_tokenizer")
    @patch("transformers.BitsAndBytesConfig")
    def test_quantize_returns_result(
        self,
        mock_bnb_cfg_cls,
        mock_load,
        _mock_size,
        _mock_assert_writable,
        _mock_warn,
        tmp_path,
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        config = _make_config(output_path=str(tmp_path / "out"))
        result = self.quantizer.quantize(config)

        assert isinstance(result, QuantizationResult)
        assert result.backend is QuantizationBackend.BNB
        assert result.scheme is QuantizationScheme.BNB_NF4
        assert result.format_type == "safetensors"
        assert result.quantized_size_bytes == 4096

        # BitsAndBytesConfig built with the per-scheme kwargs.
        bnb_kwargs = mock_bnb_cfg_cls.call_args.kwargs
        assert bnb_kwargs["load_in_4bit"] is True
        assert bnb_kwargs["bnb_4bit_quant_type"] == "nf4"
        # quantization_config passed through to the loader.
        assert "quantization_config" in mock_load.call_args.kwargs
        # Saved with safe_serialization.
        assert mock_model.save_pretrained.call_args.kwargs["safe_serialization"] is True
        mock_tokenizer.save_pretrained.assert_called_once()
