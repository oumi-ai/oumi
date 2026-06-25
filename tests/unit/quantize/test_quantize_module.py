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

"""Unit tests for main quantize module dispatch logic."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError
from oumi.quantize import quantize
from oumi.quantize.base import QuantizationResult


def _mock_quantizer(backend, scheme, size=1024, path="/test/output"):
    """Create a mock quantizer that returns a canned QuantizationResult."""
    mock = MagicMock()
    mock.raise_if_requirements_not_met.return_value = None
    mock.quantize.return_value = QuantizationResult(
        quantized_size_bytes=size,
        output_path=path,
        backend=backend,
        scheme=scheme,
        format_type="safetensors",
    )
    return mock


def _make_config(scheme):
    return QuantizationConfig(
        model=ModelParams(model_name="test/model"),
        scheme=scheme,
        output_path="test_model",
    )


class TestQuantizeDispatch:
    @patch("oumi.builders.quantizers.build_quantizer")
    def test_fp8_dynamic_dispatches(self, mock_build):
        mock_build.return_value = _mock_quantizer(
            QuantizationBackend.LLM_COMPRESSOR, QuantizationScheme.FP8_DYNAMIC
        )
        result = quantize(_make_config(QuantizationScheme.FP8_DYNAMIC))

        assert result.backend == QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme == QuantizationScheme.FP8_DYNAMIC
        mock_build.assert_called_once_with(QuantizationBackend.LLM_COMPRESSOR)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_w4a16_dispatches(self, mock_build):
        mock_build.return_value = _mock_quantizer(
            QuantizationBackend.LLM_COMPRESSOR, QuantizationScheme.W4A16
        )
        result = quantize(_make_config(QuantizationScheme.W4A16))

        assert result.backend == QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme == QuantizationScheme.W4A16
        mock_build.assert_called_once_with(QuantizationBackend.LLM_COMPRESSOR)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_bnb_nf4_dispatches(self, mock_build):
        mock_build.return_value = _mock_quantizer(
            QuantizationBackend.BNB, QuantizationScheme.BNB_NF4
        )
        result = quantize(_make_config(QuantizationScheme.BNB_NF4))

        assert result.backend == QuantizationBackend.BNB
        assert result.scheme == QuantizationScheme.BNB_NF4
        mock_build.assert_called_once_with(QuantizationBackend.BNB)


class TestQuantizeErrorHandling:
    def test_invalid_config_type(self):
        with pytest.raises(ValueError, match="Expected QuantizationConfig"):
            quantize("not a config")  # type: ignore

    def test_unsupported_scheme_rejected_at_config_creation(self):
        with pytest.raises(OumiConfigError, match="Unsupported scheme"):
            _make_config("invalid_scheme")

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_requirements_not_met(self, mock_build):
        mock_q = MagicMock()
        mock_q.raise_if_requirements_not_met.side_effect = RuntimeError("No GPU")
        mock_build.return_value = mock_q

        with pytest.raises(RuntimeError, match="No GPU"):
            quantize(_make_config(QuantizationScheme.FP8_DYNAMIC))

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantizer_failure(self, mock_build):
        mock_q = MagicMock()
        mock_q.raise_if_requirements_not_met.return_value = None
        mock_q.quantize.side_effect = RuntimeError("Quantization failed")
        mock_build.return_value = mock_q

        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize(_make_config(QuantizationScheme.FP8_DYNAMIC))

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_quantize_calls_raise_if_requirements_not_met(self, mock_build):
        mock_q = _mock_quantizer(
            QuantizationBackend.LLM_COMPRESSOR, QuantizationScheme.FP8_DYNAMIC
        )
        mock_build.return_value = mock_q

        quantize(_make_config(QuantizationScheme.FP8_DYNAMIC))

        mock_q.raise_if_requirements_not_met.assert_called_once()
        mock_q.quantize.assert_called_once()
