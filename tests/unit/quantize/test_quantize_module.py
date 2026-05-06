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


def _make_config(scheme):
    """Build a real QuantizationConfig (no patch active here)."""
    return QuantizationConfig(
        model=ModelParams(model_name="test/model"),
        scheme=scheme,
        output_path="test_model",
    )


def _mock_backend_class(backend, scheme, size=1024, path="/test/output"):
    """Mock callable (class) whose instantiation yields a stubbed quantizer."""
    instance = MagicMock()
    instance.raise_if_requirements_not_met.return_value = None
    instance.quantize.return_value = QuantizationResult(
        output_path=path,
        backend=backend,
        scheme=scheme,
        format_type="safetensors",
        quantized_size_bytes=size,
    )
    cls = MagicMock(return_value=instance)
    cls._instance = instance  # for assertions
    return cls


class TestQuantizeDispatch:
    def test_fp8_dynamic_dispatches(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        cls = _mock_backend_class(
            QuantizationBackend.LLM_COMPRESSOR, QuantizationScheme.FP8_DYNAMIC
        )
        with patch("oumi.quantize.backend_for_scheme", return_value=cls) as mock:
            result = quantize(config)
        assert result.backend is QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme is QuantizationScheme.FP8_DYNAMIC
        mock.assert_called_with(QuantizationScheme.FP8_DYNAMIC)

    def test_w4a16_dispatches(self):
        config = _make_config(QuantizationScheme.W4A16)
        cls = _mock_backend_class(
            QuantizationBackend.LLM_COMPRESSOR, QuantizationScheme.W4A16
        )
        with patch("oumi.quantize.backend_for_scheme", return_value=cls) as mock:
            result = quantize(config)
        assert result.backend is QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme is QuantizationScheme.W4A16
        mock.assert_called_with(QuantizationScheme.W4A16)

    def test_bnb_nf4_dispatches(self):
        config = _make_config(QuantizationScheme.BNB_NF4)
        cls = _mock_backend_class(
            QuantizationBackend.BNB, QuantizationScheme.BNB_NF4
        )
        with patch("oumi.quantize.backend_for_scheme", return_value=cls) as mock:
            result = quantize(config)
        assert result.backend is QuantizationBackend.BNB
        assert result.scheme is QuantizationScheme.BNB_NF4
        mock.assert_called_with(QuantizationScheme.BNB_NF4)


class TestQuantizeErrorHandling:
    def test_invalid_config_type(self):
        with pytest.raises(ValueError, match="Expected QuantizationConfig"):
            quantize("not a config")  # type: ignore

    def test_unsupported_scheme_rejected_at_config_creation(self):
        with pytest.raises(OumiConfigError, match="Unsupported scheme"):
            _make_config("invalid_scheme")

    def test_requirements_not_met(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        instance = MagicMock()
        instance.raise_if_requirements_not_met.side_effect = RuntimeError("No GPU")
        cls = MagicMock(return_value=instance)
        with patch("oumi.quantize.backend_for_scheme", return_value=cls):
            with pytest.raises(RuntimeError, match="No GPU"):
                quantize(config)

    def test_quantizer_failure(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        instance = MagicMock()
        instance.raise_if_requirements_not_met.return_value = None
        instance.quantize.side_effect = RuntimeError("Quantization failed")
        cls = MagicMock(return_value=instance)
        with patch("oumi.quantize.backend_for_scheme", return_value=cls):
            with pytest.raises(RuntimeError, match="Quantization failed"):
                quantize(config)

    def test_quantize_calls_raise_if_requirements_not_met(self):
        config = _make_config(QuantizationScheme.FP8_DYNAMIC)
        cls = _mock_backend_class(
            QuantizationBackend.LLM_COMPRESSOR, QuantizationScheme.FP8_DYNAMIC
        )
        with patch("oumi.quantize.backend_for_scheme", return_value=cls):
            quantize(config)
        cls._instance.raise_if_requirements_not_met.assert_called_once()
        cls._instance.quantize.assert_called_once()
