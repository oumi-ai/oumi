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

"""Unit tests for base quantization types: SchemeSpec, QuantizationResult, ABC."""

import pytest

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError
from oumi.quantize.base import BaseQuantization, QuantizationResult, SchemeSpec

_LLMC_ALGOS = (
    QuantizationAlgorithm.RTN,
    QuantizationAlgorithm.GPTQ,
    QuantizationAlgorithm.AWQ,
)


class TestSchemeSpec:
    def test_resolve_auto_returns_default(self):
        spec = SchemeSpec(
            default_algorithm=QuantizationAlgorithm.RTN,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=False,
        )
        assert (
            spec.resolve_algorithm(QuantizationAlgorithm.AUTO)
            is QuantizationAlgorithm.RTN
        )

    def test_resolve_explicit_allowed(self):
        spec = SchemeSpec(
            default_algorithm=QuantizationAlgorithm.RTN,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=False,
        )
        assert (
            spec.resolve_algorithm(QuantizationAlgorithm.GPTQ)
            is QuantizationAlgorithm.GPTQ
        )

    def test_resolve_disallowed_raises(self):
        spec = SchemeSpec(
            default_algorithm=QuantizationAlgorithm.BNB,
            allowed_algorithms=(QuantizationAlgorithm.BNB,),
            needs_calibration_default=False,
        )
        with pytest.raises(OumiConfigError, match="not allowed"):
            spec.resolve_algorithm(QuantizationAlgorithm.GPTQ)

    def test_needs_calibration_default_only(self):
        spec = SchemeSpec(
            default_algorithm=QuantizationAlgorithm.GPTQ,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=True,
        )
        assert spec.needs_calibration_for(QuantizationAlgorithm.GPTQ) is True
        assert spec.needs_calibration_for(QuantizationAlgorithm.RTN) is True

    def test_needs_calibration_only_when_overridden_with_calib_algo(self):
        spec = SchemeSpec(
            default_algorithm=QuantizationAlgorithm.RTN,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=False,
            calibration_required_for=(
                QuantizationAlgorithm.GPTQ,
                QuantizationAlgorithm.AWQ,
            ),
        )
        assert spec.needs_calibration_for(QuantizationAlgorithm.RTN) is False
        assert spec.needs_calibration_for(QuantizationAlgorithm.GPTQ) is True
        assert spec.needs_calibration_for(QuantizationAlgorithm.AWQ) is True


class _StubQuantizer(BaseQuantization):
    """Minimal concrete subclass for ABC-shape testing."""

    backend = QuantizationBackend.LLM_COMPRESSOR
    schemes = {
        QuantizationScheme.FP8_DYNAMIC: SchemeSpec(
            default_algorithm=QuantizationAlgorithm.RTN,
            allowed_algorithms=_LLMC_ALGOS,
            needs_calibration_default=False,
        )
    }

    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        return QuantizationResult(
            output_path="/fake/path",
            backend=self.backend,
            scheme=QuantizationScheme.FP8_DYNAMIC,
            format_type=self.output_format,
            quantized_size_bytes=1000,
        )

    def raise_if_requirements_not_met(self) -> None:
        pass


class TestBaseQuantization:
    def test_owns_true(self):
        assert _StubQuantizer.owns(QuantizationScheme.FP8_DYNAMIC) is True

    def test_owns_false(self):
        assert _StubQuantizer.owns(QuantizationScheme.BNB_NF4) is False

    def test_default_output_format(self):
        assert _StubQuantizer.output_format == "safetensors"

    def test_quantize_returns_result(self):
        quantizer = _StubQuantizer()
        result = quantizer.quantize(
            QuantizationConfig(scheme="fp8_dynamic", output_path="x")
        )
        assert isinstance(result, QuantizationResult)
        assert result.backend is QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme is QuantizationScheme.FP8_DYNAMIC


class TestQuantizationResult:
    def test_creation_with_all_fields(self):
        result = QuantizationResult(
            output_path="/path/to/model",
            backend=QuantizationBackend.LLM_COMPRESSOR,
            scheme=QuantizationScheme.FP8_DYNAMIC,
            format_type="safetensors",
            quantized_size_bytes=2048,
        )
        assert result.quantized_size_bytes == 2048
        assert result.output_path == "/path/to/model"
        assert result.backend is QuantizationBackend.LLM_COMPRESSOR
        assert result.scheme is QuantizationScheme.FP8_DYNAMIC
        assert result.format_type == "safetensors"
