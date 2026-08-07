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

"""Unit tests for quantization constants and SCHEME_REGISTRY."""

import pytest

from oumi.core.configs.quantization_config import (
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.quantize.constants import (
    BNB_SCHEMES,
    LLMCOMPRESSOR_SCHEMES,
    SCHEME_REGISTRY,
    SUPPORTED_OUTPUT_FORMATS,
    QuantizationAlgorithm,
    SchemeInfo,
)


class TestSupportedSchemes:
    def test_includes_llmcompressor_schemes(self):
        values = {s.value for s in LLMCOMPRESSOR_SCHEMES}
        for expected in ("fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16"):
            assert expected in values

    def test_includes_bnb_schemes(self):
        values = {s.value for s in BNB_SCHEMES}
        assert "bnb_nf4" in values
        assert "bnb_fp4" in values
        assert "bnb_int8" in values

    def test_no_overlap_between_llmc_and_bnb(self):
        assert set(LLMCOMPRESSOR_SCHEMES).isdisjoint(set(BNB_SCHEMES))

    def test_supported_output_formats(self):
        assert SUPPORTED_OUTPUT_FORMATS == ["safetensors"]


class TestSchemeRegistry:
    def test_all_registry_entries_are_scheme_info(self):
        for scheme, info in SCHEME_REGISTRY.items():
            assert isinstance(scheme, QuantizationScheme)
            assert isinstance(info, SchemeInfo)

    @pytest.mark.parametrize("scheme", list(SCHEME_REGISTRY.keys()))
    def test_registry_entry_has_required_fields(self, scheme):
        info = SCHEME_REGISTRY[scheme]
        assert isinstance(info.backend, QuantizationBackend)
        assert isinstance(info.default_algorithm, QuantizationAlgorithm)
        assert isinstance(info.needs_calibration, bool)
        assert isinstance(info.min_compute_capability, float)
        assert isinstance(info.description, str)
        assert info.description != ""

    @pytest.mark.parametrize(
        "scheme",
        [QuantizationScheme.FP8_DYNAMIC, QuantizationScheme.FP8_BLOCK],
    )
    def test_data_free_schemes(self, scheme):
        info = SCHEME_REGISTRY[scheme]
        assert info.needs_calibration is False
        assert info.llmc_scheme is not None

    @pytest.mark.parametrize(
        "scheme",
        [
            QuantizationScheme.W4A16,
            QuantizationScheme.W4A16_ASYM,
            QuantizationScheme.W8A16,
        ],
    )
    def test_calibration_schemes(self, scheme):
        info = SCHEME_REGISTRY[scheme]
        assert info.needs_calibration is True
        assert info.llmc_scheme is not None

    @pytest.mark.parametrize(
        "scheme",
        [
            QuantizationScheme.BNB_NF4,
            QuantizationScheme.BNB_FP4,
            QuantizationScheme.BNB_INT8,
        ],
    )
    def test_bnb_schemes(self, scheme):
        info = SCHEME_REGISTRY[scheme]
        assert info.llmc_scheme is None
        assert info.backend == QuantizationBackend.BNB
        assert info.default_algorithm == QuantizationAlgorithm.BNB
        assert info.needs_calibration is False

    def test_registry_covers_all_enum_members(self):
        for scheme in QuantizationScheme:
            assert scheme in SCHEME_REGISTRY, f"{scheme} missing from SCHEME_REGISTRY"

    def test_registry_backend_consistency(self):
        for scheme, info in SCHEME_REGISTRY.items():
            if scheme in BNB_SCHEMES:
                assert info.backend == QuantizationBackend.BNB
            elif scheme in LLMCOMPRESSOR_SCHEMES:
                assert info.backend == QuantizationBackend.LLM_COMPRESSOR
