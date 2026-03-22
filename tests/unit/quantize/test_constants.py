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

"""Unit tests for quantization constants and METHOD_REGISTRY."""

import pytest

from oumi.quantize.constants import (
    BNB_METHODS,
    LLMCOMPRESSOR_METHODS,
    METHOD_REGISTRY,
    SUPPORTED_OUTPUT_FORMATS,
    MethodInfo,
    QuantizationAlgorithm,
    QuantizationMethod,
)


class TestSupportedMethods:

    def test_includes_new_llmcompressor_methods(self):
        values = {m.value for m in LLMCOMPRESSOR_METHODS}
        for expected in ("fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16"):
            assert expected in values

    def test_includes_bnb_methods(self):
        values = {m.value for m in BNB_METHODS}
        assert "bnb_4bit" in values
        assert "bnb_8bit" in values

    def test_excludes_old_awq_names(self):
        all_values = {m.value for m in LLMCOMPRESSOR_METHODS + BNB_METHODS}
        for old in ("awq_q4_0", "awq_q4_1", "awq_q8_0"):
            assert old not in all_values

    def test_no_overlap_between_llmc_and_bnb(self):
        assert set(LLMCOMPRESSOR_METHODS).isdisjoint(set(BNB_METHODS))

    def test_supported_output_formats(self):
        assert SUPPORTED_OUTPUT_FORMATS == ["safetensors"]


class TestMethodRegistry:

    def test_all_registry_entries_are_method_info(self):
        for method, info in METHOD_REGISTRY.items():
            assert isinstance(method, QuantizationMethod)
            assert isinstance(info, MethodInfo)

    @pytest.mark.parametrize("method", list(METHOD_REGISTRY.keys()))
    def test_registry_entry_has_required_fields(self, method):
        info = METHOD_REGISTRY[method]
        assert isinstance(info.algorithm, QuantizationAlgorithm)
        assert isinstance(info.needs_calibration, bool)
        assert isinstance(info.min_compute_capability, float)
        assert isinstance(info.description, str)
        assert info.description != ""

    @pytest.mark.parametrize(
        "method",
        [QuantizationMethod.FP8_DYNAMIC, QuantizationMethod.FP8_BLOCK],
    )
    def test_data_free_methods(self, method):
        info = METHOD_REGISTRY[method]
        assert info.needs_calibration is False
        assert info.scheme is not None

    @pytest.mark.parametrize(
        "method",
        [QuantizationMethod.W4A16, QuantizationMethod.W4A16_ASYM,
         QuantizationMethod.W8A16],
    )
    def test_calibration_methods(self, method):
        info = METHOD_REGISTRY[method]
        assert info.needs_calibration is True
        assert info.scheme is not None

    @pytest.mark.parametrize(
        "method",
        [QuantizationMethod.BNB_4BIT, QuantizationMethod.BNB_8BIT],
    )
    def test_bnb_methods_no_scheme(self, method):
        info = METHOD_REGISTRY[method]
        assert info.scheme is None
        assert info.algorithm == QuantizationAlgorithm.BNB
        assert info.needs_calibration is False

    def test_registry_covers_all_enum_members(self):
        for method in QuantizationMethod:
            assert method in METHOD_REGISTRY, (
                f"{method} missing from METHOD_REGISTRY"
            )
