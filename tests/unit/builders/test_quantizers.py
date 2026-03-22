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

"""Unit tests for quantizer builders."""

import pytest

from oumi.builders.quantizers import (
    build_quantizer,
    get_available_methods,
    get_supported_formats,
    list_all_methods,
)
from oumi.quantize.base import BaseQuantization
from oumi.quantize.constants import QuantizationMethod


class TestBuildQuantizer:

    @pytest.mark.parametrize(
        "method",
        [
            QuantizationMethod.FP8_DYNAMIC,
            QuantizationMethod.FP8_BLOCK,
            QuantizationMethod.W4A16,
            QuantizationMethod.W4A16_ASYM,
            QuantizationMethod.W8A16,
        ],
    )
    def test_llmcompressor_methods(self, method):
        quantizer = build_quantizer(method)
        assert isinstance(quantizer, BaseQuantization)
        assert quantizer.__class__.__name__ == "LLMCompressorQuantization"

    @pytest.mark.parametrize(
        "method",
        [QuantizationMethod.BNB_4BIT, QuantizationMethod.BNB_8BIT],
    )
    def test_bnb_methods(self, method):
        quantizer = build_quantizer(method)
        assert isinstance(quantizer, BaseQuantization)
        assert quantizer.__class__.__name__ == "BitsAndBytesQuantization"

    def test_creates_new_instance_each_call(self):
        q1 = build_quantizer(QuantizationMethod.FP8_DYNAMIC)
        q2 = build_quantizer(QuantizationMethod.FP8_DYNAMIC)
        assert q1 is not q2


class TestGetAvailableMethods:

    def test_returns_both_quantizer_groups(self):
        methods = get_available_methods()
        assert "LLMCompressor" in methods
        assert "BitsAndBytes" in methods

    def test_llmcompressor_methods_present(self):
        methods = get_available_methods()
        llmc_values = {m.value for m in methods["LLMCompressor"]}
        for expected in ("fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16"):
            assert expected in llmc_values

    def test_bnb_methods_present(self):
        methods = get_available_methods()
        bnb_values = {m.value for m in methods["BitsAndBytes"]}
        assert "bnb_4bit" in bnb_values
        assert "bnb_8bit" in bnb_values

    def test_old_awq_names_excluded(self):
        methods = get_available_methods()
        all_values = set()
        for method_list in methods.values():
            all_values.update(m.value for m in method_list)
        for old in ("awq_q4_0", "awq_q4_1", "awq_q8_0"):
            assert old not in all_values


class TestListAllMethods:

    def test_returns_sorted_list(self):
        all_methods = list_all_methods()
        assert all_methods == sorted(all_methods)

    def test_includes_all_expected_methods(self):
        all_values = {m.value for m in list_all_methods()}
        expected = {"fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16",
                    "bnb_4bit", "bnb_8bit"}
        assert expected <= all_values


class TestGetSupportedFormats:

    def test_includes_safetensors(self):
        formats = get_supported_formats()
        assert "safetensors" in formats

    def test_returns_sorted(self):
        formats = get_supported_formats()
        assert formats == sorted(formats)
