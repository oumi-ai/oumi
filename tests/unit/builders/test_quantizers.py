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

from oumi.builders.quantizers import (
    build_quantizer,
    get_available_schemes,
    get_supported_formats,
    list_all_schemes,
)
from oumi.core.configs.quantization_config import QuantizationBackend
from oumi.quantize.base import BaseQuantization


class TestBuildQuantizer:
    def test_llmcompressor_backend(self):
        quantizer = build_quantizer(QuantizationBackend.LLM_COMPRESSOR)
        assert isinstance(quantizer, BaseQuantization)
        assert quantizer.__class__.__name__ == "LLMCompressorQuantization"

    def test_bnb_backend(self):
        quantizer = build_quantizer(QuantizationBackend.BNB)
        assert isinstance(quantizer, BaseQuantization)
        assert quantizer.__class__.__name__ == "BitsAndBytesQuantization"

    def test_creates_new_instance_each_call(self):
        q1 = build_quantizer(QuantizationBackend.LLM_COMPRESSOR)
        q2 = build_quantizer(QuantizationBackend.LLM_COMPRESSOR)
        assert q1 is not q2


class TestGetAvailableSchemes:
    def test_returns_both_quantizer_groups(self):
        schemes = get_available_schemes()
        assert "LLMCompressor" in schemes
        assert "BitsAndBytes" in schemes

    def test_llmcompressor_schemes_present(self):
        schemes = get_available_schemes()
        llmc_values = {s.value for s in schemes["LLMCompressor"]}
        for expected in ("fp8_dynamic", "fp8_block", "w4a16", "w4a16_asym", "w8a16"):
            assert expected in llmc_values

    def test_bnb_schemes_present(self):
        schemes = get_available_schemes()
        bnb_values = {s.value for s in schemes["BitsAndBytes"]}
        assert "bnb_nf4" in bnb_values
        assert "bnb_fp4" in bnb_values
        assert "bnb_int8" in bnb_values


class TestListAllSchemes:
    def test_returns_sorted_list(self):
        all_schemes = list_all_schemes()
        assert all_schemes == sorted(all_schemes)

    def test_includes_all_expected_schemes(self):
        all_values = {s.value for s in list_all_schemes()}
        expected = {
            "fp8_dynamic",
            "fp8_block",
            "w4a16",
            "w4a16_asym",
            "w8a16",
            "bnb_nf4",
            "bnb_fp4",
            "bnb_int8",
        }
        assert expected <= all_values


class TestGetSupportedFormats:
    def test_includes_safetensors(self):
        formats = get_supported_formats()
        assert "safetensors" in formats

    def test_returns_sorted(self):
        formats = get_supported_formats()
        assert formats == sorted(formats)
