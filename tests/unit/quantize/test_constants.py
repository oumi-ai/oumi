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

"""Unit tests for quantization constants."""

import pytest

from oumi.quantize.constants import (
    AWQ_DEFAULTS,
    AWQ_TO_GGUF_METHOD_MAP,
    GGUF_QUANTIZATION_MAP,
    MOCK_MODEL_SIZES,
    SUPPORTED_METHODS,
    SUPPORTED_OUTPUT_FORMATS,
)


class TestSupportedMethods:
    """Test supported quantization methods."""

    def test_awq_methods_included(self):
        """Test that AWQ methods are supported."""
        awq_methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
        for method in awq_methods:
            assert method in SUPPORTED_METHODS

    def test_bitsandbytes_methods_included(self):
        """Test that BitsAndBytes methods are supported."""
        bnb_methods = ["bnb_4bit", "bnb_8bit"]
        for method in bnb_methods:
            assert method in SUPPORTED_METHODS

    def test_gguf_methods_included(self):
        """Test that GGUF methods are supported."""
        gguf_methods = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"]
        for method in gguf_methods:
            assert method in SUPPORTED_METHODS

    def test_no_duplicate_methods(self):
        """Test that there are no duplicate methods."""
        assert len(SUPPORTED_METHODS) == len(set(SUPPORTED_METHODS))


class TestSupportedOutputFormats:
    """Test supported output formats."""

    def test_required_formats_included(self):
        """Test that required output formats are supported."""
        required_formats = ["gguf", "safetensors", "pytorch"]
        for format_name in required_formats:
            assert format_name in SUPPORTED_OUTPUT_FORMATS

    def test_no_duplicate_formats(self):
        """Test that there are no duplicate formats."""
        assert len(SUPPORTED_OUTPUT_FORMATS) == len(set(SUPPORTED_OUTPUT_FORMATS))


class TestAwqToGgufMethodMap:
    """Test AWQ to GGUF method mapping."""

    def test_awq_methods_mapped(self):
        """Test that all AWQ methods have GGUF mappings."""
        awq_methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
        for method in awq_methods:
            assert method in AWQ_TO_GGUF_METHOD_MAP
            # Verify mapping is to a valid GGUF method
            mapped_method = AWQ_TO_GGUF_METHOD_MAP[method]
            assert mapped_method in ["q4_0", "q4_1", "q8_0", "f16"]

    def test_mapping_consistency(self):
        """Test that mappings are consistent."""
        assert AWQ_TO_GGUF_METHOD_MAP["awq_q4_0"] == "q4_0"
        assert AWQ_TO_GGUF_METHOD_MAP["awq_q4_1"] == "q4_1"
        assert AWQ_TO_GGUF_METHOD_MAP["awq_q8_0"] == "q8_0"
        assert AWQ_TO_GGUF_METHOD_MAP["awq_f16"] == "f16"


class TestGgufQuantizationMap:
    """Test GGUF quantization mapping."""

    def test_required_quantization_types(self):
        """Test that required quantization types are mapped."""
        required_types = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
        for quant_type in required_types:
            assert quant_type in GGUF_QUANTIZATION_MAP
            # Verify mapping is to a numeric value
            assert isinstance(GGUF_QUANTIZATION_MAP[quant_type], int)

    def test_quantization_values(self):
        """Test that quantization values are reasonable."""
        # Values should be small positive integers (llama.cpp constants)
        for value in GGUF_QUANTIZATION_MAP.values():
            assert isinstance(value, int)
            assert 0 <= value <= 20  # Reasonable range for llama.cpp constants


class TestAwqDefaults:
    """Test AWQ default configuration."""

    def test_required_defaults_present(self):
        """Test that required default values are present."""
        required_keys = [
            "calibration_dataset",
            "calibration_split", 
            "calibration_text_column",
            "max_calibration_seq_len",
            "duo_scaling",
            "apply_clip",
            "n_parallel_calib_samples"
        ]
        for key in required_keys:
            assert key in AWQ_DEFAULTS

    def test_defaults_types(self):
        """Test that default values have correct types."""
        assert isinstance(AWQ_DEFAULTS["calibration_dataset"], str)
        assert isinstance(AWQ_DEFAULTS["calibration_split"], str)
        assert isinstance(AWQ_DEFAULTS["calibration_text_column"], str)
        assert isinstance(AWQ_DEFAULTS["max_calibration_seq_len"], int)
        assert isinstance(AWQ_DEFAULTS["duo_scaling"], bool)
        assert isinstance(AWQ_DEFAULTS["apply_clip"], bool)
        # n_parallel_calib_samples can be None or int
        assert AWQ_DEFAULTS["n_parallel_calib_samples"] is None or isinstance(AWQ_DEFAULTS["n_parallel_calib_samples"], int)

    def test_defaults_values(self):
        """Test that default values are reasonable."""
        assert AWQ_DEFAULTS["max_calibration_seq_len"] > 0
        # n_parallel_calib_samples can be None, so only check if it's not None
        if AWQ_DEFAULTS["n_parallel_calib_samples"] is not None:
            assert AWQ_DEFAULTS["n_parallel_calib_samples"] > 0


class TestMockModelSizes:
    """Test mock model sizes for simulation."""

    def test_required_sizes_present(self):
        """Test that required model sizes are present."""
        required_sizes = ["small", "7b", "13b", "70b", "default"]
        for size_key in required_sizes:
            assert size_key in MOCK_MODEL_SIZES

    def test_sizes_are_integers(self):
        """Test that all sizes are integers."""
        for size in MOCK_MODEL_SIZES.values():
            assert isinstance(size, int)
            assert size > 0

    def test_size_relationships(self):
        """Test that model sizes have logical relationships."""
        # Larger models should be larger than smaller models
        assert MOCK_MODEL_SIZES["7b"] > MOCK_MODEL_SIZES["small"]
        assert MOCK_MODEL_SIZES["13b"] > MOCK_MODEL_SIZES["7b"]
        assert MOCK_MODEL_SIZES["70b"] > MOCK_MODEL_SIZES["13b"]
        
        # All models should be larger than default
        assert MOCK_MODEL_SIZES["7b"] > MOCK_MODEL_SIZES["default"]