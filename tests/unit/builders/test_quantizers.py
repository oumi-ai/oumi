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

from oumi.builders.quantizers import build_quantizer, get_available_methods
from oumi.quantize.base import BaseQuantization


class TestQuantizerBuilders:
    """Test cases for quantizer builder functions."""

    def test_build_quantizer_routes_llmc_methods(self):
        """Test that llmc_ methods route to LlmCompressorQuantization."""
        quantizer = build_quantizer("llmc_W4A16_ASYM")
        assert quantizer.__class__.__name__ == "LlmCompressorQuantization"
        assert isinstance(quantizer, BaseQuantization)

    def test_build_quantizer_routes_bnb_methods(self):
        """Test that bnb_ methods route to BitsAndBytesQuantization."""
        quantizer = build_quantizer("bnb_4bit")
        assert quantizer.__class__.__name__ == "BitsAndBytesQuantization"
        assert isinstance(quantizer, BaseQuantization)

    def test_build_quantizer_rejects_unsupported_method(self):
        """Test that unsupported methods raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            build_quantizer("invalid_method")

    def test_get_available_methods_returns_all_quantizers(self):
        """Test that get_available_methods returns methods for all quantizers."""
        methods = get_available_methods()
        assert "LlmCompressor" in methods
        assert "BitsAndBytes" in methods
        assert "llmc_W4A16_ASYM" in methods["LlmCompressor"]
        assert "bnb_4bit" in methods["BitsAndBytes"]
