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

"""Unit tests for llm_compressor quantization."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.quantize.llm_compressor_quantizer import LlmCompressorQuantization


class TestLlmCompressorQuantization:
    """Test cases for llm_compressor quantization."""

    def test_supported_methods(self):
        """Test that llm_compressor quantizer has correct supported methods."""
        assert "llmc_W4A16" in LlmCompressorQuantization.supported_methods
        assert "llmc_W4A16_ASYM" in LlmCompressorQuantization.supported_methods
        assert "llmc_W8A8_INT" in LlmCompressorQuantization.supported_methods

    def test_requirements_not_met_missing_library(self):
        """Test error when llmcompressor is not installed."""
        quantizer = LlmCompressorQuantization()
        quantizer._llmcompressor = None
        with pytest.raises(RuntimeError, match="requires llmcompressor library"):
            quantizer.raise_if_requirements_not_met()

    @patch("torch.cuda.is_available", return_value=False)
    def test_requirements_not_met_no_gpu(self, mock_cuda):
        """Test error when no GPU is available."""
        quantizer = LlmCompressorQuantization()
        quantizer._llmcompressor = MagicMock()
        with pytest.raises(RuntimeError, match="requires a GPU"):
            quantizer.raise_if_requirements_not_met()
