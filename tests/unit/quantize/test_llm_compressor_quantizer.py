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

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.llm_compressor_quantizer import LlmCompressorQuantization


class TestLlmCompressorQuantization:
    """Test cases for llm_compressor quantization."""

    def test_validate_config_rejects_invalid_method(self):
        """Test that validate_config rejects non-llmc methods."""
        quantizer = LlmCompressorQuantization()
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_path="test",
        )
        with pytest.raises(ValueError, match="not supported by"):
            quantizer.validate_config(config)

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
