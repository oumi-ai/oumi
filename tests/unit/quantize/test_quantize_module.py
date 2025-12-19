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

"""Unit tests for main quantize module functionality."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize import quantize
from oumi.quantize.base import QuantizationResult


class TestQuantizeModule:
    """Test cases for the main quantize function."""

    def test_rejects_invalid_config_type(self):
        """Test quantize rejects non-QuantizationConfig input."""
        with pytest.raises(ValueError, match="Expected QuantizationConfig"):
            quantize("not a config")  # type: ignore

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_routes_to_correct_quantizer(self, mock_build_quantizer):
        """Test that quantize routes to correct quantizer and calls it."""
        mock_quantizer = MagicMock()
        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/test",
            quantization_method="llmc_W4A16_ASYM",
            format_type="safetensors",
        )
        mock_build_quantizer.return_value = mock_quantizer

        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="llmc_W4A16_ASYM",
            output_path="test",
        )
        result = quantize(config)

        mock_build_quantizer.assert_called_once_with("llmc_W4A16_ASYM")
        mock_quantizer.raise_if_requirements_not_met.assert_called_once()
        mock_quantizer.quantize.assert_called_once_with(config)
        assert isinstance(result, QuantizationResult)

    @patch("oumi.builders.quantizers.build_quantizer")
    def test_propagates_requirements_error(self, mock_build_quantizer):
        """Test that requirements errors are propagated."""
        mock_quantizer = MagicMock()
        mock_quantizer.raise_if_requirements_not_met.side_effect = RuntimeError(
            "Missing GPU"
        )
        mock_build_quantizer.return_value = mock_quantizer

        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="llmc_W4A16_ASYM",
            output_path="test",
        )
        with pytest.raises(RuntimeError, match="Missing GPU"):
            quantize(config)
