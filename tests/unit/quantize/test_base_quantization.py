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

"""Unit tests for base quantization functionality."""

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.constants import QuantizationMethod


class _StubQuantizer(BaseQuantization):
    """Minimal concrete subclass for testing BaseQuantization."""

    supported_methods = [QuantizationMethod.FP8_DYNAMIC]
    supported_formats = ["safetensors"]

    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        return QuantizationResult(
            quantized_size_bytes=1000,
            output_path="/fake/path",
            quantization_method=QuantizationMethod.FP8_DYNAMIC,
            format_type="safetensors",
            additional_info={},
        )

    def raise_if_requirements_not_met(self) -> None:
        pass


class TestBaseQuantization:

    def setup_method(self):
        self.quantizer = _StubQuantizer()
        self.valid_config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="fp8_dynamic",
            output_path="test_output",
            output_format="safetensors",
        )

    def test_supports_method_true(self):
        assert self.quantizer.supports_method(QuantizationMethod.FP8_DYNAMIC) is True

    def test_supports_method_false(self):
        assert self.quantizer.supports_method(QuantizationMethod.BNB_4BIT) is False

    def test_supports_format_true(self):
        assert self.quantizer.supports_format("safetensors") is True

    def test_supports_format_false(self):
        assert self.quantizer.supports_format("gguf") is False

    def test_validate_config_valid(self):
        self.quantizer.validate_config(self.valid_config)

    def test_validate_config_invalid_method(self):
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method=QuantizationMethod.BNB_4BIT,
            output_path="test",
            output_format="safetensors",
        )
        with pytest.raises(ValueError, match="not supported by"):
            self.quantizer.validate_config(config)

    def test_quantize_implementation(self):
        result = self.quantizer.quantize(self.valid_config)

        assert isinstance(result, QuantizationResult)
        assert result.quantized_size_bytes == 1000
        assert result.output_path == "/fake/path"
        assert result.quantization_method == QuantizationMethod.FP8_DYNAMIC
        assert result.format_type == "safetensors"


class TestQuantizationResult:

    def test_creation_with_all_fields(self):
        result = QuantizationResult(
            quantized_size_bytes=2048,
            output_path="/path/to/model",
            quantization_method=QuantizationMethod.FP8_DYNAMIC,
            format_type="safetensors",
            additional_info={"compression_ratio": 0.25},
        )

        assert result.quantized_size_bytes == 2048
        assert result.output_path == "/path/to/model"
        assert result.quantization_method == QuantizationMethod.FP8_DYNAMIC
        assert result.format_type == "safetensors"
        assert result.additional_info["compression_ratio"] == 0.25

    def test_default_additional_info(self):
        result = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/path",
            quantization_method=QuantizationMethod.FP8_DYNAMIC,
            format_type="format",
        )

        assert result.additional_info == {}
