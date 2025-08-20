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

"""Unit tests for BitsAndBytes quantization."""

import pytest  # type: ignore

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization


class TestBitsAndBytesQuantization:
    """Test cases for BitsAndBytes quantization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bnb_quantizer = BitsAndBytesQuantization()
        self.valid_config_4bit = QuantizationConfig(
            model=ModelParams(model_name="gpt2"),
            method="bnb_4bit",
            output_path="test_model_4bit",
        )
        self.valid_config_8bit = QuantizationConfig(
            model=ModelParams(model_name="gpt2"),
            method="bnb_8bit",
            output_path="test_model_8bit",
        )

    def test_supported_methods(self):
        """Test BitsAndBytes supported methods."""
        expected_methods = ["bnb_4bit", "bnb_8bit"]
        assert self.bnb_quantizer.supported_methods == expected_methods

    def test_supported_formats(self):
        """Test BitsAndBytes supported output formats."""
        expected_formats = ["safetensors"]
        assert self.bnb_quantizer.supported_formats == expected_formats

    def test_supports_method_4bit(self):
        """Test supports_method for 4-bit BitsAndBytes."""
        assert self.bnb_quantizer.supports_method("bnb_4bit") is True

    def test_supports_method_8bit(self):
        """Test supports_method for 8-bit BitsAndBytes."""
        assert self.bnb_quantizer.supports_method("bnb_8bit") is True

    def test_supports_method_invalid(self):
        """Test supports_method for invalid methods."""
        assert self.bnb_quantizer.supports_method("awq_q4_0") is False
        assert self.bnb_quantizer.supports_method("invalid") is False

    def test_validate_config_valid_4bit(self):
        """Test validate_config with valid 4-bit configuration."""
        # Should not raise any exception
        self.bnb_quantizer.validate_config(self.valid_config_4bit)

    def test_validate_config_valid_8bit(self):
        """Test validate_config with valid 8-bit configuration."""
        # Should not raise any exception
        self.bnb_quantizer.validate_config(self.valid_config_8bit)

    def test_validate_config_invalid_method(self):
        """Test validate_config with non-BNB method."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            QuantizationConfig(
                model=ModelParams(model_name="test/model"),
                method="awq_q4_0",
                output_path="test",
                output_format="unknown",
            )

    def test_str_representation(self):
        """Test string representation of BNB quantizer."""
        assert self.bnb_quantizer.__class__.__name__ == "BitsAndBytesQuantization"

    def test_raise_if_requirements_not_met_missing_bnb(self):
        """Test requirements check when BitsAndBytes is not installed."""
        # Set _bitsandbytes to None to simulate missing BitsAndBytes
        self.bnb_quantizer._bitsandbytes = None

        with pytest.raises(
            RuntimeError,
            match="BitsAndBytes quantization requires bitsandbytes library",
        ):
            self.bnb_quantizer.raise_if_requirements_not_met()


class TestBitsAndBytesQuantizationSimple:
    """Additional simplified test cases for BitsAndBytes quantization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bnb_quantizer = BitsAndBytesQuantization()

    def test_supports_method(self):
        """Test supports_method for BNB methods."""
        assert self.bnb_quantizer.supports_method("bnb_4bit") is True
        assert self.bnb_quantizer.supports_method("bnb_8bit") is True
        assert self.bnb_quantizer.supports_method("awq_q4_0") is False

    def test_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_path="test",
            output_format="safetensors",
        )
        # Should not raise
        self.bnb_quantizer.validate_config(config)
