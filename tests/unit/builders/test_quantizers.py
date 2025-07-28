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

from unittest.mock import Mock, patch

import pytest  # type: ignore

from oumi.builders.quantizers import (
    build_quantizer,
    get_available_methods,
    get_supported_formats,
    list_all_methods,
)
from oumi.quantize.base import BaseQuantization


class TestQuantizerBuilders:
    """Test cases for quantizer builder functions."""

    def test_build_quantizer_awq_method(self):
        """Test building AWQ quantizer for AWQ methods."""
        # Test various AWQ methods
        for method in ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]:
            quantizer = build_quantizer(method)

            # Verify correct type
            assert quantizer.__class__.__name__ == "AwqQuantization"
            assert isinstance(quantizer, BaseQuantization)

    def test_build_quantizer_bnb_method(self):
        """Test building BnB quantizer for BnB methods."""
        # Test BnB methods
        for method in ["bnb_4bit", "bnb_8bit"]:
            quantizer = build_quantizer(method)

            # Verify correct type
            assert quantizer.__class__.__name__ == "BitsAndBytesQuantization"
            assert isinstance(quantizer, BaseQuantization)

    def test_build_quantizer_unsupported_method(self):
        """Test building quantizer with unsupported method."""
        with pytest.raises(
            ValueError, match="Unsupported quantization method: invalid_method"
        ):
            build_quantizer("invalid_method")

    @patch("oumi.quantize.awq_quantizer.AwqQuantization")
    @patch("oumi.quantize.bnb_quantizer.BitsAndBytesQuantization")
    def test_build_quantizer_fallback(self, mock_bnb, mock_awq):
        """Test build_quantizer fallback when method doesn't match prefix."""
        # Setup mocks
        mock_awq_instance = Mock()
        mock_awq_instance.supports_method.return_value = False
        mock_awq.return_value = mock_awq_instance

        mock_bnb_instance = Mock()
        mock_bnb_instance.supports_method.return_value = True
        mock_bnb.return_value = mock_bnb_instance

        # Test with method that doesn't match any prefix
        result = build_quantizer("custom_method")

        # Verify BNB instance was returned (since it supports the method)
        assert result == mock_bnb_instance

        # Verify both were checked
        mock_awq_instance.supports_method.assert_called_with("custom_method")
        mock_bnb_instance.supports_method.assert_called_with("custom_method")

    def test_get_available_methods(self):
        """Test getting all available quantization methods."""
        methods = get_available_methods()

        # Verify structure
        assert isinstance(methods, dict)
        assert "AWQ" in methods
        assert "BitsAndBytes" in methods

        # Verify AWQ methods
        assert "awq_q4_0" in methods["AWQ"]
        assert "awq_q4_1" in methods["AWQ"]
        assert "awq_q8_0" in methods["AWQ"]
        assert "awq_f16" in methods["AWQ"]

        # Verify BitsAndBytes methods
        assert "bnb_4bit" in methods["BitsAndBytes"]
        assert "bnb_8bit" in methods["BitsAndBytes"]

    def test_get_supported_formats(self):
        """Test getting all supported output formats."""
        formats = get_supported_formats()

        # Verify it's a list
        assert isinstance(formats, list)

        # Verify expected formats
        assert "pytorch" in formats
        assert "safetensors" in formats

        # Verify sorted
        assert formats == sorted(formats)

    def test_list_all_methods(self):
        """Test listing all available methods."""
        all_methods = list_all_methods()

        # Verify it's a list
        assert isinstance(all_methods, list)

        # Verify contains expected methods
        expected_methods = [
            "awq_q4_0",
            "awq_q4_1",
            "awq_q8_0",
            "awq_f16",
            "bnb_4bit",
            "bnb_8bit",
        ]
        for method in expected_methods:
            assert method in all_methods

        # Verify sorted
        assert all_methods == sorted(all_methods)

    def test_get_available_methods_structure(self):
        """Test the structure of available methods dictionary."""
        methods = get_available_methods()

        # Each value should be a list
        for quantizer_name, method_list in methods.items():
            assert isinstance(method_list, list)
            assert len(method_list) > 0

            # Each method should be a string
            for method in method_list:
                assert isinstance(method, str)

    @patch("oumi.quantize.awq_quantizer.AwqQuantization")
    def test_build_quantizer_creates_new_instance(self, mock_awq):
        """Test that build_quantizer creates new instances each time."""
        # Setup mock
        mock_awq.side_effect = [Mock(), Mock()]  # Return different instances

        # Build two quantizers
        q1 = build_quantizer("awq_q4_0")
        q2 = build_quantizer("awq_q4_0")

        # Verify they are different instances
        assert q1 is not q2

        # Verify constructor was called twice
        assert mock_awq.call_count == 2
