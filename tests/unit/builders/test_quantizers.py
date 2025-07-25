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

import pytest

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
        with patch("oumi.builders.quantizers.AwqQuantization") as mock_awq:
            mock_instance = Mock()
            mock_awq.return_value = mock_instance

            result = build_quantizer("awq_4bit")

            # Verify AWQ quantizer was created
            mock_awq.assert_called_once()
            assert result == mock_instance

    def test_build_quantizer_bnb_method(self):
        """Test building BnB quantizer for BnB methods."""
        with patch("oumi.builders.quantizers.BitsAndBytesQuantization") as mock_bnb:
            mock_instance = Mock()
            mock_bnb.return_value = mock_instance

            result = build_quantizer("bnb_4bit")

            # Verify BnB quantizer was created
            mock_bnb.assert_called_once()
            assert result == mock_instance

    def test_build_quantizer_awq_prefix_variations(self):
        """Test building quantizer for various AWQ prefixed methods."""
        awq_methods = ["awq_4bit", "awq_q4_0", "awq_q8_0", "awq_q5_k_s"]

        for method in awq_methods:
            with patch("oumi.builders.quantizers.AwqQuantization") as mock_awq:
                mock_instance = Mock()
                mock_awq.return_value = mock_instance

                result = build_quantizer(method)

                # Verify AWQ quantizer was created
                mock_awq.assert_called_once()
                assert result == mock_instance

    def test_build_quantizer_bnb_prefix_variations(self):
        """Test building quantizer for various BnB prefixed methods."""
        bnb_methods = ["bnb_4bit", "bnb_8bit"]

        for method in bnb_methods:
            with patch("oumi.builders.quantizers.BitsAndBytesQuantization") as mock_bnb:
                mock_instance = Mock()
                mock_bnb.return_value = mock_instance

                result = build_quantizer(method)

                # Verify BnB quantizer was created
                mock_bnb.assert_called_once()
                assert result == mock_instance

    @patch("oumi.builders.quantizers.AwqQuantization")
    @patch("oumi.builders.quantizers.BitsAndBytesQuantization")
    def test_build_quantizer_fallback_search_success(self, mock_bnb, mock_awq):
        """Test building quantizer using fallback search."""
        # Setup mocks
        mock_awq_instance = Mock()
        mock_awq_instance.supports_method.return_value = False
        mock_awq.return_value = mock_awq_instance

        mock_bnb_instance = Mock()
        mock_bnb_instance.supports_method.return_value = True
        mock_bnb.return_value = mock_bnb_instance

        # Test with method that doesn't match prefixes but is supported by BnB
        result = build_quantizer("custom_method")

        # Verify fallback search was used
        mock_awq_instance.supports_method.assert_called_once_with("custom_method")
        mock_bnb_instance.supports_method.assert_called_once_with("custom_method")

        # Verify correct quantizer was returned
        assert result == mock_bnb_instance

    @patch("oumi.builders.quantizers.get_available_methods")
    @patch("oumi.builders.quantizers.AwqQuantization")
    @patch("oumi.builders.quantizers.BitsAndBytesQuantization")
    def test_build_quantizer_unsupported_method(
        self, mock_bnb, mock_awq, mock_get_methods
    ):
        """Test building quantizer with unsupported method."""
        # Setup mocks
        mock_awq_instance = Mock()
        mock_awq_instance.supports_method.return_value = False
        mock_awq.return_value = mock_awq_instance

        mock_bnb_instance = Mock()
        mock_bnb_instance.supports_method.return_value = False
        mock_bnb.return_value = mock_bnb_instance

        mock_get_methods.return_value = {
            "AWQ": ["awq_4bit"],
            "BitsAndBytes": ["bnb_4bit"],
        }

        # Test with unsupported method
        with pytest.raises(
            ValueError, match="Unsupported quantization method: unsupported_method"
        ):
            build_quantizer("unsupported_method")

        # Verify all quantizers were checked
        mock_awq_instance.supports_method.assert_called_once_with("unsupported_method")
        mock_bnb_instance.supports_method.assert_called_once_with("unsupported_method")

        # Verify error message includes available methods
        mock_get_methods.assert_called_once()

    @patch("oumi.builders.quantizers.AwqQuantization")
    @patch("oumi.builders.quantizers.BitsAndBytesQuantization")
    def test_get_available_methods(self, mock_bnb, mock_awq):
        """Test getting available methods from all quantizers."""
        # Setup mock supported methods
        mock_awq.supported_methods = ["awq_4bit", "awq_q4_0", "awq_q8_0"]
        mock_bnb.supported_methods = ["bnb_4bit", "bnb_8bit"]

        result = get_available_methods()

        expected = {
            "AWQ": ["awq_4bit", "awq_q4_0", "awq_q8_0"],
            "BitsAndBytes": ["bnb_4bit", "bnb_8bit"],
        }

        assert result == expected

    @patch("oumi.builders.quantizers.AwqQuantization")
    @patch("oumi.builders.quantizers.BitsAndBytesQuantization")
    def test_get_supported_formats(self, mock_bnb, mock_awq):
        """Test getting supported formats from all quantizers."""
        # Setup mock supported formats
        mock_awq.supported_formats = ["safetensors", "pytorch"]
        mock_bnb.supported_formats = ["safetensors"]

        result = get_supported_formats()

        # Should return sorted unique formats
        expected = ["pytorch", "safetensors"]

        assert result == expected

    @patch("oumi.builders.quantizers.get_available_methods")
    def test_list_all_methods(self, mock_get_methods):
        """Test listing all available methods."""
        # Setup mock available methods
        mock_get_methods.return_value = {
            "AWQ": ["awq_4bit", "awq_q4_0"],
            "BitsAndBytes": ["bnb_4bit", "bnb_8bit"],
        }

        result = list_all_methods()

        # Should return sorted list of all methods
        expected = ["awq_4bit", "awq_q4_0", "bnb_4bit", "bnb_8bit"]

        assert result == expected
        mock_get_methods.assert_called_once()

    def test_build_quantizer_returns_base_quantization(self):
        """Test that build_quantizer returns BaseQuantization instance."""
        with patch("oumi.builders.quantizers.AwqQuantization") as mock_awq:
            mock_instance = Mock(spec=BaseQuantization)
            mock_awq.return_value = mock_instance

            result = build_quantizer("awq_4bit")

            # Verify instance conforms to BaseQuantization interface
            assert hasattr(result, "quantize")
            assert hasattr(result, "supports_method")
            assert hasattr(result, "supports_format")
            assert hasattr(result, "validate_config")

    def test_build_quantizer_case_sensitivity(self):
        """Test that build_quantizer handles method names case sensitively."""
        # Test that uppercase methods don't match
        with patch(
            "oumi.builders.quantizers.get_available_methods"
        ) as mock_get_methods:
            mock_get_methods.return_value = {"AWQ": [], "BitsAndBytes": []}

            with pytest.raises(
                ValueError, match="Unsupported quantization method: AWQ_4BIT"
            ):
                build_quantizer("AWQ_4BIT")

    @patch("oumi.builders.quantizers.AwqQuantization")
    @patch("oumi.builders.quantizers.BitsAndBytesQuantization")
    def test_build_quantizer_multiple_calls_create_new_instances(
        self, mock_bnb, mock_awq
    ):
        """Test that multiple calls to build_quantizer create new instances."""
        mock_awq_instance1 = Mock()
        mock_awq_instance2 = Mock()
        mock_awq.side_effect = [mock_awq_instance1, mock_awq_instance2]

        result1 = build_quantizer("awq_4bit")
        result2 = build_quantizer("awq_4bit")

        # Verify different instances are returned
        assert result1 == mock_awq_instance1
        assert result2 == mock_awq_instance2
        assert result1 != result2

        # Verify constructor was called twice
        assert mock_awq.call_count == 2

    def test_empty_method_string(self):
        """Test build_quantizer with empty method string."""
        with patch(
            "oumi.builders.quantizers.get_available_methods"
        ) as mock_get_methods:
            mock_get_methods.return_value = {"AWQ": [], "BitsAndBytes": []}

            with pytest.raises(ValueError, match="Unsupported quantization method: "):
                build_quantizer("")

    def test_none_method(self):
        """Test build_quantizer with None method."""
        with pytest.raises(AttributeError):
            build_quantizer(None)  # type: ignore

