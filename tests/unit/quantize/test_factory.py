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

"""Tests for quantization factory."""

import pytest

from oumi.quantize.awq_quantizer import AwqQuantization
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
from oumi.quantize.factory import (
    create_quantizer,
    get_available_methods,
    get_supported_formats,
    list_all_methods,
)


class TestQuantizationFactory:
    """Test quantization factory functions."""

    def test_create_quantizer_awq(self):
        """Test creating AWQ quantizer."""
        quantizer = create_quantizer("awq_q4_0")
        assert isinstance(quantizer, AwqQuantization)

        quantizer = create_quantizer("awq_q4_1")
        assert isinstance(quantizer, AwqQuantization)

        quantizer = create_quantizer("awq_q8_0")
        assert isinstance(quantizer, AwqQuantization)

        quantizer = create_quantizer("awq_f16")
        assert isinstance(quantizer, AwqQuantization)

    def test_create_quantizer_bnb(self):
        """Test creating BitsAndBytes quantizer."""
        quantizer = create_quantizer("bnb_4bit")
        assert isinstance(quantizer, BitsAndBytesQuantization)

        quantizer = create_quantizer("bnb_8bit")
        assert isinstance(quantizer, BitsAndBytesQuantization)

    def test_create_quantizer_fallback_search(self):
        """Test fallback quantizer search for methods supported by instances."""
        # This tests the fallback mechanism where we check each quantizer
        # to see if it supports the method
        
        # AWQ methods should still work through fallback
        quantizer = create_quantizer("awq_q4_0")
        assert isinstance(quantizer, AwqQuantization)
        
        # BnB methods should still work through fallback
        quantizer = create_quantizer("bnb_4bit")
        assert isinstance(quantizer, BitsAndBytesQuantization)

    def test_create_quantizer_unsupported_method(self):
        """Test creating quantizer with unsupported method."""
        with pytest.raises(ValueError, match="Unsupported quantization method: unsupported_method"):
            create_quantizer("unsupported_method")

    def test_get_available_methods(self):
        """Test getting available methods."""
        methods = get_available_methods()
        
        assert isinstance(methods, dict)
        assert "AWQ" in methods
        assert "BitsAndBytes" in methods
        
        # Check AWQ methods
        awq_methods = methods["AWQ"]
        assert "awq_q4_0" in awq_methods
        assert "awq_q4_1" in awq_methods
        assert "awq_q8_0" in awq_methods
        assert "awq_f16" in awq_methods
        
        # Check BnB methods
        bnb_methods = methods["BitsAndBytes"]
        assert "bnb_4bit" in bnb_methods
        assert "bnb_8bit" in bnb_methods

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = get_supported_formats()
        
        assert isinstance(formats, list)
        assert "pytorch" in formats
        assert "safetensors" in formats
        
        # Should be sorted
        assert formats == sorted(formats)

    def test_list_all_methods(self):
        """Test listing all methods as a flat list."""
        methods = list_all_methods()
        
        assert isinstance(methods, list)
        
        # Should contain all AWQ methods
        assert "awq_q4_0" in methods
        assert "awq_q4_1" in methods
        assert "awq_q8_0" in methods
        assert "awq_f16" in methods
        
        # Should contain all BnB methods
        assert "bnb_4bit" in methods
        assert "bnb_8bit" in methods
        
        # Should be sorted
        assert methods == sorted(methods)
        
        # Should not contain duplicates
        assert len(methods) == len(set(methods))

    def test_factory_functions_consistency(self):
        """Test that factory functions are consistent with each other."""
        # All methods from get_available_methods should be in list_all_methods
        available_methods = get_available_methods()
        all_methods = list_all_methods()
        
        for quantizer_type, methods in available_methods.items():
            for method in methods:
                assert method in all_methods, f"Method {method} from {quantizer_type} not in all_methods"
        
        # All methods in list_all_methods should be creatable
        for method in all_methods:
            # This should not raise an exception
            quantizer = create_quantizer(method)
            assert quantizer is not None
            
            # The quantizer should support the method
            assert quantizer.supports_method(method)

    def test_factory_returns_different_instances(self):
        """Test that factory returns different instances for each call."""
        quantizer1 = create_quantizer("awq_q4_0")
        quantizer2 = create_quantizer("awq_q4_0")
        
        # Should be different instances
        assert quantizer1 is not quantizer2
        
        # But same type
        assert type(quantizer1) == type(quantizer2)

    def test_error_message_includes_available_methods(self):
        """Test that error message includes available methods information."""
        with pytest.raises(ValueError) as exc_info:
            create_quantizer("invalid_method")
        
        error_message = str(exc_info.value)
        assert "Unsupported quantization method: invalid_method" in error_message
        assert "Available methods:" in error_message