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

"""Tests for main quantize module and function."""

from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize import quantize
from oumi.quantize.awq_quantizer import AwqQuantization
from oumi.quantize.base import QuantizationResult
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization


class TestQuantizeFunction:
    """Test main quantize() function."""

    def test_quantize_invalid_config_type(self):
        """Test quantize function with invalid config type."""
        with pytest.raises(ValueError, match="Expected QuantizationConfig"):
            quantize("invalid_config")

        with pytest.raises(ValueError, match="Expected QuantizationConfig"):
            quantize({"method": "awq_q4_0"})

    @patch("oumi.quantize.create_quantizer")
    def test_quantize_awq_method(self, mock_create_quantizer):
        """Test quantize function with AWQ method."""
        # Setup mocks
        mock_quantizer = Mock(spec=AwqQuantization)
        mock_quantizer.raise_if_requirements_not_met = Mock()
        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=1024,
            output_path="/test/path",
            quantization_method="awq_q4_0",
            format_type="pytorch",
        )
        mock_create_quantizer.return_value = mock_quantizer

        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_format="pytorch",
            output_path="/test/output",
        )

        result = quantize(config)

        # Verify factory was called
        mock_create_quantizer.assert_called_once_with("awq_q4_0")
        
        # Verify requirements check was called
        mock_quantizer.raise_if_requirements_not_met.assert_called_once()
        
        # Verify quantization was performed
        mock_quantizer.quantize.assert_called_once_with(config)
        
        # Verify result
        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == "awq_q4_0"

    @patch("oumi.quantize.create_quantizer")
    def test_quantize_bnb_method(self, mock_create_quantizer):
        """Test quantize function with BnB method."""
        # Setup mocks
        mock_quantizer = Mock(spec=BitsAndBytesQuantization)
        mock_quantizer.raise_if_requirements_not_met = Mock()
        mock_quantizer.quantize.return_value = QuantizationResult(
            quantized_size_bytes=2048,
            output_path="/test/path2",
            quantization_method="bnb_4bit",
            format_type="safetensors",
        )
        mock_create_quantizer.return_value = mock_quantizer

        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_format="safetensors",
            output_path="/test/output",
        )

        result = quantize(config)

        # Verify factory was called
        mock_create_quantizer.assert_called_once_with("bnb_4bit")
        
        # Verify requirements check was called
        mock_quantizer.raise_if_requirements_not_met.assert_called_once()
        
        # Verify quantization was performed
        mock_quantizer.quantize.assert_called_once_with(config)
        
        # Verify result
        assert isinstance(result, QuantizationResult)
        assert result.quantization_method == "bnb_4bit"

    @patch("oumi.quantize.create_quantizer")
    def test_quantize_requirements_not_met(self, mock_create_quantizer):
        """Test quantize function when requirements are not met."""
        # Setup mocks
        mock_quantizer = Mock()
        mock_quantizer.raise_if_requirements_not_met.side_effect = RuntimeError("Requirements not met")
        mock_create_quantizer.return_value = mock_quantizer

        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_format="pytorch",
            output_path="/test/output",
        )

        # Should propagate the RuntimeError
        with pytest.raises(RuntimeError, match="Requirements not met"):
            quantize(config)

        # Verify quantize was not called
        mock_quantizer.quantize.assert_not_called()

    @patch("oumi.quantize.create_quantizer")
    def test_quantize_quantization_fails(self, mock_create_quantizer):
        """Test quantize function when quantization fails."""
        # Setup mocks
        mock_quantizer = Mock()
        mock_quantizer.raise_if_requirements_not_met = Mock()
        mock_quantizer.quantize.side_effect = RuntimeError("Quantization failed")
        mock_create_quantizer.return_value = mock_quantizer

        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="bnb_4bit",
            output_format="pytorch",
            output_path="/test/output",
        )

        # Should propagate the RuntimeError
        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize(config)

        # Verify requirements check was called
        mock_quantizer.raise_if_requirements_not_met.assert_called_once()
        
        # Verify quantize was called before failing
        mock_quantizer.quantize.assert_called_once_with(config)

    @patch("oumi.quantize.create_quantizer")
    def test_quantize_unsupported_method(self, mock_create_quantizer):
        """Test quantize function with unsupported method."""
        mock_create_quantizer.side_effect = ValueError("Unsupported quantization method")

        # Use a valid method for config creation, but mock factory to fail
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",  # Valid method for config creation
            output_format="pytorch",
            output_path="/test/output",
        )

        # Should propagate the ValueError from factory
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            quantize(config)

    @patch("oumi.quantize.create_quantizer")
    def test_quantize_integration_flow(self, mock_create_quantizer):
        """Test the complete integration flow without mocking quantizers."""
        # Mock the factory to avoid dependency issues
        mock_quantizer = Mock()
        mock_quantizer.raise_if_requirements_not_met.side_effect = RuntimeError("autoawq library not found")
        mock_create_quantizer.return_value = mock_quantizer
        
        config = QuantizationConfig(
            model=ModelParams(model_name="test/model"),
            method="awq_q4_0",
            output_format="pytorch",
            output_path="/test/output",
        )

        # This should work up to the requirements check
        with pytest.raises(RuntimeError, match="autoawq library not found"):
            quantize(config)
            
        # Verify the flow was followed
        mock_create_quantizer.assert_called_once_with("awq_q4_0")
        mock_quantizer.raise_if_requirements_not_met.assert_called_once()


class TestQuantizeModuleImports:
    """Test that the quantize module imports work correctly."""

    def test_main_imports_available(self):
        """Test that main quantize module imports are available."""
        from oumi.quantize import (
            AwqQuantization,
            BaseQuantization,
            BitsAndBytesQuantization,
            QuantizationResult,
            create_quantizer,
            quantize,
        )

        # Basic type checks
        assert AwqQuantization is not None
        assert BaseQuantization is not None
        assert BitsAndBytesQuantization is not None
        assert QuantizationResult is not None
        assert create_quantizer is not None
        assert quantize is not None

    def test_quantize_module_all_exports(self):
        """Test that __all__ exports are correct."""
        import oumi.quantize as quantize_module

        expected_exports = [
            "BaseQuantization",
            "QuantizationResult",
            "AwqQuantization",
            "BitsAndBytesQuantization",
            "create_quantizer",
            "quantize",
        ]

        assert hasattr(quantize_module, "__all__")
        assert set(quantize_module.__all__) == set(expected_exports)

        # Verify all exports are actually available
        for export in expected_exports:
            assert hasattr(quantize_module, export), f"Export {export} not available"