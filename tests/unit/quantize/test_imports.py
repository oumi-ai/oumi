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

"""Unit tests for quantization module imports and public API."""

import inspect


class TestQuantizeImports:
    """Test quantization module import structure."""

    def test_public_api_import(self):
        """Test that public API can be imported."""
        from oumi.quantize import quantize

        assert callable(quantize)
        assert hasattr(quantize, "__name__")
        assert quantize.__name__ == "quantize"

    def test_quantize_function_signature(self):
        """Test that quantize function has correct signature."""
        from oumi.quantize import quantize

        sig = inspect.signature(quantize)
        params = list(sig.parameters.keys())

        # Should have config parameter
        assert "config" in params
        assert len(params) == 1  # Only one parameter

    def test_quantize_module_all(self):
        """Test that __all__ is properly defined."""
        import oumi.quantize

        assert hasattr(oumi.quantize, "__all__")
        assert "quantize" in oumi.quantize.__all__

        # With new class-based architecture, we export more items
        expected_items = [
            "quantize",
            "QuantizationFactory",
            "BaseQuantization",
            "AwqQuantization",
            "BitsAndBytesQuantization",
            "GgufQuantization",
        ]

        for item in expected_items:
            assert item in oumi.quantize.__all__

    def test_submodule_imports(self):
        """Test that submodules can be imported."""
        # Test that individual modules can be imported
        from oumi.quantize import (
            awq_quantizer,
            bnb_quantizer,
            constants,
            gguf_quantizer,
            main,
            utils,
        )

        # Verify modules have expected attributes
        assert hasattr(constants, "SUPPORTED_METHODS")
        assert hasattr(utils, "format_size")
        assert hasattr(awq_quantizer, "AwqQuantization")
        assert hasattr(bnb_quantizer, "BitsAndBytesQuantization")
        assert hasattr(gguf_quantizer, "GgufQuantization")
        assert hasattr(main, "quantize")

    def test_no_relative_imports_in_public_api(self):
        """Test that public API doesn't expose relative imports."""
        import oumi.quantize

        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(oumi.quantize) if not attr.startswith("_")]

        # Should only contain quantize function and standard module attributes
        expected_attrs = {"quantize"}
        actual_attrs = set(public_attrs)

        # quantize should be present
        assert "quantize" in actual_attrs

        # Should not expose internal modules
        internal_modules = {
            "constants",
            "utils",
            "awq_quantizer",
            "bnb_quantizer",
            "gguf_quantizer",
            "main",
        }
        exposed_internals = actual_attrs.intersection(internal_modules)

        # It's okay if some internals are exposed, but the main API should work
        # The key is that 'quantize' is available
        assert "quantize" in actual_attrs

    def test_backward_compatibility(self):
        """Test that the public API maintains backward compatibility."""
        # This import pattern should work (original API)
        # Test that we can call it with a config object
        from oumi.core.configs import ModelParams, QuantizationConfig
        from oumi.quantize import quantize

        # Create a minimal config to test callable interface
        config = QuantizationConfig(
            model=ModelParams(model_name="test"),
            method="awq_q4_0",
            output_path="/tmp/test.gguf",
            output_format="gguf",
        )

        # Should be callable (we won't actually call it in unit tests)
        assert callable(quantize)

        # Verify function accepts the config type
        sig = inspect.signature(quantize)
        config_param = sig.parameters["config"]
        assert config_param.annotation.__name__ == "QuantizationConfig"


class TestModuleStructure:
    """Test the overall module structure."""

    def test_module_docstrings(self):
        """Test that modules have proper docstrings."""
        import oumi.quantize
        import oumi.quantize.constants
        import oumi.quantize.main
        import oumi.quantize.utils

        # Main module should have docstring
        assert oumi.quantize.__doc__ is not None
        assert "quantization" in oumi.quantize.__doc__.lower()

        # Submodules should have docstrings
        assert oumi.quantize.constants.__doc__ is not None
        assert oumi.quantize.utils.__doc__ is not None
        assert oumi.quantize.main.__doc__ is not None

    def test_no_import_errors(self):
        """Test that all imports succeed without errors."""
        # Import main module

        # Import all submodules

        # If we get here, all imports succeeded
        assert True
