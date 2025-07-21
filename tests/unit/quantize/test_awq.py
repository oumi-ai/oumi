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

"""Unit tests for AWQ quantization functionality."""

from oumi.quantize.awq_quantizer import AwqQuantization


class TestValidateAwqRequirements:
    """Test AWQ dependency validation."""

    def test_raise_if_requirements_not_met_function_exists(self):
        """Test that raise_if_requirements_not_met method exists and is callable."""
        quantizer = AwqQuantization()
        assert callable(quantizer.raise_if_requirements_not_met)

    def test_raise_if_requirements_not_met_with_missing_deps(self):
        """Test that method raises when dependencies are missing."""
        quantizer = AwqQuantization()
        # Force missing dependency
        quantizer._awq = None
        try:
            quantizer.raise_if_requirements_not_met()
            # Should not reach here if deps are missing
            assert False, "Expected RuntimeError for missing dependencies"
        except RuntimeError:
            # Expected behavior
            pass
