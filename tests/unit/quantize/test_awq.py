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

    def test_validate_awq_requirements_function_exists(self):
        """Test that validate_awq_requirements method exists and is callable."""
        quantizer = AwqQuantization()
        assert callable(quantizer.validate_requirements)

    def test_validate_awq_requirements_returns_bool(self):
        """Test that method returns boolean."""
        quantizer = AwqQuantization()
        result = quantizer.validate_requirements()
        # Should return bool
        assert isinstance(result, bool)
