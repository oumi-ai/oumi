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

import pytest

from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization


class TestBitsAndBytesQuantization:
    """Test cases for BitsAndBytes quantization."""

    def test_supported_methods(self):
        """Test that BNB quantizer has correct supported methods."""
        assert "bnb_4bit" in BitsAndBytesQuantization.supported_methods
        assert "bnb_8bit" in BitsAndBytesQuantization.supported_methods

    def test_requirements_not_met_missing_library(self):
        """Test error when bitsandbytes is not installed."""
        quantizer = BitsAndBytesQuantization()
        quantizer._bitsandbytes = None
        with pytest.raises(RuntimeError, match="requires bitsandbytes library"):
            quantizer.raise_if_requirements_not_met()
