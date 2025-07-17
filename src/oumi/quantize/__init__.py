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

"""Quantization module for Oumi.

This module provides comprehensive model quantization capabilities including
AWQ, BitsAndBytes, and GGUF quantization methods.
"""

from oumi.quantize.main import quantize
from oumi.quantize.factory import QuantizationFactory
from oumi.quantize.base import BaseQuantization
from oumi.quantize.awq_quantizer import AwqQuantization
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
from oumi.quantize.gguf_quantizer import GgufQuantization

__all__ = [
    "quantize",
    "QuantizationFactory",
    "BaseQuantization",
    "AwqQuantization",
    "BitsAndBytesQuantization",
    "GgufQuantization",
]
