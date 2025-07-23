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

from oumi.quantize.awq_quantizer import AwqQuantization
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
from oumi.quantize.factory import create_quantizer
from oumi.quantize.gguf_quantizer import GgufQuantization


def quantize(config) -> QuantizationResult:
    """Main quantization function that routes to appropriate quantizer.

    Args:
        config: Quantization configuration containing method, model parameters,
            and other settings.

    Returns:
        QuantizationResult containing quantization results including file sizes
        and compression ratios.

    Raises:
        ValueError: If quantization method is not supported
        RuntimeError: If quantization fails
    """
    from oumi.core.configs import QuantizationConfig

    if not isinstance(config, QuantizationConfig):
        raise ValueError(f"Expected QuantizationConfig, got {type(config)}")

    # Use factory to create appropriate quantizer
    quantizer = create_quantizer(config.method)
    quantizer.raise_if_requirements_not_met()

    return quantizer.quantize(config)


__all__ = [
    "BaseQuantization",
    "QuantizationResult",
    "AwqQuantization",
    "BitsAndBytesQuantization",
    "GgufQuantization",
    "create_quantizer",
    "quantize",
]
