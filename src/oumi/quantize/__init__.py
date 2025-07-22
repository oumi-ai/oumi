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
from oumi.core.configs import QuantizationConfig


def quantize(config: "QuantizationConfig") -> QuantizationResult:
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
    # Map quantization methods to their respective quantizers
    quantizer_map = {
        "awq_q4_0": AwqQuantization,
        "awq_q4_1": AwqQuantization,
        "awq_q8_0": AwqQuantization,
        "awq_f16": AwqQuantization,
    }
    
    # Find the appropriate quantizer for the method
    quantizer_class = quantizer_map.get(config.method)
    if quantizer_class is None:
        available_methods = list(quantizer_map.keys())
        raise ValueError(
            f"Unsupported quantization method '{config.method}'. "
            f"Available methods: {available_methods}"
        )
    
    # Initialize and run quantization
    quantizer = quantizer_class()
    quantizer.raise_if_requirements_not_met()
    
    return quantizer.quantize(config)

from oumi.quantize.base import BaseQuantization

__all__ = [
    "BaseQuantization",
    "AwqQuantization",
]
