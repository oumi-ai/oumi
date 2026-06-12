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

This module provides model quantization via LLM Compressor (FP8, GPTQ, AWQ)
and BitsAndBytes (NF4, FP4, INT8).
"""

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import QuantizationScheme
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
from oumi.quantize.constants import QuantizationAlgorithm
from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization


def quantize(config: QuantizationConfig) -> QuantizationResult:
    """Main quantization function that routes to appropriate quantizer.

    Args:
        config: Quantization configuration containing scheme, model
            parameters, and other settings. The backend is inferred
            from the scheme.

    Returns:
        QuantizationResult containing quantization results including file sizes
        and compression ratios.

    Raises:
        ValueError: If quantization configuration is invalid
        RuntimeError: If quantization fails
    """
    if not isinstance(config, QuantizationConfig):
        raise ValueError(f"Expected QuantizationConfig, got {type(config)}")

    from oumi.builders.quantizers import build_quantizer

    quantizer = build_quantizer(config.backend)
    quantizer.raise_if_requirements_not_met()

    return quantizer.quantize(config)


__all__ = [
    "BaseQuantization",
    "BitsAndBytesQuantization",
    "LLMCompressorQuantization",
    "QuantizationAlgorithm",
    "QuantizationResult",
    "QuantizationScheme",
    "quantize",
]
