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

"""Builder for creating quantization instances.

This module provides a builder pattern for creating appropriate quantization
instances based on the quantization backend. It serves as a simplified alternative
to the main quantize() function for when you need direct access to quantizer instances.
"""

from oumi.core.configs.quantization_config import (
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.quantize.base import BaseQuantization


def build_quantizer(backend: QuantizationBackend) -> BaseQuantization:
    """Create appropriate quantization instance based on backend.

    Args:
        backend: Quantization backend to use.

    Returns:
        Instance of appropriate quantization class
    """
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization

    if backend == QuantizationBackend.BNB:
        return BitsAndBytesQuantization()
    return LLMCompressorQuantization()


def get_available_schemes() -> dict[str, list[QuantizationScheme]]:
    """Returns all available schemes grouped by quantization backend.

    Returns:
        Dictionary mapping backend names to their supported schemes
    """
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization

    return {
        "LLMCompressor": LLMCompressorQuantization.supported_schemes,
        "BitsAndBytes": BitsAndBytesQuantization.supported_schemes,
    }


def get_supported_formats() -> list[str]:
    """Returns all supported output formats.

    Returns:
        List of all supported output formats
    """
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization

    formats = set()
    formats.update(LLMCompressorQuantization.supported_formats)
    formats.update(BitsAndBytesQuantization.supported_formats)

    return sorted(list(formats))


def list_all_schemes() -> list[QuantizationScheme]:
    """List all available quantization schemes.

    Returns:
        Sorted list of all available scheme enum members
    """
    schemes = []
    for scheme_list in get_available_schemes().values():
        schemes.extend(scheme_list)
    return sorted(schemes)
