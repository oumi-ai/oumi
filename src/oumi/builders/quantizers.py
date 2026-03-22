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
instances based on the quantization method. It serves as a simplified alternative
to the main quantize() function for when you need direct access to quantizer instances.
"""

from oumi.quantize.base import BaseQuantization
from oumi.quantize.constants import BNB_METHODS, QuantizationMethod


def build_quantizer(method: QuantizationMethod) -> BaseQuantization:
    """Create appropriate quantization instance based on method.

    Args:
        method: Quantization method to use.

    Returns:
        Instance of appropriate quantization class
    """
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization

    if method in BNB_METHODS:
        return BitsAndBytesQuantization()
    return LLMCompressorQuantization()


def get_available_methods() -> dict[str, list[str]]:
    """Returns all available methods grouped by quantization type.

    Returns:
        Dictionary mapping quantizer names to their supported methods
    """
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.llmcompressor_quantizer import LLMCompressorQuantization

    return {
        "LLMCompressor": LLMCompressorQuantization.supported_methods,
        "BitsAndBytes": BitsAndBytesQuantization.supported_methods,
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


def list_all_methods() -> list[str]:
    """List all available quantization methods.

    Returns:
        Sorted list of all available method names
    """
    methods = []
    for method_list in get_available_methods().values():
        methods.extend(method_list)
    return sorted(methods)
