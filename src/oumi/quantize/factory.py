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

"""Factory for creating quantization instances.

This module provides a factory pattern for creating appropriate quantization
instances based on the quantization method. It serves as a simplified alternative
to the main quantize() function for when you need direct access to quantizer instances.
"""

from oumi.quantize.base import BaseQuantization


def create_quantizer(method: str) -> BaseQuantization:
    """Create appropriate quantization instance based on method.

    Args:
        method: Quantization method name (e.g., "awq_q4_0", "bnb_4bit", "q4_0")

    Returns:
        Instance of appropriate quantization class

    Raises:
        ValueError: If method is not supported by any quantizer
    """
    # Import here to avoid circular imports
    from oumi.quantize.awq_quantizer import AwqQuantization
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.gguf_quantizer import GgufQuantization

    # Determine quantizer based on method prefix or name
    if method.startswith("awq_"):
        return AwqQuantization()
    elif method.startswith("bnb_"):
        return BitsAndBytesQuantization()
    elif method in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"]:
        return GgufQuantization()
    else:
        # Try all quantizers to find one that supports this method
        for quantizer_class in [
            AwqQuantization,
            BitsAndBytesQuantization,
            GgufQuantization,
        ]:
            instance = quantizer_class()
            if instance.supports_method(method):
                return instance

        available_methods = get_available_methods()
        raise ValueError(
            f"Unsupported quantization method: {method}. "
            f"Available methods: {list(available_methods.keys())}"
        )


def get_available_methods() -> dict[str, list[str]]:
    """Returns all available methods grouped by quantization type.

    Returns:
        Dictionary mapping quantizer names to their supported methods
    """
    # Import here to avoid circular imports
    from oumi.quantize.awq_quantizer import AwqQuantization
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.gguf_quantizer import GgufQuantization

    return {
        "AWQ": AwqQuantization.supported_methods,
        "BitsAndBytes": BitsAndBytesQuantization.supported_methods,
        "GGUF": GgufQuantization.supported_methods,
    }


def get_supported_formats() -> list[str]:
    """Returns all supported output formats.

    Returns:
        List of all supported output formats
    """
    # Import here to avoid circular imports
    from oumi.quantize.awq_quantizer import AwqQuantization
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
    from oumi.quantize.gguf_quantizer import GgufQuantization

    formats = set()
    formats.update(AwqQuantization.supported_formats)
    formats.update(BitsAndBytesQuantization.supported_formats)
    formats.update(GgufQuantization.supported_formats)

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
