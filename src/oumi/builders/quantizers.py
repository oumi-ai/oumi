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


def build_quantizer(method: str) -> BaseQuantization:
    """Create appropriate quantization instance based on method.

    Args:
        method: Quantization method name (e.g., "awq_q4_0", "bnb_4bit")

    Returns:
        Instance of appropriate quantization class

    Raises:
        ValueError: If method is not supported by any quantizer
    """
    # Import here to avoid circular imports
    from oumi.quantize.awq_quantizer import AwqQuantization
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization

    # Determine quantizer based on method prefix
    if method.startswith("awq_"):
        return AwqQuantization()
    elif method.startswith("bnb_"):
        return BitsAndBytesQuantization()
    else:
        # Try all quantizers to find one that supports this method
        for quantizer_class in [AwqQuantization, BitsAndBytesQuantization]:
            instance = quantizer_class()
            if instance.supports_method(method):
                return instance

        available_methods = get_available_methods()
        raise ValueError(
            f"Unsupported quantization method: {method}. "
            f"Available methods: {available_methods}"
        )


def get_available_methods() -> dict[str, list[str]]:
    """Returns all available methods grouped by quantization type.

    Returns:
        Dictionary mapping quantizer names to their supported methods
    """
    # Import here to avoid circular imports
    from oumi.quantize.awq_quantizer import AwqQuantization
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization

    return {
        "AWQ": AwqQuantization.supported_methods,
        "BitsAndBytes": BitsAndBytesQuantization.supported_methods,
    }


def get_supported_formats() -> list[str]:
    """Returns all supported output formats.

    Returns:
        List of all supported output formats
    """
    # Import here to avoid circular imports
    from oumi.quantize.awq_quantizer import AwqQuantization
    from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization

    formats = set()
    formats.update(AwqQuantization.supported_formats)
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
