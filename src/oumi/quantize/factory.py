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

"""Factory for creating quantization instances."""

from typing import Dict, List, Type

from oumi.quantize.base import BaseQuantization


class QuantizationFactory:
    """Factory to create appropriate quantization class based on method.

    This factory handles the creation of quantization instances and provides
    discovery of available quantization methods and formats.
    """

    # Registry will be populated as quantizer classes are registered
    _quantizers: dict[str, type[BaseQuantization]] = {}

    @classmethod
    def register_quantizer(
        cls, prefix: str, quantizer_class: type[BaseQuantization]
    ) -> None:
        """Register a quantizer class with a method prefix.

        Args:
            prefix: Method prefix (e.g., "awq", "bnb", "gguf")
            quantizer_class: Quantizer class to register
        """
        cls._quantizers[prefix] = quantizer_class

    @classmethod
    def create_quantizer(cls, method: str) -> BaseQuantization:
        """Create appropriate quantization class based on method.

        Args:
            method: Quantization method name (e.g., "awq_q4_0", "bnb_4bit")

        Returns:
            Instance of appropriate quantization class

        Raises:
            ValueError: If method is not supported by any registered quantizer
        """
        # Import here to avoid circular imports
        from oumi.quantize.awq_quantizer import AwqQuantization
        from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
        from oumi.quantize.gguf_quantizer import GgufQuantization

        # Auto-register quantizers if not already registered
        if not cls._quantizers:
            cls.register_quantizer("awq", AwqQuantization)
            cls.register_quantizer("bnb", BitsAndBytesQuantization)
            cls.register_quantizer("gguf", GgufQuantization)

        # Determine quantizer based on method
        if method.startswith("awq_"):
            return cls._quantizers["awq"]()
        elif method.startswith("bnb_"):
            return cls._quantizers["bnb"]()
        elif method in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"]:
            return cls._quantizers["gguf"]()
        else:
            # Try to find a quantizer that supports this method
            for quantizer_class in cls._quantizers.values():
                instance = quantizer_class()
                if instance.supports_method(method):
                    return instance

            raise ValueError(
                f"Unsupported quantization method: {method}. "
                f"Available methods: {cls.get_available_methods()}"
            )

    @classmethod
    def get_available_methods(cls) -> dict[str, list[str]]:
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

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Returns all supported output formats.

        Returns:
            List of all supported output formats
        """
        formats = set()

        # Import here to avoid circular imports
        from oumi.quantize.awq_quantizer import AwqQuantization
        from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization
        from oumi.quantize.gguf_quantizer import GgufQuantization

        formats.update(AwqQuantization.supported_formats)
        formats.update(BitsAndBytesQuantization.supported_formats)
        formats.update(GgufQuantization.supported_formats)

        return sorted(list(formats))

    @classmethod
    def get_quantizer_for_method(cls, method: str) -> type[BaseQuantization]:
        """Get the quantizer class that supports the given method.

        Args:
            method: Quantization method name

        Returns:
            Quantizer class that supports the method

        Raises:
            ValueError: If method is not supported
        """
        quantizer_instance = cls.create_quantizer(method)
        return quantizer_instance.__class__

    @classmethod
    def list_all_methods(cls) -> list[str]:
        """List all available quantization methods.

        Returns:
            Sorted list of all available method names
        """
        methods = []
        for method_list in cls.get_available_methods().values():
            methods.extend(method_list)
        return sorted(methods)
