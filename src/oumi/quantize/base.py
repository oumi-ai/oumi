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

"""Base quantization class and common utilities."""

from abc import ABC, abstractmethod
from typing import Any, List

from oumi.core.configs import QuantizationConfig


class BaseQuantization(ABC):
    """Abstract base class for all quantization methods.
    
    This class defines the common interface that all quantization implementations
    must follow, ensuring consistency across different quantization approaches.
    """

    # Subclasses should define these class attributes
    supported_methods: List[str] = []
    supported_formats: List[str] = []

    @abstractmethod
    def quantize(self, config: QuantizationConfig) -> dict[str, Any]:
        """Main quantization method - must be implemented by subclasses.
        
        Args:
            config: Quantization configuration containing model parameters,
                method, output path, and other settings.
                
        Returns:
            Dictionary containing quantization results including:
            - quantized_size: Human-readable size string
            - quantized_size_bytes: Size in bytes
            - output_path: Path to the quantized model
            - quantization_method: Method used for quantization
            - Additional method-specific results
            
        Raises:
            RuntimeError: If quantization fails for any reason
            ValueError: If configuration is invalid for this quantizer
        """
        raise NotImplementedError("Subclasses must implement quantize method")

    @abstractmethod
    def validate_requirements(self) -> bool:
        """Check if all required dependencies are available.
        
        Returns:
            True if all dependencies are available and quantization can proceed,
            False otherwise.
        """
        raise NotImplementedError("Subclasses must implement validate_requirements method")

    def get_supported_methods(self) -> List[str]:
        """Return list of quantization methods supported by this quantizer.
        
        Returns:
            List of method names (e.g., ["awq_q4_0", "awq_q8_0"])
        """
        return self.supported_methods.copy()

    def get_supported_formats(self) -> List[str]:
        """Return list of output formats supported by this quantizer.
        
        Returns:
            List of format names (e.g., ["gguf", "pytorch"])
        """
        return self.supported_formats.copy()

    def supports_method(self, method: str) -> bool:
        """Check if this quantizer supports the given method.
        
        Args:
            method: Quantization method name to check
            
        Returns:
            True if method is supported, False otherwise
        """
        return method in self.supported_methods

    def supports_format(self, format_name: str) -> bool:
        """Check if this quantizer supports the given output format.
        
        Args:
            format_name: Output format name to check
            
        Returns:
            True if format is supported, False otherwise
        """
        return format_name in self.supported_formats

    def validate_config(self, config: QuantizationConfig) -> None:
        """Validate configuration for this quantizer.
        
        Args:
            config: Quantization configuration to validate
            
        Raises:
            ValueError: If configuration is invalid for this quantizer
        """
        if not self.supports_method(config.method):
            raise ValueError(
                f"Method '{config.method}' not supported by {self.__class__.__name__}. "
                f"Supported methods: {self.supported_methods}"
            )

        if not self.supports_format(config.output_format):
            raise ValueError(
                f"Format '{config.output_format}' not supported by {self.__class__.__name__}. "
                f"Supported formats: {self.supported_formats}"
            )
