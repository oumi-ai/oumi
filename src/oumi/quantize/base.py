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
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs import QuantizationConfig


@dataclass
class QuantizationResult:
    """Result of quantization."""

    quantized_size_bytes: int
    """Size of the quantized model in bytes."""

    output_path: str
    """Path to the quantized model."""

    quantization_method: str
    """Quantization method used."""

    format_type: str
    """Format type of the quantized model."""

    additional_info: dict[str, Any] = field(default_factory=dict)
    """Additional information about the quantization process."""


class BaseQuantization(ABC):
    """Abstract base class for all quantization methods.

    This class defines the common interface that all quantization implementations
    must follow, ensuring consistency across different quantization approaches.
    """

    # Subclasses should define these class attributes
    supported_methods: list[str] = []
    supported_formats: list[str] = []

    @abstractmethod
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Main quantization method - must be implemented by subclasses.

        Args:
            config: Quantization configuration containing model parameters,
                method, output path, and other settings.

        Returns:
            QuantizationResult containing:
            - quantized_size_bytes: Size of the quantized model in bytes
            - output_path: Path to the quantized model
            - quantization_method: Quantization method used
            - format_type: Format type of the quantized model
            - additional_info: Additional method-specific information

        Raises:
            RuntimeError: If quantization fails for any reason
            ValueError: If configuration is invalid for this quantizer
        """
        raise NotImplementedError("Subclasses must implement quantize method")

    @abstractmethod
    def raise_if_requirements_not_met(self) -> None:
        """Raise an error if the requirements are not met."""
        raise NotImplementedError(
            "Subclasses must implement raise_if_requirements_not_met method"
        )

