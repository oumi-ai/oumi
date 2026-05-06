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

"""Base quantization types and contract.

A backend is a subclass of :class:`BaseQuantization` that declares the schemes
it owns as a class-level ``schemes`` mapping. All backend-specific metadata
(allowed algorithms, default algorithm, calibration rules, min compute
capability, description) lives in the :class:`SchemeSpec` values of that map.

To add a new backend: create a file under ``src/oumi/quantize/``, subclass
``BaseQuantization``, then register it in ``oumi/quantize/__init__.py``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError


@dataclass(frozen=True)
class SchemeSpec:
    """Per-scheme metadata declared on a backend class.

    Attributes:
        default_algorithm: Algorithm used when ``QuantizationAlgorithm.AUTO``
            is requested.
        allowed_algorithms: Algorithms accepted for this scheme. ``AUTO`` is
            always implicitly allowed (it resolves to ``default_algorithm``).
        needs_calibration_default: Whether the scheme's default algorithm
            requires calibration data.
        calibration_required_for: Algorithms that always require calibration
            data, even when the scheme's default does not (e.g. user
            overrides FP8_DYNAMIC's RTN with GPTQ).
        min_compute_capability: Minimum NVIDIA GPU compute capability
            required for *inference* (not quantization). See
            https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
        description: Human-readable description for ``--list-schemes``.
    """

    default_algorithm: QuantizationAlgorithm
    allowed_algorithms: tuple[QuantizationAlgorithm, ...]
    needs_calibration_default: bool
    calibration_required_for: tuple[QuantizationAlgorithm, ...] = ()
    min_compute_capability: float = 0.0
    description: str = ""

    def resolve_algorithm(
        self, requested: QuantizationAlgorithm
    ) -> QuantizationAlgorithm:
        """Return the concrete algorithm to use, or raise if disallowed."""
        if requested is QuantizationAlgorithm.AUTO:
            return self.default_algorithm
        if requested not in self.allowed_algorithms:
            allowed = sorted(a.value for a in self.allowed_algorithms)
            raise OumiConfigError(
                f"Algorithm {requested.value!r} is not allowed for this scheme. "
                f"Use 'auto' or one of: {allowed}."
            )
        return requested

    def needs_calibration_for(self, algorithm: QuantizationAlgorithm) -> bool:
        """Whether the given algorithm requires calibration data."""
        return self.needs_calibration_default or algorithm in self.calibration_required_for


@dataclass
class QuantizationResult:
    """Result of quantization."""

    output_path: str
    """Path to the quantized model."""

    backend: QuantizationBackend
    """Quantization backend used."""

    scheme: QuantizationScheme
    """Quantization scheme used."""

    format_type: str
    """Output format of the quantized model."""

    quantized_size_bytes: int
    """Size of the quantized model in bytes."""


class BaseQuantization(ABC):
    """Self-contained quantization backend.

    Subclasses declare:
      * ``backend`` — the :class:`QuantizationBackend` they implement.
      * ``schemes`` — every scheme they support, with metadata.
      * ``output_format`` — defaults to ``"safetensors"``.

    and implement two methods (``raise_if_requirements_not_met`` and
    ``quantize``).
    """

    backend: ClassVar[QuantizationBackend]
    output_format: ClassVar[str] = "safetensors"
    schemes: ClassVar[dict[QuantizationScheme, SchemeSpec]]

    @classmethod
    def owns(cls, scheme: QuantizationScheme) -> bool:
        """Whether this backend implements the given scheme."""
        return scheme in cls.schemes

    @abstractmethod
    def raise_if_requirements_not_met(self) -> None:
        """Raise ``RuntimeError`` if dependencies / hardware are missing."""

    @abstractmethod
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Quantize a model end-to-end and return the result."""
