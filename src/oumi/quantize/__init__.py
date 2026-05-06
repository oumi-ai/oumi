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

Provides model quantization via LLM Compressor (FP8, GPTQ, AWQ) and
BitsAndBytes (NF4, FP4, INT8). Each backend's full business logic lives
in its own file (``llmcompressor.py``, ``bnb.py``).

To add a new backend:
  1. Create a new file under ``src/oumi/quantize/`` with a subclass of
     :class:`BaseQuantization` declaring its ``schemes`` map.
  2. Add it to ``_BACKENDS`` below.
  3. Add any new scheme/algorithm names to the enums in
     ``oumi.core.configs.quantization_config``.
"""

from typing import cast

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import QuantizationScheme
from oumi.exceptions import OumiConfigError
from oumi.quantize.base import BaseQuantization, QuantizationResult, SchemeSpec
from oumi.quantize.bnb import BitsAndBytesQuantization
from oumi.quantize.llmcompressor import LLMCompressorQuantization

# Adding a new backend = create file + add import above + append here.
_BACKENDS: list[type[BaseQuantization]] = [
    LLMCompressorQuantization,
    BitsAndBytesQuantization,
]


def _build_scheme_index() -> dict[QuantizationScheme, type[BaseQuantization]]:
    """Build the scheme→backend index, validating uniqueness at import time."""
    index: dict[QuantizationScheme, type[BaseQuantization]] = {}
    for cls in _BACKENDS:
        for scheme in cls.schemes:
            if scheme in index:
                raise RuntimeError(
                    f"Scheme {scheme.value!r} declared by both "
                    f"{index[scheme].__name__} and {cls.__name__}."
                )
            index[scheme] = cls
    return index


_SCHEME_TO_BACKEND: dict[QuantizationScheme, type[BaseQuantization]] = (
    _build_scheme_index()
)


def backend_for_scheme(scheme: QuantizationScheme) -> type[BaseQuantization]:
    """Return the backend class that owns ``scheme``."""
    try:
        return _SCHEME_TO_BACKEND[scheme]
    except KeyError:
        raise OumiConfigError(
            f"No backend registered for scheme {scheme.value!r}."
        ) from None


def all_schemes() -> dict[
    QuantizationScheme, tuple[type[BaseQuantization], SchemeSpec]
]:
    """Merged view of every backend's schemes. For ``--list-schemes``."""
    return {
        scheme: (cls, cls.schemes[scheme])
        for scheme, cls in _SCHEME_TO_BACKEND.items()
    }


def quantize(config: QuantizationConfig) -> QuantizationResult:
    """Quantize a model. Backend is dispatched from ``config.scheme``.

    Args:
        config: Quantization configuration. The backend is inferred from
            ``config.scheme``.

    Returns:
        :class:`QuantizationResult` with output path, backend, scheme,
        format, and size.

    Raises:
        OumiConfigError: If configuration is invalid.
        RuntimeError: If backend dependencies or hardware are missing,
            or if quantization itself fails.
    """
    if not isinstance(config, QuantizationConfig):
        raise ValueError(f"Expected QuantizationConfig, got {type(config)}")

    backend = backend_for_scheme(cast(QuantizationScheme, config.scheme))()
    backend.raise_if_requirements_not_met()
    return backend.quantize(config)


__all__ = [
    "BaseQuantization",
    "BitsAndBytesQuantization",
    "LLMCompressorQuantization",
    "QuantizationResult",
    "QuantizationScheme",
    "SchemeSpec",
    "all_schemes",
    "backend_for_scheme",
    "quantize",
]
