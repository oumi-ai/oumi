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

"""Constants and mappings for quantization methods."""

from dataclasses import dataclass

from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationMethod,
)


@dataclass(frozen=True)
class MethodInfo:
    """Metadata for a single quantization method."""

    scheme: str | None
    algorithm: QuantizationAlgorithm
    needs_calibration: bool
    min_compute_capability: float
    description: str


METHOD_REGISTRY: dict[QuantizationMethod, MethodInfo] = {
    # --- Data-free methods (no calibration needed) ---
    QuantizationMethod.FP8_DYNAMIC: MethodInfo(
        scheme="FP8_DYNAMIC",
        algorithm=QuantizationAlgorithm.RTN,
        needs_calibration=False,
        min_compute_capability=8.9,
        description="FP8 dynamic quantization (data-free, Hopper+)",
    ),
    QuantizationMethod.FP8_BLOCK: MethodInfo(
        scheme="FP8_BLOCK",
        algorithm=QuantizationAlgorithm.RTN,
        needs_calibration=False,
        min_compute_capability=8.9,
        description="FP8 block-wise quantization (data-free, Hopper+)",
    ),
    # --- Calibration-based weight-only methods ---
    QuantizationMethod.W4A16: MethodInfo(
        scheme="W4A16",
        algorithm=QuantizationAlgorithm.GPTQ,
        needs_calibration=True,
        min_compute_capability=7.5,
        description="4-bit weight quantization via GPTQ (Turing+)",
    ),
    QuantizationMethod.W4A16_ASYM: MethodInfo(
        scheme="W4A16_ASYM",
        algorithm=QuantizationAlgorithm.AWQ,
        needs_calibration=True,
        min_compute_capability=7.5,
        description="4-bit asymmetric weight quantization via AWQ (Turing+)",
    ),
    QuantizationMethod.W8A16: MethodInfo(
        scheme="W8A16",
        algorithm=QuantizationAlgorithm.GPTQ,
        needs_calibration=True,
        min_compute_capability=7.5,
        description="8-bit weight quantization via GPTQ (Turing+)",
    ),
    # --- BitsAndBytes methods ---
    QuantizationMethod.BNB_4BIT: MethodInfo(
        scheme=None,
        algorithm=QuantizationAlgorithm.BNB,
        needs_calibration=False,
        min_compute_capability=7.0,
        description="BitsAndBytes NF4 4-bit quantization",
    ),
    QuantizationMethod.BNB_8BIT: MethodInfo(
        scheme=None,
        algorithm=QuantizationAlgorithm.BNB,
        needs_calibration=False,
        min_compute_capability=7.0,
        description="BitsAndBytes INT8 quantization",
    ),
}

LLMCOMPRESSOR_METHODS: list[QuantizationMethod] = [
    m
    for m, info in METHOD_REGISTRY.items()
    if info.algorithm != QuantizationAlgorithm.BNB
]

BNB_METHODS: list[QuantizationMethod] = [
    m
    for m, info in METHOD_REGISTRY.items()
    if info.algorithm == QuantizationAlgorithm.BNB
]

SUPPORTED_OUTPUT_FORMATS = ["safetensors"]
