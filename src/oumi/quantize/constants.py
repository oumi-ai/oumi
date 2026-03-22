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

"""Constants and registry for quantization schemes.

The ``SCHEME_REGISTRY`` maps each :class:`QuantizationScheme` to its metadata.

**LLM Compressor schemes** are derived from:
  - Scheme definitions: https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
  - Scheme reference: https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/
  - Algorithm guide: https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-algo/

**BitsAndBytes schemes** are derived from:
  - BitsAndBytesConfig: https://huggingface.co/docs/transformers/main/en/main_classes/quantization#transformers.BitsAndBytesConfig
  - 4-bit (NF4/FP4): https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
  - LLM.int8(): https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear8bit
"""

from dataclasses import dataclass

from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)


@dataclass(frozen=True)
class SchemeInfo:
    """Metadata for a quantization scheme.

    Attributes:
        backend: Which quantization library implements this scheme.
        llmc_scheme: Scheme string passed to LLM Compressor modifiers
            (e.g., QuantizationModifier(scheme=...)). None for BnB schemes.
            Ref: https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/
        default_algorithm: Default compression algorithm for this scheme.
            Ref: https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-algo/
        needs_calibration: Whether the default algorithm requires calibration data.
        min_compute_capability: Minimum NVIDIA GPU compute capability for
            *inference* (not quantization). From the scheme compatibility table at:
            https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
        description: Human-readable description.
    """

    backend: QuantizationBackend
    llmc_scheme: str | None
    default_algorithm: QuantizationAlgorithm
    needs_calibration: bool
    min_compute_capability: float
    description: str


@dataclass(frozen=True)
class AlgorithmInfo:
    """Metadata for a quantization algorithm.

    Attributes:
        needs_calibration: Whether this algorithm requires calibration data.
            None means it depends on the scheme (used by AUTO).
        description: Human-readable description.
    """

    needs_calibration: bool | None
    description: str


SCHEME_REGISTRY: dict[QuantizationScheme, SchemeInfo] = {
    # --- LLM Compressor: data-free methods (no calibration needed) ---
    QuantizationScheme.FP8_DYNAMIC: SchemeInfo(
        backend=QuantizationBackend.LLM_COMPRESSOR,
        llmc_scheme="FP8_DYNAMIC",
        default_algorithm=QuantizationAlgorithm.RTN,
        needs_calibration=False,
        min_compute_capability=8.9,
        description="FP8 dynamic quantization (data-free, Hopper+)",
    ),
    QuantizationScheme.FP8_BLOCK: SchemeInfo(
        backend=QuantizationBackend.LLM_COMPRESSOR,
        llmc_scheme="FP8_BLOCK",
        default_algorithm=QuantizationAlgorithm.RTN,
        needs_calibration=False,
        min_compute_capability=8.9,
        description="FP8 block-wise quantization (data-free, Hopper+)",
    ),
    # --- LLM Compressor: calibration-based weight-only methods ---
    QuantizationScheme.W4A16: SchemeInfo(
        backend=QuantizationBackend.LLM_COMPRESSOR,
        llmc_scheme="W4A16",
        default_algorithm=QuantizationAlgorithm.GPTQ,
        needs_calibration=True,
        min_compute_capability=7.5,
        description="4-bit weight quantization via GPTQ (Turing+)",
    ),
    QuantizationScheme.W4A16_ASYM: SchemeInfo(
        backend=QuantizationBackend.LLM_COMPRESSOR,
        llmc_scheme="W4A16_ASYM",
        default_algorithm=QuantizationAlgorithm.AWQ,
        needs_calibration=True,
        min_compute_capability=7.5,
        description="4-bit asymmetric weight quantization via AWQ (Turing+)",
    ),
    QuantizationScheme.W8A16: SchemeInfo(
        backend=QuantizationBackend.LLM_COMPRESSOR,
        llmc_scheme="W8A16",
        default_algorithm=QuantizationAlgorithm.GPTQ,
        needs_calibration=True,
        min_compute_capability=7.5,
        description="8-bit weight quantization via GPTQ (Turing+)",
    ),
    # --- BitsAndBytes methods ---
    QuantizationScheme.BNB_NF4: SchemeInfo(
        backend=QuantizationBackend.BNB,
        llmc_scheme=None,
        default_algorithm=QuantizationAlgorithm.BNB,
        needs_calibration=False,
        min_compute_capability=7.0,
        description="BitsAndBytes NormalFloat 4-bit quantization",
    ),
    QuantizationScheme.BNB_FP4: SchemeInfo(
        backend=QuantizationBackend.BNB,
        llmc_scheme=None,
        default_algorithm=QuantizationAlgorithm.BNB,
        needs_calibration=False,
        min_compute_capability=7.0,
        description="BitsAndBytes FloatingPoint 4-bit quantization",
    ),
    QuantizationScheme.BNB_INT8: SchemeInfo(
        backend=QuantizationBackend.BNB,
        llmc_scheme=None,
        default_algorithm=QuantizationAlgorithm.BNB,
        needs_calibration=False,
        min_compute_capability=7.0,
        description="BitsAndBytes LLM.int8() quantization",
    ),
}

ALGORITHM_REGISTRY: dict[QuantizationAlgorithm, AlgorithmInfo] = {
    QuantizationAlgorithm.AUTO: AlgorithmInfo(
        needs_calibration=None,
        description=(
            "Automatic selection. Picks the best algorithm for the chosen scheme."
        ),
    ),
    QuantizationAlgorithm.RTN: AlgorithmInfo(
        needs_calibration=False,
        description=(
            "Round-To-Nearest. Fast, data-free weight quantization. "
            "Best for FP8 schemes where calibration adds no benefit."
        ),
    ),
    QuantizationAlgorithm.GPTQ: AlgorithmInfo(
        needs_calibration=True,
        description=(
            "GPTQ. Calibration-based weight quantization that minimizes "
            "layer-wise reconstruction error. Best for W4A16 / W8A16."
        ),
    ),
    QuantizationAlgorithm.AWQ: AlgorithmInfo(
        needs_calibration=True,
        description=(
            "Activation-aware Weight Quantization. Preserves salient "
            "weights using activation statistics. Best for W4A16_ASYM."
        ),
    ),
    QuantizationAlgorithm.BNB: AlgorithmInfo(
        needs_calibration=False,
        description=(
            "BitsAndBytes. On-the-fly quantization at load time. "
            "Used internally by BnB schemes (bnb_nf4, bnb_fp4, bnb_int8)."
        ),
    ),
}


def get_default_schemes_by_algorithm() -> dict[
    QuantizationAlgorithm, list[QuantizationScheme]
]:
    """Returns a mapping from each algorithm to the schemes it is the default for."""
    result: dict[QuantizationAlgorithm, list[QuantizationScheme]] = {}
    for scheme, info in SCHEME_REGISTRY.items():
        result.setdefault(info.default_algorithm, []).append(scheme)
    return result


LLMCOMPRESSOR_SCHEMES: list[QuantizationScheme] = [
    s
    for s, info in SCHEME_REGISTRY.items()
    if info.backend == QuantizationBackend.LLM_COMPRESSOR
]

BNB_SCHEMES: list[QuantizationScheme] = [
    s for s, info in SCHEME_REGISTRY.items() if info.backend == QuantizationBackend.BNB
]

SUPPORTED_OUTPUT_FORMATS = ["safetensors"]
