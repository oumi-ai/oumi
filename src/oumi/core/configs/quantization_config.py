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

from dataclasses import dataclass, field
from enum import Enum

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.model_params import ModelParams


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""

    FP8_DYNAMIC = "fp8_dynamic"
    FP8_BLOCK = "fp8_block"
    W4A16 = "w4a16"
    W4A16_ASYM = "w4a16_asym"
    W8A16 = "w8a16"
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"


class QuantizationAlgorithm(str, Enum):
    """Quantization algorithm selection.

    AUTO defers to the default algorithm defined in METHOD_REGISTRY for the
    chosen quantization method.
    """

    AUTO = "auto"
    RTN = "rtn"
    GPTQ = "gptq"
    AWQ = "awq"
    BNB = "bnb"


def _coerce_enum(value, enum_cls: type[Enum], label: str):
    """Coerce a plain string to an enum member, raising ValueError on mismatch.

    Accepts both enum values (e.g. 'fp8_dynamic') and enum names (e.g. 'FP8_DYNAMIC'),
    since OmegaConf serializes enum fields by name when saving to YAML.
    """
    if isinstance(value, str) and not isinstance(value, enum_cls):
        # Try by value first (e.g. 'fp8_dynamic'), then by name (e.g. 'FP8_DYNAMIC').
        try:
            return enum_cls(value)
        except ValueError:
            pass
        try:
            return enum_cls[value]
        except KeyError:
            pass
        raise ValueError(
            f"Unsupported {label}: {value}. "
            f"Must be one of: {[e.value for e in enum_cls]}."
        )
    return value


@dataclass
class QuantizationConfig(BaseConfig):
    """Configuration for model quantization via LLM Compressor or BitsAndBytes.

    Reduces model size by converting weights from higher precision to lower
    precision formats, optimized for deployment with vLLM.

    Example:
        >>> config = QuantizationConfig(
        ...     model=ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct"),
        ...     method=QuantizationMethod.FP8_DYNAMIC,
        ...     output_path="llama3-8b-fp8"
        ... )
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Model to quantize. Supports HuggingFace IDs or local paths."""

    method: str | None = None
    """Quantization method (required). LLM Compressor: fp8_dynamic, fp8_block,
    w4a16, w4a16_asym, w8a16. BitsAndBytes: bnb_4bit, bnb_8bit.
    Accepts enum values (fp8_dynamic) or names (FP8_DYNAMIC); coerced in __post_init__."""

    output_path: str = "quantized_model"
    """Output directory for the quantized model."""

    output_format: str = "safetensors"
    """Output format: 'safetensors'."""

    # --- Algorithm control ---

    algorithm: str = QuantizationAlgorithm.AUTO.value
    """Compression algorithm: 'auto', 'rtn', 'gptq', 'awq', 'bnb'.
    'auto' selects the best algorithm for the chosen method.
    Accepts enum values (auto) or names (AUTO); coerced in __post_init__."""

    verbose: bool = False
    """Enable verbose logging during quantization."""

    ignore_layers: list[str] = field(default_factory=lambda: ["lm_head"])
    """Layer name patterns to exclude from quantization (regex supported)."""

    # --- Calibration settings ---

    calibration_dataset: str = "HuggingFaceH4/ultrachat_200k"
    """HuggingFace dataset ID for calibration data."""

    calibration_split: str = "train_sft"
    """Dataset split to use for calibration."""

    calibration_samples: int = 512
    """Number of calibration samples. 512 (balanced), 128 (faster), 1024 (accurate)."""

    max_seq_length: int = 2048
    """Max sequence length for calibration tokenization."""

    # --- Algorithm-specific settings ---

    group_size: int = 128
    """Weight grouping size for GPTQ/AWQ (64/128/256)."""

    dampening_frac: float = 0.1
    """GPTQ dampening fraction."""

    # --- Output control ---

    save_compressed: bool = True
    """Save in compressed-tensors format for optimized vLLM serving."""

    def __post_init__(self):
        """Post-initialization validation."""
        from oumi.quantize.constants import SUPPORTED_OUTPUT_FORMATS

        if self.method is None:
            raise ValueError(
                "Quantization method is required. "
                f"Must be one of: {[m.value for m in QuantizationMethod]}."
            )
        self.method = _coerce_enum(
            self.method, QuantizationMethod, "quantization method"
        )
        self.algorithm = _coerce_enum(
            self.algorithm, QuantizationAlgorithm, "algorithm"
        )

        if self.output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output format: {self.output_format}. "
                f"Must be one of: {SUPPORTED_OUTPUT_FORMATS}."
            )
