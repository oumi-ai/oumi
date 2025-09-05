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
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.model_params import ModelParams


@dataclass
class QuantizationConfig(BaseConfig):
    """Configuration for model quantization.

    Reduces model size by converting weights from higher precision (float32) to
    lower precision (int4, int8) formats while maintaining performance.

    Tested on NVIDIA H100 GPU with models up to 14B parameters.

    Example:
        >>> config = QuantizationConfig(
        ...     model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
        ...     method="awq_q4_0",
        ...     output_path="llama2-7b-q4.gguf"
        ... )
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Model to quantize. Supports HuggingFace IDs, local paths, or Oumi models."""

    method: str = "awq_q4_0"
    """Quantization method. AWQ methods (awq_q4_0, awq_q8_0) provide best quality.
    Direct GGUF methods (q4_0, q8_0) for llama.cpp. Precision methods (f16, f32)."""

    output_path: str = "quantized_model"
    """Output file path for the quantized model."""

    output_format: str = "safetensors"
    """Output format: 'safetensors'."""

    batch_size: Optional[int] = None
    """Batch size for calibration. Auto-sized if None. Typical: 32, 8-32, 1-8."""

    verbose: bool = False
    """Enable detailed progress logging."""

    # AWQ-specific configuration
    awq_group_size: int = 128
    """AWQ weight grouping size. 128 (balanced), 64 (higher accuracy), 256 (faster)."""

    awq_zero_point: bool = True
    """Enable zero-point quantization for AWQ. Generally recommended."""

    awq_version: str = "GEMM"
    """AWQ kernel version. 'GEMM' (faster, default) or 'GEMV'."""

    cleanup_temp: bool = True
    """Remove temporary AWQ files after conversion."""

    calibration_samples: int = 512
    """AWQ calibration samples. 512 (balanced), 128 (faster), 1024 (more accurate)."""

    def __post_init__(self):
        """Post-initialization validation."""
        from oumi.quantize.constants import SUPPORTED_METHODS, SUPPORTED_OUTPUT_FORMATS

        # Validate output format
        if self.output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output format: {self.output_format}. "
                f"Must be one of: {SUPPORTED_OUTPUT_FORMATS}."
            )

        # Validate quantization method
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported quantization method: {self.method}. "
                f"Must be one of: {SUPPORTED_METHODS}."
            )
