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
        ...     method="llmc_W4A16_ASYM",
        ...     output_path="llama2-7b-w4a16"
        ... )
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Model to quantize. Supports HuggingFace IDs, local paths, or Oumi models."""

    method: str = "llmc_W4A16_ASYM"
    """Quantization method. llm_compressor methods (llmc_W4A16, llmc_W4A16_ASYM,
    llmc_W8A8_INT, llmc_W8A8_FP8, etc.) or BitsAndBytes (bnb_4bit, bnb_8bit)."""

    output_path: str = "quantized_model"
    """Output file path for the quantized model."""

    output_format: str = "safetensors"
    """Output format: 'safetensors'."""

    # llm_compressor configuration
    llmc_targets: list[str] = field(default_factory=lambda: ["Linear"])
    """Layer types to target for quantization. Default targets Linear layers."""

    llmc_ignore: list[str] = field(default_factory=lambda: ["lm_head"])
    """Layers to exclude from quantization. Default ignores lm_head."""

    llmc_smoothing_strength: float = 0.8
    """SmoothQuant smoothing strength for W8A8 methods. Range 0.0-1.0."""

    # Calibration configuration
    calibration_dataset: str = "open_platypus"
    """Calibration dataset from HuggingFace datasets library.
    Options: 'open_platypus', 'ultrachat-200k', 'wikitext', etc."""

    max_seq_length: int = 2048
    """Maximum sequence length for calibration tokenization."""

    calibration_samples: int = 512
    """Number of calibration samples. 512 (default), 128 (faster), 1024 (more accurate)."""

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
