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

    This configuration class defines parameters for quantizing models to reduce their
    size and memory requirements while maintaining inference performance. Quantization
    converts model weights from higher precision (e.g., float32) to lower precision
    (e.g., int4, int8) formats.

    The quantization process supports multiple methods and output formats, allowing
    flexibility for different deployment scenarios and inference engines.

    Example:
        Basic quantization to GGUF format:

        >>> config = QuantizationConfig(
        ...     model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
        ...     method="q4_0",
        ...     output_path="llama2-7b-q4.gguf",
        ...     output_format="gguf"
        ... )

        Quantization with custom settings:

        >>> config = QuantizationConfig(
        ...     model=ModelParams(model_name="./my_local_model"),
        ...     method="q8_0",
        ...     output_path="./quantized/model.safetensors",
        ...     output_format="safetensors",
        ...     verbose=True
        ... )

    Attributes:
        model: Parameters for the model to be quantized.
        method: Quantization method to use.
        output_path: Path where the quantized model will be saved.
        output_format: Output format for the quantized model.
        batch_size: Batch size for quantization process.
        verbose: Enable verbose logging during quantization.
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model to be quantized.

    This should specify the model name or path, along with any additional
    parameters needed to load the model (e.g., tokenizer_name, model_kwargs).

    The model can be:
    - A HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
    - A local path to a model directory
    - An Oumi model registry identifier
    """

    method: str = "q4_0"
    """Quantization method to use.

    The quantization method determines the precision and algorithm used to
    compress the model weights. Different methods offer trade-offs between
    model size, inference speed, and accuracy.

    Supported methods:

    **GGUF-compatible methods (for llama.cpp):**
    - ``q4_0``: 4-bit quantization with block-wise scaling (default)
    - ``q4_1``: 4-bit quantization with improved accuracy via bias terms
    - ``q5_0``: 5-bit quantization for better quality than 4-bit
    - ``q5_1``: 5-bit quantization with bias terms for highest 5-bit quality
    - ``q8_0``: 8-bit quantization for minimal quality loss

    **Precision methods:**
    - ``f16``: 16-bit floating point (half precision)
    - ``f32``: 32-bit floating point (no quantization, format conversion only)

    **Recommendations:**
    - Use ``q4_0`` for general purpose with good compression (4x smaller)
    - Use ``q8_0`` for minimal quality loss (2x smaller)
    - Use ``f16`` for GPU inference with moderate compression
    """

    output_path: str = "quantized_model.gguf"
    """Path where the quantized model will be saved.

    The output path determines both the location and filename of the quantized model.
    The file extension should match the output_format:
    - For GGUF format: use .gguf extension
    - For safetensors format: use .safetensors extension or directory path
    - For PyTorch format: use .pt/.pth extension or directory path

    Examples:
    - "models/llama2-7b-q4.gguf"
    - "./quantized/model/"
    - "/tmp/quantized_model.safetensors"
    """

    output_format: str = "gguf"
    """Output format for the quantized model.

    The output format determines the serialization format and compatibility
    with different inference engines.

    Supported formats:

    - ``gguf``: GGUF format (default)
        - Compatible with llama.cpp and derivatives
        - Single-file format with metadata
        - Best for CPU inference and edge deployment
        - Supports all quantization methods

    - ``safetensors``: Safetensors format
        - Compatible with HuggingFace transformers
        - Safe tensor serialization format
        - Good for GPU inference
        - Supports BitsAndBytes quantization methods

    - ``pytorch``: PyTorch format
        - Native PyTorch serialization
        - Compatible with PyTorch inference
        - Supports torch quantization methods
        - Good for research and development
    """

    batch_size: Optional[int] = None
    """Batch size for quantization process.

    The batch size controls how many samples are processed simultaneously
    during quantization calibration (if applicable). A larger batch size
    can improve quantization quality but requires more memory.

    If not specified (None), the quantization process will use automatic
    batch sizing based on available memory and model size.

    Typical values:
    - Small models (< 1B params): 32-128
    - Medium models (1B-7B params): 8-32
    - Large models (> 7B params): 1-8
    """

    verbose: bool = False
    """Enable verbose logging during quantization.

    When enabled, provides detailed progress information including:
    - Model loading progress
    - Quantization method details
    - Layer-by-layer processing status
    - Memory usage information
    - Final compression statistics

    Useful for debugging and monitoring long-running quantization jobs.
    """
