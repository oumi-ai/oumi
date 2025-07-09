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

"""Main quantization logic and orchestration."""

from typing import Any

from oumi.core.configs import QuantizationConfig
from oumi.utils.logging import logger

from .awq import (
    quantize_awq_to_pytorch,
    simulate_awq_quantization,
    validate_awq_requirements,
)
from .bitsandbytes import (
    quantize_awq_fallback_to_pytorch,
    quantize_to_pytorch,
    quantize_to_safetensors,
    quantize_with_bitsandbytes,
)
from .gguf import quantize_to_gguf
from .utils import (
    calculate_compression_ratio,
    get_model_size_info,
    validate_quantization_config,
)


def quantize(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize a model according to the provided configuration.

    This function performs model quantization to reduce model size and memory
    requirements. It supports multiple quantization methods and output formats,
    allowing flexibility for different deployment scenarios.

    Args:
        config: Configuration for quantization including model params, method,
            output path, and format.

    Returns:
        A dictionary containing quantization results including file sizes
        and compression ratios.

    Raises:
        ValueError: If the quantization method is not supported, output format
            is invalid, or the model path/identifier is not found.
        RuntimeError: If the quantization process fails due to insufficient
            memory, missing dependencies, or other runtime errors.
    """
    logger.info(f"Starting quantization of model: {config.model.model_name}")
    logger.info(f"Quantization method: {config.method}")
    logger.info(f"Output path: {config.output_path}")

    # Validate configuration
    validate_quantization_config(config)

    # Get original model size information
    size_info, original_size = get_model_size_info(config)
    result = size_info.copy()

    # Validate AWQ requirements if using AWQ methods
    awq_simulation_mode = False
    awq_fallback_mode = False
    if config.method.startswith("awq_"):
        awq_available = validate_awq_requirements()
        if awq_available == "bitsandbytes":
            awq_fallback_mode = True
            logger.info("Using BitsAndBytes fallback for AWQ quantization.")
        elif not awq_available:
            awq_simulation_mode = True
            logger.info("AWQ dependencies not available. Running in simulation mode.")

    try:
        # Route to appropriate quantization method
        if config.method.startswith("awq_"):
            if awq_simulation_mode:
                result.update(simulate_awq_quantization(config))
            elif awq_fallback_mode:
                result.update(quantize_awq_fallback_to_pytorch(config))
            else:
                result.update(quantize_awq_to_pytorch(config))
        elif config.method.startswith("bnb_"):
            result.update(quantize_with_bitsandbytes(config))
        elif config.output_format == "gguf":
            result.update(quantize_to_gguf(config))
        elif config.output_format == "safetensors":
            result.update(quantize_to_safetensors(config))
        elif config.output_format == "pytorch":
            result.update(quantize_to_pytorch(config))

        # Calculate compression ratio if we have both sizes
        if "quantized_size_bytes" in result:
            compression_ratio = calculate_compression_ratio(
                original_size, result["quantized_size_bytes"]
            )
            result["compression_ratio"] = compression_ratio

        logger.info("Quantization completed successfully!")
        return result

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise RuntimeError(f"Quantization failed: {e}") from e
