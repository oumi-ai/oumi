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

from pathlib import Path
from typing import Any

from oumi.core.configs import QuantizationConfig
from oumi.quantize.factory import QuantizationFactory
from oumi.quantize.utils import (
    calculate_compression_ratio,
    get_model_size_info,
    is_valid_hf_model_id,
)
from oumi.utils.logging import logger


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

    # Check if model path exists or is a valid model identifier
    model_path = config.model.model_name
    if not (Path(model_path).exists() or is_valid_hf_model_id(model_path)):
        raise ValueError(f"Model not found: {model_path}")

    # Get original model size information
    size_info, original_size = get_model_size_info(config)
    result = size_info.copy()

    # Get appropriate quantizer from factory
    quantizer = QuantizationFactory.create_quantizer(config.method)

    # Check if quantizer requirements are met
    quantizer.raise_if_requirements_not_met()

    # Perform quantization
    quantization_result = quantizer.quantize(config)
    result.update(quantization_result)

    # Calculate compression ratio if we have both sizes
    if "quantized_size_bytes" in result:
        compression_ratio = calculate_compression_ratio(
            original_size, result["quantized_size_bytes"]
        )
        result["compression_ratio"] = compression_ratio

    logger.info("Quantization completed successfully!")
    return result
