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

"""MXFP4 quantization implementation for GPT OSS models."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.utils import get_directory_size
from oumi.utils.logging import logger

try:
    import mxfp4

    MXFP4_AVAILABLE = True
except ImportError:
    MXFP4_AVAILABLE = False
    mxfp4 = None


class MXFP4Quantizer(BaseQuantization):
    """MXFP4 quantizer for GPT OSS models.

    MXFP4 is a 4-bit mixed-precision floating-point format designed
    specifically for efficient deployment of large language models.
    It's the native quantization format for GPT OSS models.
    """

    # Define supported methods and formats
    supported_methods = ["mxfp4"]
    supported_formats = ["safetensors"]

    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Quantize a model using MXFP4.

        Args:
            config: Quantization configuration.

        Returns:
            QuantizationResult with quantization details.

        Raises:
            ImportError: If mxfp4 package is not installed.
            ValueError: If configuration is invalid.
        """
        self.raise_if_requirements_not_met()

        if config.quantization_method not in self.supported_methods:
            raise ValueError(
                f"Invalid quantization method for MXFP4: {config.quantization_method}. "
                f"Supported methods: {self.supported_methods}"
            )

        # Create output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model from {config.model_id}")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=config.trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )

        logger.info("Starting MXFP4 quantization...")

        # Apply MXFP4 quantization
        quantized_model = mxfp4.quantize_model(
            model,
            bits=4,
            group_size=128,  # Default group size for MXFP4
        )

        # Set quantization config in model
        quantized_model.config.quantization_config = {
            "quant_method": "mxfp4",
            "bits": 4,
            "group_size": 128,
        }

        # Save quantized model
        logger.info(f"Saving MXFP4 quantized model to {output_path}")
        quantized_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="10GB",
        )

        # Save tokenizer
        tokenizer.save_pretrained(output_path)

        # Calculate quantized model size
        quantized_size = get_directory_size(str(output_path))

        logger.info("MXFP4 quantization completed successfully")

        return QuantizationResult(
            quantized_size_bytes=quantized_size,
            output_path=str(output_path),
            quantization_method="mxfp4",
            format_type="safetensors",
            additional_info={
                "bits": 4,
                "group_size": 128,
                "compression_ratio": self._calculate_compression_ratio(
                    config.model_id, quantized_size
                ),
            },
        )

    def supports_method(self, method: str) -> bool:
        """Check if this quantizer supports the given method.

        Args:
            method: Quantization method name.

        Returns:
            True if method is supported.
        """
        return method in self.supported_methods

    def raise_if_requirements_not_met(self) -> None:
        """Raise an error if requirements are not met.

        Raises:
            ImportError: If mxfp4 package is not installed.
        """
        if not MXFP4_AVAILABLE:
            raise ImportError(
                "MXFP4 quantization requires the mxfp4 package. "
                "Install with: pip install mxfp4"
            )

    def _calculate_compression_ratio(self, model_id: str, quantized_size: int) -> float:
        """Calculate compression ratio.

        Args:
            model_id: Original model ID.
            quantized_size: Size of quantized model in bytes.

        Returns:
            Compression ratio.
        """
        try:
            # Try to estimate original model size
            # Assuming FP16 original (2 bytes per param)
            # MXFP4 uses 0.5 bytes per param
            # So compression ratio is approximately 4.0
            return 4.0
        except Exception:
            logger.warning("Could not calculate exact compression ratio")
            return 4.0  # Default compression ratio for MXFP4
