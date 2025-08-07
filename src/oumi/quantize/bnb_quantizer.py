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

"""BitsAndBytes quantization implementation."""

import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger


class BitsAndBytesQuantization(BaseQuantization):
    """BitsAndBytes quantization implementation.

    This class handles quantization using the BitsAndBytes library,
    supporting both 4-bit and 8-bit quantization methods.
    """

    supported_methods = ["bnb_4bit", "bnb_8bit"]
    supported_formats = ["safetensors"]

    def __init__(self):
        """Initialize BitsAndBytes quantizer."""
        self._bitsandbytes = importlib.util.find_spec("bitsandbytes")

    @override
    def raise_if_requirements_not_met(self) -> None:
        """Check if BitsAndBytes dependencies are available.

        Raises:
            RuntimeError: If BitsAndBytes dependencies are not available.
        """
        if self._bitsandbytes is None:
            raise RuntimeError(
                "BitsAndBytes quantization requires bitsandbytes library.\n"
                "Install with: pip install bitsandbytes"
            )

        # Import to get version info
        try:
            import bitsandbytes  # type: ignore

            logger.info(f"BitsAndBytes library found: {bitsandbytes.__version__}")
        except (ImportError, AttributeError):
            logger.info("BitsAndBytes library found (version unknown)")

    @override
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Main quantization method for BitsAndBytes.

        Args:
            config: Quantization configuration

        Returns:
            QuantizationResult containing quantization results
        """
        # Validate configuration for this quantizer
        self.validate_config(config)

        # Check requirements
        self.raise_if_requirements_not_met()

        logger.info("Starting BitsAndBytes quantization pipeline...")

        # Perform quantization
        model, tokenizer = self._quantize_model(config)

        # Save model based on output format
        output_path = self._save_model(model, tokenizer, config)

        quantized_size = get_directory_size(output_path)

        logger.info("✅ BitsAndBytes quantization successful!")
        logger.info(f"📊 Quantized size: {format_size(quantized_size)}")
        logger.info(f"💡 Model saved to: {output_path}")

        return QuantizationResult(
            quantization_method=config.method,
            quantized_size_bytes=quantized_size,
            output_path=output_path,
            format_type=config.output_format,
        )

    def _quantize_model(self, config: QuantizationConfig):
        """Quantize model using BitsAndBytes."""
        logger.info(
            f"Loading model for BitsAndBytes quantization: {config.model.model_name}"
        )
        logger.info("📥 Loading base model...")

        # Configure quantization based on method
        quantization_config = self._get_quantization_config(config.method)

        logger.info(f"🔧 Using {config.method} quantization")

        # Load and quantize model
        torch_dtype = config.model.torch_dtype
        if torch_dtype == torch.float32:
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            quantization_config=quantization_config,
            device_map=config.model.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.model.trust_remote_code,
            **(config.model.model_kwargs or {}),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
            **(config.model.tokenizer_kwargs or {}),
        )

        return model, tokenizer

    def _get_quantization_config(self, method: str):
        """Get BitsAndBytes quantization config based on method."""
        from transformers import BitsAndBytesConfig

        if method == "bnb_4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif method == "bnb_8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise ValueError(f"Unsupported BitsAndBytes method: {method}")

    def _save_model(self, model, tokenizer, config: QuantizationConfig) -> str:
        """Save quantized model based on output format."""
        # Ensure output directory exists
        output_path = Path(config.output_path)
        if output_path.suffix:
            # If output_path has an extension, treat parent as directory
            output_dir = output_path.parent
        else:
            # If no extension, treat as directory
            output_dir = output_path

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save based on format
        logger.info(f"Saving quantized model to: {output_dir}")
        model.save_pretrained(
            str(output_dir),
            safe_serialization=True,  # use safetensors
        )
        tokenizer.save_pretrained(str(output_dir))

        return str(output_dir)
