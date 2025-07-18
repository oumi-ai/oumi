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
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger


class BitsAndBytesQuantization(BaseQuantization):
    """BitsAndBytes quantization implementation.

    This class handles quantization using the BitsAndBytes library,
    supporting both 4-bit and 8-bit quantization methods.
    """

    supported_methods = ["bnb_4bit", "bnb_8bit"]
    supported_formats = ["pytorch", "safetensors"]

    def __init__(self):
        """Initialize BitsAndBytes quantizer."""
        self._bitsandbytes = importlib.util.find_spec("bitsandbytes")

    @override
    def validate_requirements(self) -> bool:
        """Check if BitsAndBytes dependencies are available.

        Returns:
            True if all dependencies are available, False otherwise.
        """
        if importlib.util.find_spec("bitsandbytes") is None:
            logger.error(
                "BitsAndBytes quantization requires bitsandbytes library.\n"
                "Install with: pip install bitsandbytes"
            )
            return False

        # Import to get version info
        logger.info(f"BitsAndBytes library found: {self._bitsandbytes.version}")  # type: ignore
        return True

    @override
    def quantize(self, config: QuantizationConfig) -> dict[str, Any]:
        """Main quantization method for BitsAndBytes.

        Args:
            config: Quantization configuration

        Returns:
            Dictionary containing quantization results
        """
        # Validate configuration for this quantizer
        self.validate_config(config)

        # Check requirements
        if not self.validate_requirements():
            raise RuntimeError("BitsAndBytes requirements not met")

        # Route to appropriate method based on output format
        if config.output_format == "safetensors":
            return self._quantize_to_safetensors(config)
        elif config.output_format == "pytorch":
            return self._quantize_to_pytorch(config)
        else:
            raise ValueError(f"Unsupported output format: {config.output_format}")

    def _quantize_to_safetensors(self, config: QuantizationConfig) -> dict[str, Any]:
        """Quantize model to safetensors format using BitsAndBytes."""
        logger.info("Quantizing to safetensors format with BitsAndBytes")

        from transformers import BitsAndBytesConfig

        # Configure quantization based on method
        if config.method == "bnb_4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.method == "bnb_8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise ValueError(
                f"Method {config.method} not supported for safetensors format"
            )

        # Load and quantize model
        logger.info(f"Loading model: {config.model.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **(config.model.model_kwargs or {}),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=True,
        )

        # Ensure output directory exists
        output_path = Path(config.output_path)
        if output_path.suffix:
            # If output_path has an extension, treat parent as directory
            output_dir = output_path.parent
        else:
            # If no extension, treat as directory
            output_dir = output_path

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save quantized model
        logger.info(f"Saving quantized model to: {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        quantized_size = get_directory_size(str(output_dir))

        logger.info("✅ BitsAndBytes quantization successful!")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"📊 Quantized size: {format_size(quantized_size)}")

        return {
            "quantization_method": f"BitsAndBytes {config.method}",
            "quantized_size": format_size(quantized_size),
            "quantized_size_bytes": quantized_size,
            "output_path": str(output_dir),
            "safetensors_format": True,
        }

    def _quantize_to_pytorch(self, config: QuantizationConfig) -> dict[str, Any]:
        """Quantize model to PyTorch format using BitsAndBytes."""
        logger.info("Quantizing to PyTorch format with BitsAndBytes")

        from transformers import BitsAndBytesConfig

        # Configure quantization based on method
        if config.method == "bnb_4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.method == "bnb_8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise ValueError(f"Method {config.method} not supported for PyTorch format")

        # Load and quantize model
        logger.info(f"Loading model: {config.model.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **(config.model.model_kwargs or {}),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=True,
        )

        # Ensure output directory exists
        output_path = Path(config.output_path)
        if output_path.suffix:
            # If output_path has an extension, treat parent as directory
            output_dir = output_path.parent
        else:
            # If no extension, treat as directory
            output_dir = output_path

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save quantized model
        logger.info(f"Saving quantized model to: {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=False)
        tokenizer.save_pretrained(str(output_dir))

        quantized_size = get_directory_size(str(output_dir))

        logger.info("✅ BitsAndBytes quantization successful!")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"📊 Quantized size: {format_size(quantized_size)}")

        return {
            "quantization_method": f"BitsAndBytes {config.method}",
            "quantized_size": format_size(quantized_size),
            "quantized_size_bytes": quantized_size,
            "output_path": str(output_dir),
            "pytorch_format": True,
        }
