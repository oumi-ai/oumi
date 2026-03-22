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
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.constants import BNB_METHODS, QuantizationMethod
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger


class BitsAndBytesQuantization(BaseQuantization):
    """BitsAndBytes quantization implementation.

    This class handles quantization using the BitsAndBytes library,
    supporting both 4-bit and 8-bit quantization methods.
    """

    supported_methods = BNB_METHODS
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
        self.validate_config(config)

        logger.info("Starting BitsAndBytes quantization pipeline...")

        model, tokenizer = self._quantize_model(config)

        output_dir = Path(config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving quantized model to: {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        quantized_size = get_directory_size(str(output_dir))

        logger.info("BitsAndBytes quantization successful!")
        logger.info(f"Quantized size: {format_size(quantized_size)}")
        logger.info(f"Model saved to: {output_dir}")

        method = cast(QuantizationMethod, config.method)
        return QuantizationResult(
            quantization_method=method,
            quantized_size_bytes=quantized_size,
            output_path=str(output_dir),
            format_type=config.output_format,
        )

    def _quantize_model(self, config: QuantizationConfig):
        """Quantize model using BitsAndBytes."""
        logger.info(
            f"Loading model for BitsAndBytes quantization: {config.model.model_name}"
        )

        method = cast(QuantizationMethod, config.method)
        quantization_config = self._get_quantization_config(method)
        logger.info(f"Using {method.value} quantization")

        model_kwargs = dict(config.model.model_kwargs or {})
        # Remove keys passed explicitly below to avoid duplicate keyword argument errors.
        model_kwargs.pop("device_map", None)
        model_kwargs.pop("torch_dtype", None)
        model_kwargs["trust_remote_code"] = config.model.trust_remote_code

        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            **model_kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
            **(config.model.tokenizer_kwargs or {}),
        )

        return model, tokenizer

    def _get_quantization_config(self, method: QuantizationMethod):
        """Get BitsAndBytes quantization config based on method."""
        from transformers import BitsAndBytesConfig

        if method == QuantizationMethod.BNB_4BIT:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif method == QuantizationMethod.BNB_8BIT:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise ValueError(f"Unsupported BitsAndBytes method: {method}")
