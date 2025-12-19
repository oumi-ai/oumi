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

"""llm_compressor quantization implementation."""

import importlib
import importlib.util
from typing import Any

import torch
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.constants import LLMC_METHOD_CONFIG
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger


class LlmCompressorQuantization(BaseQuantization):
    """llm_compressor quantization implementation.

    This class handles quantization using the llm_compressor library,
    supporting AWQ, GPTQ, and FP8 quantization methods with various schemes
    like W4A16, W4A16_ASYM, W8A8, and FP8.
    """

    supported_methods = [
        "llmc_W4A16",
        "llmc_W4A16_ASYM",
        "llmc_W8A16",
        "llmc_W8A8_INT",
        "llmc_W8A8_FP8",
        "llmc_FP8_BLOCK",
    ]
    supported_formats = ["safetensors"]

    def __init__(self):
        """Initialize llm_compressor quantizer."""
        self._llmcompressor = None
        if importlib.util.find_spec("llmcompressor") is not None:
            self._llmcompressor = importlib.import_module("llmcompressor")

    @override
    def raise_if_requirements_not_met(self) -> None:
        """Check if llm_compressor dependencies are available."""
        if self._llmcompressor is None:
            raise RuntimeError(
                "llm_compressor quantization requires llmcompressor library.\n"
                "Install with: `pip install oumi[quantization]`\n"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "llm_compressor quantization requires a GPU. "
                "Please use a machine with at least 1 GPU."
            )

    @override
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Main quantization method using llm_compressor.

        Args:
            config: Quantization configuration

        Returns:
            QuantizationResult containing quantization results
        """
        self.validate_config(config)
        logger.info("Starting llm_compressor quantization pipeline...")

        # Perform quantization using oneshot API
        self._quantize(config)

        # Get output size
        quantized_size = get_directory_size(config.output_path)

        logger.info("llm_compressor quantization successful!")
        logger.info(f"Quantized size: {format_size(quantized_size)}")
        logger.info(f"Model saved to: {config.output_path}")

        return QuantizationResult(
            quantization_method=config.method,
            quantized_size_bytes=quantized_size,
            output_path=config.output_path,
            format_type=config.output_format,
        )

    def _get_modifier(self, config: QuantizationConfig) -> Any:
        """Get the appropriate modifier based on method configuration."""
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

        method_config = LLMC_METHOD_CONFIG[config.method]
        modifier_type = method_config["modifier"]
        scheme = method_config["scheme"]

        modifier_kwargs = {
            "scheme": scheme,
            "targets": config.llmc_targets,
            "ignore": config.llmc_ignore,
        }

        if modifier_type == "AWQModifier":
            return AWQModifier(**modifier_kwargs)
        elif modifier_type == "GPTQModifier":
            return GPTQModifier(**modifier_kwargs)
        elif modifier_type == "QuantizationModifier":
            return QuantizationModifier(**modifier_kwargs)
        else:
            raise ValueError(f"Unknown modifier type: {modifier_type}")

    def _get_recipe(self, config: QuantizationConfig) -> list[Any]:
        """Build the quantization recipe based on method configuration."""
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        method_config = LLMC_METHOD_CONFIG[config.method]
        requires_smoothquant = method_config["requires_smoothquant"]

        recipe = []

        # Add SmoothQuant for W8A8 INT8 methods
        if requires_smoothquant:
            recipe.append(
                SmoothQuantModifier(smoothing_strength=config.llmc_smoothing_strength)
            )

        # Add the main quantization modifier
        recipe.append(self._get_modifier(config))

        return recipe

    def _quantize(self, config: QuantizationConfig) -> None:
        """Quantize model using llm_compressor oneshot API."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from llmcompressor import oneshot

        logger.info(f"Loading model for quantization: {config.model.model_name}")
        logger.info(f"Using method: {config.method}")

        # Load model and tokenizer
        logger.info("Loading base model...")
        model_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": config.model.trust_remote_code,
            **(config.model.model_kwargs or {}),
        }

        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name, **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
            **(config.model.tokenizer_kwargs or {}),
        )

        # Build the recipe
        recipe = self._get_recipe(config)
        logger.info(f"Using recipe: {recipe}")
        logger.info(f"Using {config.calibration_samples} calibration samples")
        logger.info(f"Calibration dataset: {config.calibration_dataset}")

        # Run oneshot quantization
        logger.info("Starting quantization with oneshot API...")
        oneshot(
            model=model,
            dataset=config.calibration_dataset,
            recipe=recipe,
            max_seq_length=config.max_seq_length,
            num_calibration_samples=config.calibration_samples,
        )

        # Save the quantized model
        logger.info(f"Saving quantized model to: {config.output_path}")
        model.save_pretrained(config.output_path, save_compressed=True)
        tokenizer.save_pretrained(config.output_path)
