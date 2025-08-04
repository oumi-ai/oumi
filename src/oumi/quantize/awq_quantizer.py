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

"""AWQ (Activation-aware Weight Quantization) quantizer implementation."""

import importlib
import importlib.util

import torch
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization, QuantizationResult
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger

# AWQ configuration defaults
AWQ_DEFAULTS = {
    "calibration_dataset": "pileval",
    "calibration_split": "train",
    "calibration_text_column": "text",
    "max_calibration_seq_len": 512,
    "duo_scaling": True,
    "apply_clip": True,
    "n_parallel_calib_samples": None,
}


class AwqQuantization(BaseQuantization):
    """AWQ (Activation-aware Weight Quantization) implementation.

    This class handles AWQ quantization with support for simulation mode
    when AWQ libraries are not available.
    """

    supported_methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
    supported_formats = ["safetensors"]

    def __init__(self):
        """Initialize AWQ quantizer."""
        if importlib.util.find_spec("awq") is not None:
            self._awq = importlib.import_module("awq")
        else:
            self._awq = None

    @override
    def raise_if_requirements_not_met(self):
        """Check if AWQ dependencies are available."""
        if self._awq is None:
            raise RuntimeError(
                "AWQ quantization requires autoawq library.\n"
                "Install with: `pip install oumi[quantization]`\n"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "AWQ quantization requires a GPU. "
                "Please use a machine with at least 1 GPU."
            )

    @override
    def quantize(self, config: QuantizationConfig) -> QuantizationResult:
        """Main quantization method for AWQ.

        Args:
            config: Quantization configuration

        Returns:
            Dictionary containing quantization results
        """
        self.validate_config(config)
        logger.info("Starting AWQ quantization pipeline...")

        # Step 1: AWQ quantization
        model, tokenizer = self._quantize(config)

        # Step 2: Save as PyTorch format
        logger.info("PyTorch format requested. Saving AWQ model...")

        model.save_quantized(config.output_path)
        tokenizer.save_pretrained(config.output_path)

        awq_size = get_directory_size(config.output_path)

        logger.info("✅ AWQ quantization successful! Saved as PyTorch format.")
        logger.info(f"📊 Quantized size: {format_size(awq_size)}")
        logger.info(
            f"💡 Use this model with: "
            f"AutoAWQForCausalLM.from_quantized('{config.output_path}')"
        )
        quantization_result = QuantizationResult(
            quantization_method=config.method,
            quantized_size_bytes=awq_size,
            output_path=config.output_path,
            format_type=config.output_format,
        )

        return quantization_result

    def _quantize(self, config: QuantizationConfig):
        """Quantize model using AWQ algorithm with calibration."""
        from transformers import AutoTokenizer

        logger.info(f"Loading model for AWQ quantization: {config.model.model_name}")

        # 1. Load model and tokenizer
        logger.info("📥 Loading base model...")

        model_kwargs = {
            "safetensors": True,
            "trust_remote_code": config.model.trust_remote_code,
            **(config.model.model_kwargs or {}),
        }

        model = self._awq.AutoAWQForCausalLM.from_pretrained(  # type: ignore
            config.model.model_name, **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
            **(config.model.tokenizer_kwargs or {}),
        )

        logger.info("🔧 Configuring AWQ quantization parameters...")

        # 2. Prepare quantization config
        w_bit_dict = {
            "awq_q4_0": 4,
            "awq_q4_1": 4,
            "awq_q8_0": 8,
            "awq_f16": 16,
        }
        w_bit = w_bit_dict[config.method]
        quant_config = {
            "zero_point": config.awq_zero_point,
            "q_group_size": config.awq_group_size,
            "w_bit": w_bit,
            "version": config.awq_version,
        }

        logger.info(f"⚙️  AWQ config: {quant_config}")
        logger.info(f"📊 Using {config.calibration_samples} calibration samples")
        logger.info("🧮 Starting AWQ calibration and quantization...")

        # 3. Perform AWQ quantization with calibration
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=AWQ_DEFAULTS["calibration_dataset"],
            split=AWQ_DEFAULTS["calibration_split"],
            text_column=AWQ_DEFAULTS["calibration_text_column"],
            max_calib_samples=config.calibration_samples,
            max_calib_seq_len=AWQ_DEFAULTS["max_calibration_seq_len"],
            duo_scaling=AWQ_DEFAULTS["duo_scaling"],
            apply_clip=AWQ_DEFAULTS["apply_clip"],
            n_parallel_calib_samples=AWQ_DEFAULTS["n_parallel_calib_samples"],
        )

        return model, tokenizer
