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

"""GGUF quantization implementation."""

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization
from oumi.quantize.constants import GGUF_QUANTIZATION_MAP
from oumi.quantize.utils import format_size
from oumi.utils.logging import logger


class GgufQuantization(BaseQuantization):
    """GGUF quantization implementation using llama.cpp tools.

    This class handles direct quantization to GGUF format, which is optimized
    for llama.cpp inference. It supports various quantization levels from
    4-bit to 16-bit precision.
    """

    supported_methods = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"]
    supported_formats = ["gguf"]

    def __init__(self):
        """Initialize GGUF quantizer."""
        self._llama_cpp = importlib.util.find_spec("llama_cpp")

    @override
    def raise_if_requirements_not_met(self) -> None:
        """Check if GGUF quantization dependencies are available.

        Raises:
            RuntimeError: If llama-cpp-python is not available.
        """
        if self._llama_cpp is None:
            raise RuntimeError(
                "GGUF quantization requires llama-cpp-python.\n"
                "Install with: pip install llama-cpp-python"
            )

        try:
            import llama_cpp

            logger.info(f"llama-cpp-python found: {llama_cpp.__version__}")
        except (ImportError, AttributeError):
            logger.info("llama-cpp-python found (version unknown)")

    @override
    def quantize(self, config: QuantizationConfig) -> dict[str, Any]:
        """Main quantization method for GGUF.

        Args:
            config: Quantization configuration

        Returns:
            Dictionary containing quantization results
        """
        # Validate configuration for this quantizer
        self.validate_config(config)

        logger.info("Quantizing to GGUF format")

        # Use llama-cpp-python for quantization
        return self._quantize_with_llamacpp(config)

    def _quantize_with_llamacpp(self, config: QuantizationConfig) -> dict[str, Any]:
        """Quantize using llama-cpp-python."""
        try:
            model_path = config.model.model_name

            # If it's a local directory, use it directly
            if Path(model_path).exists():
                logger.info(f"Using local model: {model_path}")
            else:
                # It's a HuggingFace model, download it first
                logger.info(f"Downloading model from HuggingFace: {model_path}")
                # For now, let the conversion handle the download

            # Use the fixed llama-cpp-python conversion
            result = self._convert_with_llamacpp_python(
                model_path, config.output_path, config.method
            )

            if Path(config.output_path).exists():
                quantized_size = Path(config.output_path).stat().st_size

                logger.info("✅ GGUF quantization successful!")
                logger.info(f"📁 Output: {config.output_path}")
                logger.info(f"📊 Quantized size: {format_size(quantized_size)}")

                return {
                    "quantization_method": f"GGUF {config.method}",
                    "quantized_size": format_size(quantized_size),
                    "quantized_size_bytes": quantized_size,
                    "output_path": config.output_path,
                    "gguf_format": True,
                }
            else:
                return result

        except ImportError:
            raise RuntimeError("llama-cpp-python not available for GGUF quantization")

    def _convert_with_llamacpp_python(
        self, model_path: str, output_path: str, method: str
    ) -> dict[str, Any]:
        """Convert model to GGUF using llama-cpp-python."""
        # Create temporary directory for conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load and convert model
            logger.info(f"Loading model for GGUF conversion: {model_path}")

            # First, load the model with transformers and save in a format llama.cpp
            # can use
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            # Save model in compatible format
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # Map quantization method to llama.cpp format
            if method in GGUF_QUANTIZATION_MAP:
                quantization_type = GGUF_QUANTIZATION_MAP[method]
            else:
                raise ValueError(f"Unsupported quantization method: {method}")

            # Perform quantization using llama.cpp
            try:
                # This is a simplified version - in practice, you'd use the actual
                # llama.cpp quantization tools or python bindings
                self._quantize_with_llamacpp_binary(
                    temp_dir, output_path, quantization_type
                )

            except Exception as e:
                logger.error(f"llama.cpp quantization failed: {e}")
                raise e

        return {"status": "success"}

    def _quantize_with_llamacpp_binary(
        self, model_dir: str, output_path: str, quantization_type: str
    ) -> None:
        """Use llama.cpp binary tools for quantization."""
        logger.info(f"Quantizing with llama.cpp: {quantization_type}")

        # This is where the actual llama.cpp quantization would happen
        # For now, raise an error since we don't have a real implementation
        raise NotImplementedError(
            "GGUF quantization is not yet fully implemented. "
            "This would require actual llama.cpp integration."
        )
