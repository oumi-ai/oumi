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

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oumi.core.configs import QuantizationConfig
from oumi.utils.logging import logger

from .utils import format_size, get_directory_size


def quantize_to_safetensors(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize model to safetensors format using transformers quantization."""
    logger.info("Quantizing to safetensors format")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise RuntimeError(
            "BitsAndBytesConfig not available. Please install bitsandbytes."
        )

    # Configure quantization based on method
    if config.method in ["q4_0", "q4_1"]:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif config.method in ["q8_0"]:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Method {config.method} not supported for safetensors format")

    # Load and quantize model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Save quantized model
    output_dir = Path(config.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    quantized_size = get_directory_size(str(output_dir))
    return {
        "quantized_size": format_size(quantized_size),
        "quantized_size_bytes": quantized_size,
    }


def quantize_to_pytorch(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize model to PyTorch format using torch quantization."""
    logger.info("Quantizing to PyTorch format")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.float16 if config.method == "f16" else torch.float32,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Apply quantization based on method
    if config.method in ["q8_0"]:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif config.method == "f16":
        model = model.half()
    # f32 is already the default

    # Save quantized model
    output_dir = Path(config.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    quantized_size = get_directory_size(str(output_dir))
    return {
        "quantized_size": format_size(quantized_size),
        "quantized_size_bytes": quantized_size,
    }


def quantize_with_bitsandbytes(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize model using BitsAndBytes library."""
    logger.info(f"Quantizing with BitsAndBytes: {config.method}")

    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

        logger.info(f"✓ BitsAndBytes available: {bnb.__version__}")
    except ImportError as e:
        raise RuntimeError(
            "BitsAndBytes is not installed. "
            "Please install it with 'pip install bitsandbytes'"
        ) from e

    # Configure BitsAndBytes
    if config.method == "bnb_4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        bits = 4
    elif config.method == "bnb_8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        bits = 8
    else:
        raise ValueError(f"Unsupported BitsAndBytes method: {config.method}")

    logger.info(
        f"Loading model with {bits}-bit quantization: {config.model.model_name}"
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create output directory
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save quantized model
    if config.output_format == "pytorch":
        # Save as PyTorch model
        output_dir = output_path.with_suffix("")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        quantized_size = get_directory_size(output_dir)
        logger.info("✅ BitsAndBytes quantization completed!")
        logger.info(f"📁 Quantized model saved to: {output_dir}")
        logger.info(f"📊 Quantized size: {format_size(quantized_size)}")

        return {
            "quantization_method": f"BitsAndBytes {bits}-bit",
            "quantized_size": format_size(quantized_size),
            "quantized_size_bytes": quantized_size,
            "output_path": str(output_dir),
        }
    else:
        # For other formats, save as PyTorch and note the limitation
        logger.warning(
            f"Output format '{config.output_format}' not directly supported with BitsAndBytes. Saving as PyTorch format."
        )
        output_dir = output_path.with_suffix("").with_suffix(".pytorch")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        quantized_size = get_directory_size(output_dir)
        logger.info("✅ BitsAndBytes quantization completed!")
        logger.info(f"📁 Quantized model saved to: {output_dir}")
        logger.info(f"📊 Quantized size: {format_size(quantized_size)}")

        return {
            "quantization_method": f"BitsAndBytes {bits}-bit",
            "quantized_size": format_size(quantized_size),
            "quantized_size_bytes": quantized_size,
            "output_path": str(output_dir),
        }


def quantize_awq_fallback_to_pytorch(config: QuantizationConfig) -> dict[str, Any]:
    """AWQ quantization using BitsAndBytes fallback (for macOS/systems without AutoAWQ)."""
    logger.info("🔧 AWQ FALLBACK MODE: Using BitsAndBytes for real quantization")
    logger.info(f"   Model: {config.model.model_name}")
    logger.info(f"   Method: {config.method} (mapped to BitsAndBytes)")
    logger.info(f"   Output: {config.output_path}")

    from transformers import BitsAndBytesConfig

    # Map AWQ method to BitsAndBytes configuration
    if config.method in ["awq_q4_0", "awq_q4_1"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
            bnb_4bit_use_double_quant=True,  # Nested quantization
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("   Using 4-bit BitsAndBytes quantization")
    elif config.method == "awq_q8_0":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("   Using 8-bit BitsAndBytes quantization")
    else:  # awq_f16
        bnb_config = None
        logger.info("   Using 16-bit precision (no quantization)")

    logger.info("📥 Loading model from HuggingFace...")

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if not bnb_config else None,
        device_map="auto",
        trust_remote_code=False,
        **(config.model.model_kwargs or {}),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name or config.model.model_name
    )

    logger.info("💾 Saving quantized model...")

    # Save as PyTorch format
    output_path = Path(config.output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    quantized_size = get_directory_size(str(output_dir))

    return {
        "quantization_method": f"BitsAndBytes ({config.method})",
        "quantized_size": format_size(quantized_size),
        "quantized_size_bytes": quantized_size,
        "output_path": str(output_dir),
        "fallback_mode": True,
        "backend": "BitsAndBytes",
    }
