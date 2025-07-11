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

"""AWQ (Activation-aware Weight Quantization) implementation."""

from pathlib import Path
from typing import Any, Union

from oumi.core.configs import QuantizationConfig
from oumi.quantize.constants import AWQ_DEFAULTS, CHUNK_SIZE, MOCK_MODEL_SIZES
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger


def validate_awq_requirements() -> Union[bool, str]:
    """Check if AWQ dependencies are available.

    Returns:
        True if all dependencies are available, False if simulation mode should be used.
        "bitsandbytes" if BitsAndBytes fallback is available.
    """
    try:
        import awq

        logger.info(f"AWQ library found: autoawq {awq.__version__}")

        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                logger.warning(
                    "CUDA not available. AWQ quantization may be slow on CPU."
                )
        except ImportError:
            raise RuntimeError("AWQ quantization requires PyTorch")

        return True

    except ImportError:
        # Check for BitsAndBytes fallback
        try:
            import bitsandbytes
            import torch

            logger.warning(
                "AutoAWQ not available, but BitsAndBytes found.\n"
                "Using BitsAndBytes quantization as fallback for AWQ methods.\n"
                f"BitsAndBytes version: {bitsandbytes.__version__}"
            )
            return "bitsandbytes"

        except ImportError:
            logger.warning(
                "AWQ quantization requires autoawq library or bitsandbytes fallback.\n"
                "Install with: pip install autoawq (Linux/Windows with CUDA)\n"
                "Or: pip install bitsandbytes (macOS/CPU fallback)\n"
                "Running in simulation mode for testing..."
            )
            return False


def quantize_with_awq(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize model using AWQ algorithm with calibration.

    AWQ (Activation-aware Weight Quantization) uses calibration data to identify
    important weights that should be preserved during quantization.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"Loading model for AWQ quantization: {config.model.model_name}")

    # 1. Load model and tokenizer
    logger.info("📥 Loading base model...")
    model = AutoAWQForCausalLM.from_pretrained(
        config.model.model_name,
        **{
            "safetensors": True,
            "trust_remote_code": True,
            **(config.model.model_kwargs or {}),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name or config.model.model_name, trust_remote_code=True
    )

    logger.info("🔧 Configuring AWQ quantization parameters...")

    # 2. Prepare quantization config
    quant_config = {
        "zero_point": config.awq_zero_point,
        "q_group_size": config.awq_group_size,
        "w_bit": 4,  # AWQ uses 4-bit quantization
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

    # 4. Save AWQ quantized model
    temp_awq_path = f"{config.output_path}_awq_temp"
    logger.info(f"Saving AWQ model to: {temp_awq_path}")

    model.save_quantized(temp_awq_path)
    tokenizer.save_pretrained(temp_awq_path)

    awq_size = get_directory_size(temp_awq_path)

    return {"awq_model_path": temp_awq_path, "awq_size": awq_size}


def quantize_awq_to_pytorch(config: QuantizationConfig) -> dict[str, Any]:
    """Complete AWQ quantization pipeline for PyTorch output."""
    logger.info("Starting AWQ quantization pipeline...")

    # Step 1: AWQ quantization
    awq_result = quantize_with_awq(config)
    awq_model_path = awq_result["awq_model_path"]

    # Step 2: Save as PyTorch format
    logger.info("PyTorch format requested. Saving AWQ model...")

    output_path = config.output_path
    if not output_path.endswith(".pytorch"):
        output_path = f"{output_path}.pytorch"

    # Move AWQ model to final output path
    if awq_model_path != output_path:
        if Path(output_path).exists():
            import shutil

            shutil.rmtree(output_path)
        import shutil

        shutil.move(awq_model_path, output_path)

    awq_size = get_directory_size(output_path)

    logger.info("✅ AWQ quantization successful! Saved as PyTorch format.")
    logger.info(f"📁 Output: {output_path}")
    logger.info(f"📊 Quantized size: {format_size(awq_size)}")
    logger.info(
        f"💡 Use this model with: AutoAWQForCausalLM.from_quantized('{output_path}')"
    )

    return {
        "quantization_method": "AWQ → PyTorch",
        "awq_size": format_size(awq_size),
        "quantized_size": format_size(awq_size),
        "quantized_size_bytes": awq_size,
        "output_path": output_path,
        "pytorch_format": True,
    }


def simulate_awq_quantization(config: QuantizationConfig) -> dict[str, Any]:
    """Simulate AWQ quantization when dependencies are not available."""
    logger.info("🔧 SIMULATION MODE: AWQ quantization simulation")
    logger.info(f"   Model: {config.model.model_name}")
    logger.info(f"   Method: {config.method}")
    logger.info(f"   Output: {config.output_path}")

    # Create a mock output file
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine mock file size based on model name
    model_name_lower = config.model.model_name.lower()
    if "small" in model_name_lower:
        mock_size = MOCK_MODEL_SIZES["small"]
    elif "7b" in model_name_lower:
        if config.method == "awq_q4_0":
            mock_size = MOCK_MODEL_SIZES["7b_q4"]
        elif config.method == "awq_q8_0":
            mock_size = MOCK_MODEL_SIZES["7b_q8"]
        else:
            mock_size = MOCK_MODEL_SIZES["7b_q4"]
    else:
        mock_size = MOCK_MODEL_SIZES["default"]

    # Create mock output file
    with open(output_path, "wb") as f:
        written = 0
        while written < mock_size:
            remaining = min(CHUNK_SIZE, mock_size - written)
            f.write(b"\x00" * remaining)
            written += remaining

    mock_file_size = output_path.stat().st_size

    logger.info("✅ SIMULATION: AWQ quantization completed successfully!")
    logger.info(f"📁 SIMULATION: Mock output created at: {config.output_path}")
    logger.info(f"📊 SIMULATION: Mock file size: {format_size(mock_file_size)}")

    return {
        "quantization_method": f"SIMULATED: AWQ → PyTorch ({config.method})",
        "quantized_size": format_size(mock_file_size),
        "quantized_size_bytes": mock_file_size,
        "output_path": config.output_path,
        "simulation_mode": True,
        "awq_dependencies_missing": True,
    }
