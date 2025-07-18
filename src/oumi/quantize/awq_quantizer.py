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

from pathlib import Path
from typing import Any, Union

from oumi.core.configs import QuantizationConfig
from oumi.quantize.base import BaseQuantization
from oumi.quantize.constants import AWQ_DEFAULTS, MOCK_MODEL_SIZES
from oumi.quantize.utils import format_size, get_directory_size
from oumi.utils.logging import logger


class AwqQuantization(BaseQuantization):
    """AWQ (Activation-aware Weight Quantization) implementation.
    
    This class handles AWQ quantization with support for simulation mode
    and fallback to BitsAndBytes when AWQ is not available.
    """

    supported_methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
    supported_formats = ["gguf", "pytorch"]

    def __init__(self):
        self._awq_available = None
        self._fallback_mode = None
        self._simulation_mode = None

    def validate_requirements(self) -> Union[bool, str]:
        """Check if AWQ dependencies are available.
        
        Returns:
            True if all dependencies are available, False if simulation mode should be used.
            "bitsandbytes" if BitsAndBytes fallback is available.
        """
        if self._awq_available is not None:
            return self._awq_available

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

            self._awq_available = True
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
                self._awq_available = "bitsandbytes"
                return "bitsandbytes"

            except ImportError:
                logger.warning(
                    "AWQ quantization requires autoawq library or bitsandbytes fallback.\n"
                    "Install with: pip install autoawq (Linux/Windows with CUDA)\n"
                    "Or: pip install bitsandbytes (macOS/CPU fallback)\n"
                    "Running in simulation mode for testing..."
                )
                self._awq_available = False
                return False

    def quantize(self, config: QuantizationConfig) -> dict[str, Any]:
        """Main quantization method for AWQ.
        
        Args:
            config: Quantization configuration
            
        Returns:
            Dictionary containing quantization results
        """
        # Validate configuration for this quantizer
        self.validate_config(config)

        # Check requirements and determine mode
        requirements = self.validate_requirements()

        if requirements == "bitsandbytes":
            # Use BitsAndBytes fallback
            return self._quantize_with_fallback(config)
        elif requirements is False:
            # Use simulation mode
            return self._simulate_quantization(config)
        else:
            # Use real AWQ quantization
            return self._quantize_with_awq(config)

    def _quantize_with_awq(self, config: QuantizationConfig) -> dict[str, Any]:
        """Perform real AWQ quantization."""
        logger.info("Starting AWQ quantization pipeline...")

        # Step 1: AWQ quantization
        awq_result = self._quantize_model_with_awq(config)
        awq_model_path = awq_result["awq_model_path"]

        # Step 2: Handle output format
        if config.output_format == "pytorch":
            return self._save_as_pytorch(config, awq_model_path)
        elif config.output_format == "gguf":
            return self._convert_to_gguf(config, awq_model_path)
        else:
            raise ValueError(f"Unsupported output format: {config.output_format}")

    def _quantize_model_with_awq(self, config: QuantizationConfig) -> dict[str, Any]:
        """Quantize model using AWQ algorithm with calibration."""
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
            config.model.tokenizer_name or config.model.model_name,
            trust_remote_code=True
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

    def _save_as_pytorch(self, config: QuantizationConfig, awq_model_path: str) -> dict[str, Any]:
        """Save AWQ model as PyTorch format."""
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

    def _convert_to_gguf(self, config: QuantizationConfig, awq_model_path: str) -> dict[str, Any]:
        """Convert AWQ model to GGUF format."""
        from oumi.quantize.gguf_quantizer import GgufQuantization

        logger.info("Converting AWQ model to GGUF format...")

        try:
            # Use GGUF quantizer for conversion
            gguf_quantizer = GgufQuantization()
            result = gguf_quantizer.convert_awq_to_gguf(awq_model_path, config)

            # Clean up temporary AWQ files if requested
            if config.cleanup_temp:
                import shutil
                shutil.rmtree(awq_model_path)
                logger.info(f"Cleaned up temporary AWQ files: {awq_model_path}")

            return result

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            # Fall back to PyTorch format
            logger.info("Falling back to PyTorch format...")
            result = self._save_as_pytorch(config, awq_model_path)
            result["gguf_conversion_failed"] = True
            return result

    def _quantize_with_fallback(self, config: QuantizationConfig) -> dict[str, Any]:
        """Use BitsAndBytes fallback for AWQ quantization."""
        from oumi.quantize.bnb_quantizer import BitsAndBytesQuantization

        logger.info("Using BitsAndBytes fallback for AWQ quantization...")
        bnb_quantizer = BitsAndBytesQuantization()
        return bnb_quantizer.quantize_awq_fallback(config)

    def _simulate_quantization(self, config: QuantizationConfig) -> dict[str, Any]:
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
            mock_size = MOCK_MODEL_SIZES["7b"]
        elif "13b" in model_name_lower:
            mock_size = MOCK_MODEL_SIZES["13b"]
        elif "70b" in model_name_lower:
            mock_size = MOCK_MODEL_SIZES["70b"]
        else:
            mock_size = MOCK_MODEL_SIZES["default"]

        # Adjust size based on quantization method
        if config.method == "awq_q4_0":
            mock_size = int(mock_size * 0.25)  # 4x compression
        elif config.method == "awq_q8_0":
            mock_size = int(mock_size * 0.5)   # 2x compression
        elif config.method == "awq_f16":
            mock_size = int(mock_size * 0.6)   # 1.6x compression

        # Create mock file
        with open(output_path, "wb") as f:
            # Write mock data in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            remaining = mock_size
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                f.write(b"0" * write_size)
                remaining -= write_size

        logger.info("✅ SIMULATION: Created mock quantized file")
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"📊 Simulated size: {format_size(mock_size)}")
        logger.info("⚠️  This is a simulation. Install autoawq for real quantization.")

        return {
            "quantization_method": f"SIMULATED: AWQ → PyTorch ({config.method})",
            "quantized_size": format_size(mock_size),
            "quantized_size_bytes": mock_size,
            "output_path": str(output_path),
            "simulation_mode": True,
            "awq_dependencies_missing": True,
        }
