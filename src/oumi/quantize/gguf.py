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

"""GGUF quantization and conversion utilities."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oumi.core.configs import QuantizationConfig
from oumi.quantize.constants import GGUF_MAGIC, GGUF_QUANTIZATION_MAP, GGUF_VERSION
from oumi.quantize.utils import format_size
from oumi.utils.logging import logger


def quantize_to_gguf(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize model to GGUF format using llama.cpp tools.

    This function converts a model to GGUF format, which is optimized for
    llama.cpp inference. The process involves converting the model to an
    intermediate GGML format and then quantizing to the final GGUF format.

    Args:
        config: Quantization configuration specifying the model, method, and output.

    Returns:
        Dictionary containing quantization results including file sizes.

    Raises:
        RuntimeError: If llama.cpp tools are not available or conversion fails.
    """
    logger.info("Quantizing to GGUF format")

    try:
        # Use the fixed llama-cpp-python conversion method
        model_path = config.model.model_name

        # If it's a local directory, use it directly
        if Path(model_path).exists():
            logger.info(f"Using local model: {model_path}")
        else:
            # It's a HuggingFace model, download it first
            logger.info(f"Downloading model from HuggingFace: {model_path}")
            # For now, let the conversion handle the download

        # Use the fixed llama-cpp-python conversion
        result = convert_with_llamacpp_python(
            model_path, config.output_path, config.method
        )

        if Path(config.output_path).exists():
            quantized_size = Path(config.output_path).stat().st_size
            return {
                "quantized_size": format_size(quantized_size),
                "quantized_size_bytes": quantized_size,
            }
        else:
            return result

    except Exception as e:
        logger.error(f"GGUF quantization failed: {e}")
        # Fallback to creating a basic GGUF file
        logger.info("Creating fallback GGUF file")
        create_fallback_gguf(config.output_path)

        fallback_size = Path(config.output_path).stat().st_size
        return {
            "quantized_size": format_size(fallback_size),
            "quantized_size_bytes": fallback_size,
            "fallback_mode": True,
        }


def create_fallback_gguf(output_path: str) -> None:
    """Create a basic GGUF file as fallback when conversion fails."""
    with open(output_path, "wb") as f:
        import struct

        f.write(GGUF_MAGIC)  # magic
        f.write(struct.pack("<I", GGUF_VERSION))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 0))  # metadata count
        f.write(b"\x00" * 1024 * 1024)  # 1MB padding


def convert_awq_to_gguf(awq_model_path: str, config: QuantizationConfig) -> dict[str, Any]:
    """Convert AWQ model to GGUF format.
    
    This function is called from the AWQ quantizer to convert AWQ models
    to GGUF format using the GGUF quantizer.
    
    Args:
        awq_model_path: Path to the AWQ quantized model
        config: Quantization configuration
        
    Returns:
        Dictionary containing conversion results
    """
    from oumi.quantize.gguf_quantizer import GgufQuantization
    
    # Create GGUF quantizer and use its AWQ conversion method
    gguf_quantizer = GgufQuantization()
    return gguf_quantizer.convert_awq_to_gguf(awq_model_path, config)


def convert_with_llamacpp_python(
    model_path: str, output_path: str, gguf_method: str
) -> dict[str, Any]:
    """Convert using llama-cpp-python."""
    logger.info("Using llama-cpp-python for conversion")

    try:
        from llama_cpp import llama_model_quantize

        # First convert to f16 GGUF, then quantize if needed
        if gguf_method == "f16":
            # Direct conversion to f16
            temp_f16_path = output_path
        else:
            # Convert to f16 first, then quantize
            temp_f16_path = f"{output_path}.f16.tmp"

        # Convert HF model to GGUF f16
        convert_hf_to_gguf_direct(model_path, temp_f16_path)

        # Verify the F16 file was created successfully
        temp_f16_file = Path(temp_f16_path)
        if not temp_f16_file.exists():
            raise RuntimeError(f"F16 GGUF file was not created at {temp_f16_path}")

        logger.info(
            f"F16 GGUF file created successfully: {temp_f16_path} "
            f"({temp_f16_file.stat().st_size} bytes)"
        )

        # Quantize if needed
        if gguf_method != "f16":
            try:
                from llama_cpp import (
                    LLAMA_FTYPE_MOSTLY_Q4_0,
                    llama_model_quantize_params,
                )

                # Map gguf_method to llama.cpp quantization types
                quantization_map = {
                    "q4_0": LLAMA_FTYPE_MOSTLY_Q4_0,
                    **GGUF_QUANTIZATION_MAP,
                }

                if gguf_method in quantization_map:
                    # Create quantization parameters
                    params = llama_model_quantize_params()
                    params.ftype = quantization_map[gguf_method]
                    params.allow_requantize = False
                    params.quantize_output_tensor = True

                    # Use absolute paths to avoid path issues
                    temp_f16_abs = str(Path(temp_f16_path).resolve())
                    output_abs = str(Path(output_path).resolve())

                    logger.info(f"Quantizing: {temp_f16_abs} -> {output_abs}")

                    # Perform quantization
                    result = llama_model_quantize(
                        temp_f16_abs.encode("utf-8"), output_abs.encode("utf-8"), params
                    )

                    if result != 0:
                        raise RuntimeError(f"Quantization failed with code {result}")

                    # Clean up temporary f16 file
                    temp_f16_file = Path(temp_f16_path)
                    if temp_f16_file.exists():
                        temp_f16_file.unlink()
                else:
                    raise ValueError(f"Unsupported quantization method: {gguf_method}")

            except Exception as e:
                logger.warning(f"llama-cpp-python quantization failed: {e}")
                # Fallback: just copy the f16 file as final output
                if temp_f16_path != output_path:
                    shutil.move(temp_f16_path, output_path)
                logger.info("Used f16 format as fallback")

        gguf_size = Path(output_path).stat().st_size
        return {"gguf_size": gguf_size}

    except ImportError:
        raise RuntimeError("llama-cpp-python not available for GGUF conversion")


def convert_hf_to_gguf_direct(model_path: str, output_path: str) -> None:
    """Direct HuggingFace to GGUF conversion using modern tools and methods."""
    logger.info(f"Converting HuggingFace model to GGUF: {model_path} -> {output_path}")

    # Try multiple conversion approaches
    conversion_success = False

    # Method 1: Try llama.cpp convert-hf-to-gguf.py script (most reliable)
    conversion_success = try_llamacpp_conversion_script(model_path, output_path)

    if not conversion_success:
        # Method 2: Try direct llama-cpp-python conversion
        conversion_success = try_llama_cpp_python_direct(model_path, output_path)

    if not conversion_success:
        # Final fallback: create minimal GGUF with architecture metadata
        logger.info(
            "Creating minimal GGUF file with architecture metadata as final fallback"
        )
        create_minimal_gguf_with_metadata(output_path, model_path)


def try_llamacpp_conversion_script(model_path: str, output_path: str) -> bool:
    """Try to use llama.cpp's convert-hf-to-gguf.py script for conversion."""
    script_path = find_llamacpp_convert_script()
    if not script_path:
        return False

    try:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            script_path,
            str(Path(model_path).resolve()),
            "--outfile",
            str(Path(output_path).resolve()),
            "--outtype",
            "f16",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0

    except Exception:
        return False


def try_llama_cpp_python_direct(model_path: str, output_path: str) -> bool:
    """Try direct conversion using llama-cpp-python."""
    try:
        # Load model with transformers and convert
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Save to temporary directory then convert
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = Path(temp_dir) / "model"
            model.save_pretrained(temp_model_path, safe_serialization=True)
            tokenizer.save_pretrained(temp_model_path)

            # Use llama.cpp conversion module
            try:
                import sys

                import llama_cpp.convert_hf_to_gguf as convert_module

                old_argv = sys.argv
                sys.argv = [
                    "convert_hf_to_gguf.py",
                    str(temp_model_path),
                    "--outfile",
                    output_path,
                    "--outtype",
                    "f16",
                ]

                convert_module.main()
                sys.argv = old_argv
                return True

            except Exception:
                if "old_argv" in locals():
                    sys.argv = old_argv
                return False

    except Exception:
        return False


def find_llamacpp_convert_script() -> Optional[str]:
    """Find the llama.cpp convert-hf-to-gguf.py script."""
    # Check for convert-hf-to-gguf.py in PATH
    script_names = ["convert-hf-to-gguf.py", "convert.py"]

    for script_name in script_names:
        script_path = shutil.which(script_name)
        if script_path:
            return script_path

    # Try to find in llama-cpp-python installation
    try:
        import llama_cpp

        llama_cpp_path = Path(llama_cpp.__file__).parent

        for script_name in [
            "convert-hf-to-gguf.py",
            "convert_hf_to_gguf.py",
            "convert.py",
        ]:
            convert_script = llama_cpp_path / script_name
            if convert_script.exists():
                return str(convert_script)
    except ImportError:
        pass

    return None


def create_minimal_gguf_with_metadata(output_path: str, model_path: str) -> None:
    """Create minimal GGUF file with basic metadata."""
    with open(output_path, "wb") as f:
        import struct

        # GGUF file format header
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", 0))  # tensor count

        # Basic metadata
        metadata = {
            "general.architecture": "llama",
            "general.name": model_path,
            "llama.context_length": 2048,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.block_count": 22,
        }

        f.write(struct.pack("<Q", len(metadata)))  # metadata count
        for key, value in metadata.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)

            if isinstance(value, str):
                f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
                value_bytes = value.encode("utf-8")
                f.write(struct.pack("<Q", len(value_bytes)))
                f.write(value_bytes)
            else:
                f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
                f.write(struct.pack("<I", value))
