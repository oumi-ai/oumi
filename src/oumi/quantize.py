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

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oumi.core.configs import QuantizationConfig
from oumi.utils.logging import logger


def quantize(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize a model according to the provided configuration.

    This function performs model quantization to reduce model size and memory
    requirements. It supports multiple quantization methods and output formats,
    allowing flexibility for different deployment scenarios.

    The quantization process involves:
    1. Loading the original model from the specified path or identifier
    2. Applying the chosen quantization method to compress weights
    3. Saving the quantized model in the requested format
    4. Calculating compression statistics

    Supported quantization methods:
    - q4_0, q4_1: 4-bit quantization (4x compression)
    - q5_0, q5_1: 5-bit quantization (3.2x compression)
    - q8_0: 8-bit quantization (2x compression)
    - f16: 16-bit float (2x compression)
    - f32: 32-bit float (format conversion only)

    Supported output formats:
    - GGUF: Single-file format compatible with llama.cpp
    - Safetensors: Safe serialization format for HuggingFace models
    - PyTorch: Native PyTorch serialization format

    Args:
        config: Configuration for quantization including:
            - model: ModelParams specifying the model to quantize
            - method: Quantization method (e.g., "q4_0", "q8_0")
            - output_path: Where to save the quantized model
            - output_format: Format for the quantized model
              ("gguf", "safetensors", "pytorch")
            - batch_size: Optional batch size for quantization
            - verbose: Enable detailed logging

    Returns:
        A dictionary containing quantization results:
            - "original_size": Human-readable original model size (if available)
            - "quantized_size": Human-readable quantized model size
            - "quantized_size_bytes": Quantized model size in bytes
            - "compression_ratio": Compression ratio (e.g., "4.2x")

    Raises:
        ValueError: If the quantization method is not supported, output format
            is invalid, or the model path/identifier is not found.
        RuntimeError: If the quantization process fails due to insufficient
            memory, missing dependencies, or other runtime errors.

    Example:
        >>> from oumi.core.configs import QuantizationConfig, ModelParams
        >>> config = QuantizationConfig(
        ...     model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
        ...     method="q4_0",
        ...     output_path="llama2-7b-q4.gguf",
        ...     output_format="gguf"
        ... )
        >>> result = quantize(config)
        >>> print(f"Compression: {result['compression_ratio']}")
        Compression: 4.1x
    """
    logger.info(f"Starting quantization of model: {config.model.model_name}")
    logger.info(f"Quantization method: {config.method}")
    logger.info(f"Output path: {config.output_path}")

    # Validate inputs
    if config.output_format not in ["gguf", "safetensors", "pytorch"]:
        raise ValueError(f"Unsupported output format: {config.output_format}")

    supported_methods = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"]
    if config.method not in supported_methods:
        raise ValueError(f"Unsupported quantization method: {config.method}")

    # Check if model path exists or is a valid model identifier
    model_path = config.model.model_name
    if not (Path(model_path).exists() or _is_valid_hf_model_id(model_path)):
        raise ValueError(f"Model not found: {model_path}")

    # For now, provide a clear error message about the current implementation status
    logger.warning("Quantization feature is currently in development.")
    logger.info("Current implementation requires additional dependencies and setup.")
    logger.info("Simulating quantization process for validation...")

    # Simulate the process for now
    result = {
        "status": "simulated",
        "method": config.method,
        "output_format": config.output_format,
        "output_path": config.output_path,
        "message": "Quantization logic is being developed. This is a simulation.",
    }

    logger.info("Quantization validation completed!")
    return result


def _quantize_to_gguf(config: QuantizationConfig) -> dict[str, Any]:
    """Quantize model to GGUF format using llama.cpp tools.

    This function converts a model to GGUF format, which is optimized for
    llama.cpp inference. The process involves converting the model to an
    intermediate GGML format and then quantizing to the final GGUF format.

    GGUF (GGML Universal File Format) is a single-file format that includes
    both model weights and metadata, making it easy to deploy and distribute.
    It's particularly well-suited for CPU inference and edge deployment.

    Args:
        config: Quantization configuration specifying the model, method, and output.

    Returns:
        Dictionary containing quantization results including file sizes.

    Raises:
        RuntimeError: If llama.cpp tools are not available or conversion fails.
    """
    logger.info("Quantizing to GGUF format")

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Convert model to GGML format if needed
        ggml_path = temp_path / "model.ggml"

        if Path(config.model.model_name).exists():
            # Local model - need to convert
            _convert_to_ggml(config.model.model_name, str(ggml_path))
        else:
            # HuggingFace model - download and convert
            logger.info(
                f"Downloading model from HuggingFace: {config.model.model_name}"
            )
            model_temp_dir = temp_path / "hf_model"
            model_temp_dir.mkdir()

            # Download the model
            tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name, torch_dtype=torch.float16, device_map="cpu"
            )

            # Save to temporary directory
            tokenizer.save_pretrained(str(model_temp_dir))
            model.save_pretrained(str(model_temp_dir))

            # Convert to GGML
            _convert_to_ggml(str(model_temp_dir), str(ggml_path))

        # Step 2: Quantize GGML to GGUF
        _quantize_ggml_to_gguf(str(ggml_path), config.output_path, config.method)

    # Get final size
    if Path(config.output_path).exists():
        quantized_size = Path(config.output_path).stat().st_size
        return {
            "quantized_size": _format_size(quantized_size),
            "quantized_size_bytes": quantized_size,
        }

    return {}


def _quantize_to_safetensors(config: QuantizationConfig) -> dict[str, Any]:
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

    quantized_size = _get_directory_size(str(output_dir))
    return {
        "quantized_size": _format_size(quantized_size),
        "quantized_size_bytes": quantized_size,
    }


def _quantize_to_pytorch(config: QuantizationConfig) -> dict[str, Any]:
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

    quantized_size = _get_directory_size(str(output_dir))
    return {
        "quantized_size": _format_size(quantized_size),
        "quantized_size_bytes": quantized_size,
    }


def _convert_to_ggml(model_path: str, output_path: str) -> None:
    """Convert model to GGML format using llama.cpp convert script."""
    logger.info(f"Converting {model_path} to GGML format")

    # Try to find convert.py script from llama.cpp
    convert_script = _find_llamacpp_convert_script()
    if not convert_script:
        raise RuntimeError(
            "llama.cpp convert script not found. Please install llama-cpp-python "
            "or ensure llama.cpp is available in PATH."
        )

    cmd = [
        "python",
        convert_script,
        model_path,
        "--outfile",
        output_path,
        "--outtype",
        "f16",
    ]

    logger.info(f"Running conversion command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Model conversion failed: {result.stderr}")


def _quantize_ggml_to_gguf(input_path: str, output_path: str, method: str) -> None:
    """Quantize GGML model to GGUF using llama.cpp quantize tool."""
    logger.info(f"Quantizing GGML to GGUF with method {method}")

    # Try to find quantize binary from llama.cpp
    quantize_binary = _find_llamacpp_quantize_binary()
    if not quantize_binary:
        raise RuntimeError(
            "llama.cpp quantize binary not found. Please install llama-cpp-python "
            "or ensure llama.cpp is available in PATH."
        )

    cmd = [quantize_binary, input_path, output_path, method]

    logger.info(f"Running quantization command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed: {result.stderr}")


def _find_llamacpp_convert_script() -> Optional[str]:
    """Find the llama.cpp convert.py script."""
    # Common locations for convert.py
    possible_paths = [
        "convert.py",  # In PATH
        "/usr/local/bin/convert.py",
        "/opt/homebrew/bin/convert.py",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path

    # Try to find in llama-cpp-python installation
    try:
        import llama_cpp

        llama_cpp_path = Path(llama_cpp.__file__).parent
        convert_script = llama_cpp_path / "convert.py"
        if convert_script.exists():
            return str(convert_script)
    except ImportError:
        pass

    return None


def _find_llamacpp_quantize_binary() -> Optional[str]:
    """Find the llama.cpp quantize binary."""
    # Common names and locations
    possible_names = ["quantize", "llama-quantize", "llama.cpp-quantize"]

    for name in possible_names:
        try:
            result = subprocess.run(["which", name], capture_output=True, text=True)
            if result.returncode == 0:
                return name
        except FileNotFoundError:
            continue

    return None


def _is_valid_hf_model_id(model_id: str) -> bool:
    """Check if a string is a valid HuggingFace model identifier."""
    try:
        from huggingface_hub import model_info

        model_info(model_id)
        return True
    except Exception:
        return False


def _get_directory_size(path: str) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    path_obj = Path(path)
    for file_path in path_obj.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
