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

"""Common utilities for quantization operations."""

from pathlib import Path
from typing import Optional

from oumi.quantize.constants import MODEL_SIZE_ESTIMATES, SIZE_UNITS
from oumi.utils.logging import logger


def is_valid_hf_model_id(model_id: str) -> bool:
    """Check if a string is a valid HuggingFace model identifier."""
    try:
        from huggingface_hub import model_info

        model_info(model_id)
        return True
    except Exception:
        return False


def get_directory_size(path: str) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    path_obj = Path(path)
    for file_path in path_obj.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def get_hf_model_size(model_id: str) -> Optional[int]:
    """Get the size of a HuggingFace model by querying the model info."""
    try:
        from huggingface_hub import HfApi, model_info

        # Try using HfApi to get file info with sizes
        api = HfApi()
        try:
            files = api.list_repo_files(model_id, repo_type="model")
            total_size = 0

            for filename in files:
                if (
                    filename.endswith(".safetensors")
                    or filename.endswith(".bin")
                    or filename.endswith(".pth")
                ):
                    try:
                        # Get file info including size
                        file_info = api.repo_info(model_id, files_metadata=True)
                        for sibling in file_info.siblings or []:
                            if (
                                sibling.rfilename == filename
                                and hasattr(sibling, "size")
                                and sibling.size
                            ):
                                total_size += sibling.size
                    except Exception:
                        continue

            if total_size > 0:
                return total_size
        except Exception:
            pass

        # Fallback: estimate based on model parameters
        try:
            info = model_info(model_id)
            if hasattr(info, "safetensors") and info.safetensors:
                # Sum up parameters and estimate size
                total_params = 0
                for params_info in info.safetensors.get("parameters", {}).values():
                    if isinstance(params_info, (int, float)):
                        total_params += params_info

                if total_params > 0:
                    # Estimate: 2 bytes per parameter for fp16 models
                    return int(total_params * 2)
        except Exception:
            pass

        # Final fallback: use reasonable estimates
        model_id_lower = model_id.lower()
        if "tinyllama" in model_id_lower and "1.1b" in model_id_lower:
            return MODEL_SIZE_ESTIMATES["tinyllama-1.1b"]

        return None
    except Exception as e:
        logger.warning(f"Failed to get HuggingFace model size: {e}")
        return None


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in SIZE_UNITS:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes = int(size_bytes / 1024.0)
    return f"{size_bytes:.1f} PB"


def validate_quantization_config(config) -> None:
    """Validate quantization configuration."""
    from oumi.quantize.constants import SUPPORTED_METHODS, SUPPORTED_OUTPUT_FORMATS

    # Validate output format
    if config.output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {config.output_format}")

    # Validate quantization method
    if config.method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported quantization method: {config.method}")

    # Check if model path exists or is a valid model identifier
    model_path = config.model.model_name
    if not (Path(model_path).exists() or is_valid_hf_model_id(model_path)):
        raise ValueError(f"Model not found: {model_path}")


def calculate_compression_ratio(
    original_size: Optional[int], quantized_size: int
) -> str:
    """Calculate compression ratio between original and quantized model sizes."""
    if original_size and quantized_size:
        ratio = original_size / quantized_size
        return f"{ratio:.2f}x"
    return "Unknown"


def get_model_size_info(
    config, original_size: Optional[int] = None
) -> tuple[dict[str, str], Optional[int]]:
    """Get model size information for quantization results."""
    model_path = config.model.model_name

    # Get original model size if not provided
    if original_size is None:
        if Path(model_path).exists():
            original_size = get_directory_size(model_path)
        else:
            original_size = get_hf_model_size(model_path)

    result = {}
    if original_size:
        result["original_size"] = format_size(original_size)
    else:
        result["original_size"] = "Unknown"

    return result, original_size
