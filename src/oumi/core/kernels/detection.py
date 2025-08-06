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

"""Kernel detection utilities for optimized model performance."""

from typing import Optional

from oumi.utils.logging import logger


def is_kernels_available() -> bool:
    """Check if the HuggingFace kernels package is available.
    
    Returns:
        True if the kernels package can be imported, False otherwise.
    """
    try:
        import kernels  # pyright: ignore[reportMissingImports]
        return True
    except ImportError:
        return False


def is_flash_attn3_kernel_available() -> bool:
    """Check if the Flash Attention 3 kernel from HF Hub is available.
    
    Returns:
        True if the flash-attn3 kernel can be loaded, False otherwise.
    """
    if not is_kernels_available():
        return False
        
    try:
        from kernels import get_kernel  # pyright: ignore[reportMissingImports]
        # Try to load the flash-attn3 kernel
        kernel = get_kernel("kernels-community/flash-attn3")
        return kernel is not None
    except Exception as e:
        logger.debug(f"Flash Attention 3 kernel not available: {e}")
        return False


def get_available_kernels_info() -> dict[str, bool]:
    """Get information about available kernel optimizations.
    
    Returns:
        Dictionary with kernel availability status.
    """
    return {
        "kernels_package": is_kernels_available(),
        "flash_attn3_kernel": is_flash_attn3_kernel_available(),
    }


def load_flash_attn3_kernel() -> Optional[object]:
    """Load the Flash Attention 3 kernel from the HF Hub.
    
    Returns:
        The loaded kernel object, or None if not available.
        
    Raises:
        ImportError: If kernels package is not available.
        RuntimeError: If kernel cannot be loaded.
    """
    if not is_kernels_available():
        raise ImportError(
            "kernels package is required. Install with: pip install kernels"
        )
    
    try:
        from kernels import get_kernel  # pyright: ignore[reportMissingImports]
        kernel = get_kernel("kernels-community/flash-attn3")
        if kernel is None:
            raise RuntimeError("Failed to load flash-attn3 kernel from Hub")
        return kernel
    except Exception as e:
        raise RuntimeError(f"Failed to load flash-attn3 kernel: {e}") from e