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

"""Model enhancement utilities for applying optimized kernels."""

from typing import Any, Optional

import torch.nn as nn

from oumi.core.configs.params.model_params import ModelParams
from oumi.core.kernels.detection import is_flash_attn3_kernel_available, load_flash_attn3_kernel
from oumi.utils.logging import logger


def enhance_model_with_kernels(model: nn.Module, model_params: ModelParams) -> nn.Module:
    """Apply HuggingFace kernels optimizations to model after loading.
    
    Args:
        model: The PyTorch model to enhance
        model_params: Model configuration parameters
        
    Returns:
        Enhanced model with applied kernel optimizations
    """
    if not model_params.enable_hf_kernels:
        return model
        
    logger.info("Checking for available kernel optimizations...")
    
    # Apply Flash Attention 3 kernels if available and using FA
    if (model_params.attn_implementation == "flash_attention_2" and 
        is_flash_attn3_kernel_available()):
        logger.info("Applying Flash Attention 3 kernels optimization")
        try:
            model = _apply_flash_attn3_kernels(model)
        except Exception as e:
            logger.warning(f"Failed to apply Flash Attention 3 kernels: {e}")
    else:
        logger.info("Flash Attention 3 kernels not available or not using Flash Attention")
    
    return model


def _apply_flash_attn3_kernels(model: nn.Module) -> nn.Module:
    """Apply Flash Attention 3 kernels to attention layers in the model.
    
    Args:
        model: The model to enhance
        
    Returns:
        Model with FA3 kernels applied
        
    Raises:
        RuntimeError: If kernel application fails
    """
    try:
        # Load the flash-attn3 kernel
        flash_attn3_kernel = load_flash_attn3_kernel()
        
        # Find attention layers in the model
        attention_layers = _find_attention_layers(model)
        
        if not attention_layers:
            logger.warning("No attention layers found to optimize")
            return model
            
        logger.info(f"Found {len(attention_layers)} attention layers to optimize")
        
        # Apply kernels to attention layers
        from kernels import kernelize  # pyright: ignore[reportMissingImports]
        
        optimized_count = 0
        for layer_name, layer in attention_layers:
            try:
                kernelize(layer, flash_attn3_kernel)
                optimized_count += 1
                logger.debug(f"Applied FA3 kernel to layer: {layer_name}")
            except Exception as e:
                logger.debug(f"Failed to apply kernel to {layer_name}: {e}")
        
        logger.info(f"Successfully applied FA3 kernels to {optimized_count}/{len(attention_layers)} layers")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to apply Flash Attention 3 kernels: {e}") from e


def _find_attention_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find attention layers in the model for kernel optimization.
    
    Args:
        model: The model to search
        
    Returns:
        List of (layer_name, layer_module) tuples for attention layers
    """
    attention_layers = []
    
    # Common attention layer names in different model architectures
    attention_patterns = [
        "attn",
        "attention", 
        "self_attn",
        "self_attention",
        "multihead_attn",
        "multi_head_attention",
    ]
    
    for name, module in model.named_modules():
        # Check if module name contains attention patterns
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in attention_patterns):
            # Additional checks to ensure this is actually an attention layer
            if _is_attention_layer(module):
                attention_layers.append((name, module))
                
    return attention_layers


def _is_attention_layer(module: nn.Module) -> bool:
    """Check if a module is likely an attention layer.
    
    Args:
        module: The module to check
        
    Returns:
        True if the module appears to be an attention layer
    """
    # Check for common attention layer attributes
    attention_attributes = [
        "num_heads",
        "head_dim", 
        "embed_dim",
        "num_attention_heads",
        "attention_head_size",
        "q_proj",
        "k_proj", 
        "v_proj",
        "query",
        "key",
        "value",
    ]
    
    module_attrs = dir(module)
    return any(attr in module_attrs for attr in attention_attributes)


def get_kernel_optimization_info(model: nn.Module, model_params: ModelParams) -> dict[str, Any]:
    """Get information about available kernel optimizations for the model.
    
    Args:
        model: The model to analyze
        model_params: Model configuration parameters
        
    Returns:
        Dictionary with optimization information
    """
    info = {
        "hf_kernels_enabled": model_params.enable_hf_kernels,
        "attention_implementation": model_params.attn_implementation,
        "flash_attn3_kernel_available": is_flash_attn3_kernel_available(),
        "attention_layers_count": len(_find_attention_layers(model)),
    }
    
    return info