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

"""VRAM estimation for training configurations.

This module provides estimates for GPU memory requirements based on
model size, fine-tuning approach, and training parameters.

References:
- https://blog.eleuther.ai/transformer-math/
- https://huggingface.co/docs/transformers/perf_train_gpu_one
"""

import math
from typing import TYPE_CHECKING, Optional

from oumi.core.configs.metadata.config_metadata import FinetuningType

if TYPE_CHECKING:
    from oumi.core.configs.base_config import BaseConfig


# Bytes per parameter for different dtypes
_BYTES_PER_PARAM = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}


def estimate_training_vram_gb(
    model_size_billions: float,
    finetuning_type: Optional[FinetuningType] = None,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    gradient_checkpointing: bool = True,
) -> float:
    """Estimate minimum VRAM required for training in GB.

    This provides a rough estimate based on:
    - Model weights memory
    - Optimizer states (AdamW: 2 states per param in fp32)
    - Gradients memory
    - Activation memory (with overhead factor)

    Args:
        model_size_billions: Model size in billions of parameters.
        finetuning_type: Type of fine-tuning (FULL, LORA, QLORA).
        dtype: Data type for model weights ("float32", "float16", "bfloat16").
        batch_size: Per-device batch size.
        gradient_checkpointing: Whether gradient checkpointing is enabled.

    Returns:
        Estimated VRAM requirement in GB.

    Note:
        These are rough estimates. Actual VRAM usage depends on many factors:
        - Model architecture (attention implementation, MLP ratio)
        - Sequence length
        - Framework overhead (PyTorch, CUDA)
        - Memory fragmentation
    """
    params = model_size_billions * 1e9
    bytes_per_param = _BYTES_PER_PARAM.get(dtype, 2)

    if finetuning_type == FinetuningType.QLORA:
        # QLoRA: 4-bit quantized model + LoRA adapters
        # Model weights in 4-bit (~0.5 bytes per param)
        model_memory_gb = (params * 0.5) / 1e9

        # LoRA adds ~1-2% trainable parameters
        lora_ratio = 0.02
        trainable_params = params * lora_ratio

        # Optimizer states for trainable params (AdamW: m, v in fp32)
        optimizer_memory_gb = (trainable_params * 2 * 4) / 1e9

        # Gradients for trainable params
        gradient_memory_gb = (trainable_params * bytes_per_param) / 1e9

        total = model_memory_gb + optimizer_memory_gb + gradient_memory_gb

    elif finetuning_type == FinetuningType.LORA:
        # LoRA: full precision model (frozen) + LoRA adapters
        model_memory_gb = (params * bytes_per_param) / 1e9

        # LoRA adds ~1-2% trainable parameters
        lora_ratio = 0.02
        trainable_params = params * lora_ratio

        # Optimizer states for trainable params
        optimizer_memory_gb = (trainable_params * 2 * 4) / 1e9

        # Gradients for trainable params
        gradient_memory_gb = (trainable_params * bytes_per_param) / 1e9

        total = model_memory_gb + optimizer_memory_gb + gradient_memory_gb

    else:
        # Full fine-tuning: all parameters trainable
        # Model weights
        model_memory_gb = (params * bytes_per_param) / 1e9

        # Optimizer states (AdamW: 2 fp32 states per param)
        optimizer_memory_gb = (params * 2 * 4) / 1e9

        # Gradients
        gradient_memory_gb = (params * bytes_per_param) / 1e9

        total = model_memory_gb + optimizer_memory_gb + gradient_memory_gb

    # Activation memory overhead
    # This is highly variable but we use rough multipliers
    if gradient_checkpointing:
        activation_overhead = 0.3  # ~30% overhead with checkpointing
    else:
        activation_overhead = 0.6  # ~60% overhead without

    total *= 1 + activation_overhead

    # Batch size scaling (larger batches need more activation memory)
    # This is a rough approximation
    batch_multiplier = 1 + (batch_size - 1) * 0.15
    total *= min(batch_multiplier, 2.5)

    # Add framework overhead (~10%)
    total *= 1.1

    return round(total, 1)


def estimate_vram_from_config(
    config: "BaseConfig",
    model_size_billions: Optional[float],
    finetuning_type: Optional[FinetuningType],
) -> Optional[float]:
    """Estimate VRAM from a training config.

    Args:
        config: The training config.
        model_size_billions: Model size in billions.
        finetuning_type: Type of fine-tuning.

    Returns:
        Estimated VRAM in GB or None if cannot estimate.
    """
    from oumi.core.configs.training_config import TrainingConfig

    if not isinstance(config, TrainingConfig):
        return None

    if model_size_billions is None:
        return None

    # Extract relevant parameters from config
    dtype = getattr(config.model, "torch_dtype_str", "bfloat16") or "bfloat16"
    batch_size = getattr(config.training, "per_device_train_batch_size", 1) or 1
    gradient_checkpointing = getattr(
        config.training, "enable_gradient_checkpointing", True
    )

    return estimate_training_vram_gb(
        model_size_billions=model_size_billions,
        finetuning_type=finetuning_type,
        dtype=dtype,
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
    )


def get_recommended_gpus(vram_gb: float, gpu_memory_gb: float = 80.0) -> int:
    """Calculate recommended number of GPUs based on VRAM requirements.

    Args:
        vram_gb: Required VRAM in GB.
        gpu_memory_gb: Available memory per GPU in GB (default: 80 for H100/A100-80GB).

    Returns:
        Recommended number of GPUs.
    """
    if vram_gb <= gpu_memory_gb:
        return 1
    return math.ceil(vram_gb / gpu_memory_gb)


def get_vram_tier(vram_gb: float) -> str:
    """Map VRAM requirement to a human-readable tier.

    Args:
        vram_gb: Required VRAM in GB.

    Returns:
        String describing the tier (e.g., "consumer", "datacenter").
    """
    if vram_gb <= 8:
        return "consumer (RTX 3070/4070)"
    elif vram_gb <= 16:
        return "prosumer (RTX 3090/4090)"
    elif vram_gb <= 24:
        return "workstation (RTX 4090, A5000)"
    elif vram_gb <= 48:
        return "datacenter (A40, L40)"
    elif vram_gb <= 80:
        return "datacenter (A100-80GB, H100)"
    else:
        num_gpus = get_recommended_gpus(vram_gb)
        return f"multi-gpu ({num_gpus}x H100/A100)"
