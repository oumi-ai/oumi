import os
from typing import Any, NamedTuple, Optional

import numpy as np
import torch

from lema.logging import logger
from lema.utils.debugging_utils import get_nvidia_gpu_memory_utilization


def device_cleanup() -> None:
    """Empties cuda cache, good to do before and after training for cleanup."""
    if torch.cuda.is_available():
        logger.debug("Cleaning up GPU memory.")
        logger.debug(
            "GPU memory occupied before cleanup: "
            f"{get_nvidia_gpu_memory_utilization()} MB"
        )

        torch.cuda.empty_cache()

        logger.debug(f"Memory after cleanup: {get_nvidia_gpu_memory_utilization()} MB")


def limit_per_process_memory(percent: float = 0.95) -> None:
    """Limits process memory by a certain percentage.

    On Windows and WSL, there's a pool of 'shared gpu memory'.
    This pool is using the RAM (slow) on one's machine rather than actual
    VRAM (fast). Setting this value ensures your machine never uses the slow
    memory and OOMs instead. Note that this may not be needed on Linux machines
    since this is an OS-level feature.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(percent)


def log_versioning_info() -> None:
    """Logs misc versioning information."""
    logger.info(f"Torch version: {torch.__version__}. NumPy version: {np.__version__}")
    if not torch.cuda.is_available():
        logger.info("CUDA is not available!")
        return

    def _format_cudnn_version(v: Optional[int]) -> str:
        if v is None:
            return ""
        return ".".join(map(str, (v // 1000, v // 100 % 10, v % 100)))

    # For AMD GPUs, these functions return ROCm, MlOpen versions respectively.
    logger.info(
        f"CUDA version: {torch.version.cuda} "
        f"CuDNN version: {_format_cudnn_version(torch.backends.cudnn.version())}"
    )


def log_devices_info() -> None:
    """Logs high-level info about all available accelerator devices."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available!")
        return

    num_devices = torch.cuda.device_count()
    logger.info(f"CUDA devices: {num_devices}")

    def _mem_to_gb(x):
        return round(float(x) / 1024**3, 2)

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_allocated = torch.cuda.memory_allocated(i)
        mem_reserved = torch.cuda.memory_reserved(i)
        capability = torch.cuda.get_device_capability(i)
        logger.info(
            f"device({i})='{device_name}' "
            f"Capability: {capability} "
            f"Memory: [Total: {_mem_to_gb(mem_total)}GB "
            f"Free: {_mem_to_gb(mem_free)}GB "
            f"Allocated: {_mem_to_gb(mem_allocated)}GB "
            f"Cached: {_mem_to_gb(mem_reserved)}GB]"
        )


DeviceRankInfo = NamedTuple(
    "DeviceRankInfo",
    [
        ("world_size", int),
        ("rank", int),
        ("local_world_size", int),
        ("local_rank", int),
    ],
)


def get_device_rank_info() -> DeviceRankInfo:
    """Returns device rank and world size."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive. Actual: {world_size}.")
    rank = int(os.environ.get("RANK", 0))
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"RANK must be within this range [0, {world_size}). Actual: {rank}."
        )
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if local_world_size <= 0 or local_world_size > world_size:
        raise ValueError(
            f"LOCAL_WORLD_SIZE must be within this range [1, {world_size}]. "
            f"Actual: {local_world_size}."
        )
    max_allowed_local_rank = min(rank, local_world_size - 1)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank < 0 or local_rank > max_allowed_local_rank:
        raise ValueError(
            f"LOCAL_RANK must be within this range [0, {max_allowed_local_rank}]. "
            f"Actual: {local_rank}."
        )
    return DeviceRankInfo(
        world_size=world_size,
        rank=rank,
        local_world_size=local_world_size,
        local_rank=local_rank,
    )


def create_model_summary(model: Any) -> str:
    """Creates a model summary as a free-formed string."""
    lines = ["Model summary:", repr(model), ""]

    module_lines = [f"{name} ({type(layer)})" for name, layer in model.named_modules()]

    lines.append(f"Modules ({len(module_lines)}):")
    lines.extend(module_lines)
    lines.append("")

    # TODO: Consider whether to use `torchsummary` library here.
    # Caveat: it may require sample inputs/shapes, and other aux info.
    return "\n".join(lines)


def log_model_summary(model) -> None:
    """Logs a model summary."""
    logger.info(create_model_summary(model))


def log_trainable_parameters(model: torch.nn.Module) -> None:
    """Logs the number of trainable parameters of the model.

    Args:
        model: The torch-implemented neural network.

    Note: original code:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        (
            f"Trainable params: {trainable_params} || All params: {all_param} "
            f"|| Trainable%: {100 * trainable_params / all_param :.4f}"
        )
    )
