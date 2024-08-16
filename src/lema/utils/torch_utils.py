import os
from pprint import pformat
from typing import Any, List, NamedTuple, Optional

import numpy as np
import torch

from lema.core.types import TrainingConfig
from lema.utils.debugging_utils import get_nvidia_gpu_memory_utilization
from lema.utils.logging import logger


def device_cleanup() -> None:
    """Empties cuda cache, good to do before and after training for cleanup."""
    if torch.cuda.is_available():
        logger.debug("Cleaning up GPU memory.")
        logger.debug(
            "GPU memory occupied before cleanup: "
            f"{get_nvidia_gpu_memory_utilization()} MiB"
        )

        torch.cuda.empty_cache()

        logger.debug(f"Memory after cleanup: {get_nvidia_gpu_memory_utilization()} MiB")


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


def log_training_config(config: TrainingConfig) -> None:
    """Logs training config."""
    logger.info(f"TrainingConfig: {pformat(config)}")


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
        return

    ncpus = os.cpu_count()
    num_devices = torch.cuda.device_count()
    logger.info(f"CPU cores: {ncpus} CUDA devices: {num_devices}")

    def _mem_to_gib(x):
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
            f"Memory: [Total: {_mem_to_gib(mem_total)}GiB "
            f"Free: {_mem_to_gib(mem_free)}GiB "
            f"Allocated: {_mem_to_gib(mem_allocated)}GiB "
            f"Cached: {_mem_to_gib(mem_reserved)}GiB]"
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


class ModelParameterCount(NamedTuple):
    all_params: int
    trainable_params: int
    embedding_params: int


def _get_parameter_names(
    model: torch.nn.Module, forbidden_layer_types: List[Any]
) -> List[str]:
    """Returns the names of the model parameters that are not inside a forbidden layer.

    Borrowed from
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in _get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in
    # any child.
    result += list(model._parameters.keys())
    return result


def count_model_parameters(model: torch.nn.Module) -> ModelParameterCount:
    """Counts the number of parameters in a model.

    Args:
        model: The torch-implemented neural network.

    Returns:
        A tuple of (total_parameters, trainable_parameters).
    """
    trainable_params = 0
    all_params = 0
    embedding_params = 0
    embedding_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            # Embedding layers appear in named_parameters with ".weight" at the end
            embedding_layer_names.append(name + ".weight")

    for name, param in model.named_parameters():
        param_count = param.numel()
        all_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        if name in embedding_layer_names:
            embedding_params += param_count

    return ModelParameterCount(
        all_params=all_params,
        trainable_params=trainable_params,
        embedding_params=embedding_params,
    )


def log_trainable_parameters(model: torch.nn.Module) -> None:
    """Logs the number of trainable parameters of the model.

    Args:
        model: The torch-implemented neural network.

    Note: original code:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """
    params = count_model_parameters(model)
    all_params = params.all_params
    trainable_params = params.trainable_params
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_params} "
        f"|| Trainable%: {100 * trainable_params / all_params :.4f}"
    )
