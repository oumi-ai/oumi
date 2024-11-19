import gc
import os
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypeVar, cast

import numpy as np
import torch

from oumi.utils.device_utils import get_nvidia_gpu_memory_utilization
from oumi.utils.logging import logger


def device_cleanup() -> None:
    """Empties gpu cache, good to do before and after training for cleanup."""
    logger.debug("Running garbage collection.")
    gc.collect()

    if torch.cuda.is_available():
        logger.debug("Cleaning up GPU memory.")
        logger.debug(
            "GPU memory occupied before cleanup: "
            f"{get_nvidia_gpu_memory_utilization()} MiB"
        )

        torch.cuda.empty_cache()

        logger.debug(f"Memory after cleanup: {get_nvidia_gpu_memory_utilization()} MiB")

    elif torch.backends.mps.is_available():
        logger.debug("Cleaning up MPS memory.")
        torch.mps.empty_cache()


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


def format_cudnn_version(v: Optional[int]) -> str:
    """Formats the cuDNN version number.

    Args:
        v: The cuDNN version number.

    Returns:
        A formatted string.
    """
    if v is None:
        return ""
    return ".".join(map(str, (v // 1000, v // 100 % 10, v % 100)))


def log_versioning_info() -> None:
    """Logs misc versioning information."""
    logger.info(f"Torch version: {torch.__version__}. NumPy version: {np.__version__}")
    if not torch.cuda.is_available():
        logger.info("CUDA is not available!")
        return

    # For AMD GPUs, these functions return ROCm, MlOpen versions respectively.
    logger.info(
        f"CUDA version: {torch.version.cuda} "
        f"CuDNN version: {format_cudnn_version(torch.backends.cudnn.version())}"
    )


def log_devices_info(filepath: Optional[Path] = None) -> None:
    """Logs high-level info about all available accelerator devices."""
    if not torch.cuda.is_available():
        return

    ncpus = os.cpu_count()
    num_devices = torch.cuda.device_count()
    log_lines = [f"CPU cores: {ncpus} CUDA devices: {num_devices}"]

    def _mem_to_gib(x):
        return round(float(x) / 1024**3, 2)

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_allocated = torch.cuda.memory_allocated(i)
        mem_reserved = torch.cuda.memory_reserved(i)
        capability = torch.cuda.get_device_capability(i)
        log_lines.append(
            f"device({i})='{device_name}' "
            f"Capability: {capability} "
            f"Memory: [Total: {_mem_to_gib(mem_total)}GiB "
            f"Free: {_mem_to_gib(mem_free)}GiB "
            f"Allocated: {_mem_to_gib(mem_allocated)}GiB "
            f"Cached: {_mem_to_gib(mem_reserved)}GiB]"
        )

    all_text = "\n".join(log_lines)
    logger.info(all_text)

    if filepath:
        with filepath.open("w", encoding="utf-8") as f:
            f.write(all_text)


def log_peak_gpu_memory():
    """Log the peak GPU memory usage."""
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")


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


def log_model_summary(model, filepath: Optional[Path] = None) -> None:
    """Logs a model summary."""
    model_summary = create_model_summary(model)
    logger.info(model_summary)

    if filepath:
        with filepath.open("w", encoding="utf-8") as f:
            f.write(model_summary)


class ModelParameterCount(NamedTuple):
    all_params: int
    trainable_params: int
    embedding_params: int


def _get_parameter_names(
    model: torch.nn.Module, forbidden_layer_types: list[Any]
) -> list[str]:
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


def get_torch_dtype(torch_dtype_str: str) -> torch.dtype:
    """Converts string dtype to torch.dtype."""
    torch_dtype_str = torch_dtype_str.lower()
    if torch_dtype_str in ["f64", "float64", "double"]:
        return torch.float64
    elif torch_dtype_str in ["f32", "float32", "float"]:
        return torch.float32
    elif torch_dtype_str in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif torch_dtype_str in ["f16", "float16", "half"]:
        return torch.float16
    elif torch_dtype_str in ["uint8"]:
        return torch.uint8
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype_str}")


def coerce_model_to_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Coerces the model to the desired dtype.

    This is needed as a temporary workaround to support QLoRA FSDP training. See:
    https://github.com/huggingface/accelerate/issues/1620#issuecomment-2407102051
    """
    for name, module in model.named_modules():
        try:
            module.to(dtype)
        except Exception as e:
            logger.warning(
                f"Failed to coerce module {name} to dtype {dtype}. Error: {e}"
            )


T = TypeVar("T")


def convert_to_list_of_tensors(values: list[T]) -> list[torch.Tensor]:
    """Converts a list of array-like objects into alist of torch tensors."""
    if len(values) == 0:
        return []

    first_item = values[0]
    if isinstance(first_item, torch.Tensor):
        return [cast(torch.Tensor, item) for item in values]

    if isinstance(first_item, np.ndarray):
        return [torch.from_numpy(item) for item in values]
    elif isinstance(first_item, list):
        return [torch.from_numpy(np.asarray(item)) for item in values]

    raise ValueError(
        f"Unsupported element type: {type(first_item)}. "
        "Must be numpy array, torch tensor, or Python list."
    )


def _pad_sequences_impl(
    sequences: list[torch.Tensor], *, padding_value: float = 0
) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value
    )


def pad_sequences_right_side(
    sequences: list[T], *, padding_value: float = 0
) -> torch.Tensor:
    """Pads a list of variable-length tensors to a single tensor.

    Appends `padding_value` to the right side of each sequence
    to expand to the longest length.

    Args:
        sequences: list of variable length sequences.
        padding_value: value for padded elements. Default: 0.

    Returns:
        A tensor with shape (B, L, ...), where B is a batch size (`len(sequences)`),
        L is the longest length (`max(len(sequences[i]))`)
    """
    if len(sequences) == 0:
        raise ValueError("Empty list is not allowed.")
    tensor_sequences = convert_to_list_of_tensors(sequences)

    return _pad_sequences_impl(tensor_sequences, padding_value=padding_value)


def pad_sequences_left_side(
    sequences: list[T], *, padding_value: float = 0
) -> torch.Tensor:
    """Pads a list of variable-length tensors to a single tensor.

    Prepends `padding_value` to the left side of each sequence
    to expand to the longest length.

    Args:
        sequences: list of variable length sequences.
        padding_value: value for padded elements. Default: 0.

    Returns:
        A tensor with shape (B, L, ...), where B is a batch size (`len(sequences)`),
        L is the longest length (`max(len(sequences[i]))`)
    """
    if len(sequences) == 0:
        raise ValueError("Empty list is not allowed.")
    tensor_sequences = convert_to_list_of_tensors(sequences)

    # FIXME OPE-644 Start using `torch.nn.utils.rnn.pad_sequence(padding_size="left")`
    # after we migrate to torch >=2.5.*.

    # For now, do this to achieve left side padding:
    # 1. Reverse all input sequences.
    # 2. Right pad.
    # 3. Unreverse all sequences in right-padded result.
    # Note that torch.flip() copies tensors, so there is performance cost.

    tensor_sequences = [torch.flip(s, dims=(0,)) for s in tensor_sequences]
    result = _pad_sequences_impl(tensor_sequences, padding_value=padding_value)
    result = torch.flip(result, dims=(1,))
    return result


def pad_sequences(
    sequences: list[T], *, padding_value: float = 0, padding_side: Optional[str] = None
) -> torch.Tensor:
    """Pads a list of variable-length tensors to a single tensor.

    Args:
        sequences: list of variable length sequences.
        padding_value: value for padded elements. Default: 0.
        padding_side: side to apply padding to. Valid values:  'right', 'left'.

    Returns:
        A tensor with shape (B, L, ...), where B is a batch size (`len(sequences)`),
        L is the longest length (`max(len(sequences[i]))`)
    """
    if not padding_side or padding_side == "right":
        return pad_sequences_right_side(sequences, padding_value=padding_value)
    elif padding_side == "left":
        return pad_sequences_left_side(sequences, padding_value=padding_value)

    raise ValueError(
        f"Unsupported padding side: '{padding_side}'. Valid values: 'right', 'left'."
    )


def create_ones_like(
    values: T,
) -> T:
    """Converts an array-like object into an object of the same type filled with 1-s.

    Supports nested lists, in which case all elements must be of the same type.
    """
    if isinstance(values, torch.Tensor):
        return torch.ones_like(values)
    elif isinstance(values, np.ndarray):
        return np.ones_like(values)
    elif not isinstance(values, list):
        raise ValueError(
            f"Unsupported type: {type(values)}. "
            "Must be numpy array, torch tensor, or Python list."
        )

    if len(values) == 0:
        return cast(T, [])

    first_item = values[0]
    if isinstance(first_item, (int, float)):
        result = list(np.ones_like(values))
    else:
        # Nested list
        first_item_type = type(first_item)
        result = []
        for idx, item in enumerate(values):
            if idx > 0 and not isinstance(item, first_item_type):
                raise ValueError(
                    "Sequence contains elements of different types: "
                    f"{first_item_type} and {type(item)}."
                )
            result.append(create_ones_like(item))

    return cast(T, result)


def get_first_dim_len(x: Any) -> int:
    """Returns length of the first dimension."""
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return int(x.shape[0])
    elif isinstance(x, list):
        return len(x)

    raise ValueError(
        f"Unsupported type: {type(x)}. "
        "Must be numpy array, torch tensor, or Python list."
    )
