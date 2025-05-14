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

import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import torch

from oumi.utils.device_utils import get_nvidia_gpu_memory_utilization
from oumi.utils.logging import logger
from oumi.utils.str_utils import compute_utf8_len


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

    # pyright seems to have an issue with torch==2.5.1
    # torch.version is always available, but pyright doesn't know that
    if hasattr(torch, "version"):
        logger.info(f"CUDA version: {torch.version.cuda} ")  # type: ignore

    # For AMD GPUs, these functions return ROCm, MlOpen versions respectively.
    logger.info(
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


def get_device_name() -> str:
    """Returns the name of the device, assuming all are identical."""
    device_name = "CPU"
    if torch.cuda.is_available():
        # Assume all devices are identical
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device_name = "MPS"
    return device_name


@dataclass(frozen=True)
class ModelParameterCount:
    all_params: int
    trainable_params: int
    embedding_params: int

    def __post_init__(self):
        """Ensure that the parameters are valid."""
        for name, value in [
            ("all_params", self.all_params),
            ("trainable_params", self.trainable_params),
            ("embedding_params", self.embedding_params),
        ]:
            if value < 0:
                raise ValueError(f"`{name}` ({value}) must be >= 0.")
            if self.trainable_params > self.all_params:
                raise ValueError(
                    f"`trainable_params` ({self.trainable_params}) cannot be "
                    f"greater than `all_params` ({self.all_params})."
                )
            if self.embedding_params > self.all_params:
                raise ValueError(
                    f"`embedding_params` ({self.embedding_params}) cannot be "
                    f"greater than `all_params` ({self.all_params})."
                )

    @property
    def trainable_params_percent(self) -> float:
        """Percentage of trainable parameters [0.0, 100.0]."""
        if self.all_params == 0:
            return 0.0
        return 100 * self.trainable_params / self.all_params

    @property
    def frozen_params_percent(self) -> float:
        """Percentage of frozen parameters [0.0, 100.0]."""
        return 100.0 - self.trainable_params_percent


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
    """Creates a basic counter of the parameters in a neural model.

    Args:
        model: The torch-implemented neural network.

    Returns:
        ModelParameterCount: A ModelParameterCount for the underlying model.
    """
    trainable_params = 0
    all_params = 0
    embedding_params = 0
    embedding_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            # Embedding layers appear in named_parameters with ".weight" at the end
            param_name = f"{name}.weight" if name else "weight"
            embedding_layer_names.append(param_name)

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


def log_number_of_model_parameters(
    model: torch.nn.Module, use_icons: bool = True
) -> None:
    """Logs the number of parameters of the model.

    Args:
        model: The torch-implemented neural network.
        use_icons: Whether to display emojis/icons in the log output.
    """
    params = count_model_parameters(model)

    # Icons if enabled, else fallback to plain text
    total_label = "🔢 Total" if use_icons else "Total"
    embedding_label = "🔗 Embedding" if use_icons else "Embedding"
    trainable_label = "🎯 Trainable" if use_icons else "Trainable"
    frozen_label = "🔒 Frozen" if use_icons else "Frozen"

    n_space = 11 if use_icons else 9

    logger.info(
        f"\nModel Parameters Summary:\n"
        f"{total_label:<{n_space}} parameters: {params.all_params:,}\n"
        f"{embedding_label:<{n_space}} parameters: {params.embedding_params:,}\n"
        f"{trainable_label:<{n_space}} parameters: {params.trainable_params:,}\n"
        f"{frozen_label:<{n_space}} parameters: "
        f"{params.all_params - params.trainable_params:,} "
        f"({params.frozen_params_percent:.2f}%)\n"
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


def get_dtype_size_in_bytes(
    dtype: Union[str, torch.dtype, npt.DTypeLike],
) -> int:
    """Returns size of this dtype in bytes."""
    if isinstance(dtype, torch.dtype):
        return dtype.itemsize
    elif isinstance(dtype, str):
        if not dtype:
            raise ValueError("Empty string is not a valid dtype")
        try:
            # Try to parse using non-standard names like "f64"
            return get_torch_dtype(dtype).itemsize
        except ValueError:
            return np.dtype(dtype).itemsize

    return np.dtype(dtype).itemsize


def _estimate_item_size_in_bytes(item: Any) -> int:
    if isinstance(item, (int, float)):
        return 4
    elif isinstance(item, (np.ndarray, torch.Tensor)):
        num_elements = math.prod(item.shape)
        return num_elements * get_dtype_size_in_bytes(item.dtype)
    elif isinstance(item, list):
        return _estimate_sample_list_size_in_bytes(item)
    elif isinstance(item, str):
        return compute_utf8_len(item)
    elif isinstance(item, (str, bytes)):
        return len(item)

    return 0


def _estimate_sample_list_size_in_bytes(sample_list: list[Any]) -> int:
    num_elements = len(sample_list)
    if num_elements <= 0:
        return 0
    return sum(_estimate_item_size_in_bytes(item) for item in sample_list)


def estimate_sample_dict_size_in_bytes(sample: dict[str, Any]) -> int:
    """Estimates the approximate total number of bytes in a provided sample.

    Training sample is expected to be a dictionary, where a value is a list,
    tensor, or a numpy array.

    The function works in best effort mode i.e., 100% accuaracy isn't guaranteed.
    The implementation is slow, and shouldn't be called in performance-sensitive code.
    """
    result = 0
    for key, val in sample.items():
        result += compute_utf8_len(key)
        result += _estimate_item_size_in_bytes(val)
    return result


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
    return pad_sequences(sequences, padding_value=padding_value, padding_side="right")


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
    return pad_sequences(sequences, padding_value=padding_value, padding_side="left")


def pad_sequences(
    sequences: list[T], *, padding_value: float = 0, padding_side: Optional[str] = None
) -> torch.Tensor:
    """Pads a list of variable-length tensors to a single tensor.

    Args:
        sequences: list of variable length sequences.
        padding_value: value for padded elements. Default: 0.
        padding_side: side to apply padding to. Valid values:  'right', 'left'.
            If unspecified (`None`), defaults to `right`.

    Returns:
        A tensor with shape (B, L, ...), where B is a batch size (`len(sequences)`),
        L is the longest length (`max(len(sequences[i]))`)
    """
    if not padding_side:
        padding_side = "right"

    if padding_side not in ("right", "left"):
        raise ValueError(
            f"Unsupported padding side: '{padding_side}'. "
            "Valid values: 'right', 'left'."
        )

    if len(sequences) == 0:
        raise ValueError("Empty list is not allowed.")
    tensor_sequences = convert_to_list_of_tensors(sequences)

    try:
        return torch.nn.utils.rnn.pad_sequence(
            tensor_sequences,
            batch_first=True,
            padding_value=padding_value,
            padding_side=padding_side,
        )
    except RuntimeError:
        logger.error(
            "Failed to pad sequences with the shapes: "
            + ", ".join([f"{t.shape}" for t in tensor_sequences])
        )
        raise


class _DimMinMaxSizes(NamedTuple):
    dim_index: int

    min_size: int
    max_size: int

    @property
    def has_variable_sizes(self) -> bool:
        return self.min_size != self.max_size


def _get_dims_min_max_size(tensors_list: list[torch.Tensor]) -> list[_DimMinMaxSizes]:
    num_tensors = len(tensors_list)
    if num_tensors <= 0:
        return []

    first_shape = tensors_list[0].shape

    min_dim_sizes = list(first_shape)
    max_dim_sizes = list(first_shape)
    num_dims = len(min_dim_sizes)

    for tensor_idx in range(num_tensors - 1):
        curr_shape = tensors_list[tensor_idx + 1].shape
        if num_dims != len(curr_shape):
            raise ValueError(
                "Tensors have different number of dimensions: "
                f"{num_dims} vs {len(curr_shape)}! "
                f"Shapes: {first_shape}, {curr_shape}"
            )
        for idx in range(num_dims):
            min_dim_sizes[idx] = min(min_dim_sizes[idx], curr_shape[idx])
            max_dim_sizes[idx] = max(max_dim_sizes[idx], curr_shape[idx])

    return [
        _DimMinMaxSizes(
            dim_index=idx, min_size=min_dim_sizes[idx], max_size=max_dim_sizes[idx]
        )
        for idx in range(num_dims)
    ]


def _format_dims_min_max_sizes(dim_sizes: list[_DimMinMaxSizes]) -> str:
    result: list[str] = [""] * len(dim_sizes)
    for idx, item in enumerate(dim_sizes):
        result[idx] = (
            f"{item.min_size}...{item.max_size}"
            if item.has_variable_sizes
            else f"{item.min_size}"
        )
    return "[" + ", ".join(result) + "]"


def _pad_to_max_dim_and_stack_impl(
    tensors_list: list[torch.Tensor],
    *,
    max_variable_sized_dims: int,
    padding_value: float,
    pad_on_left_side: bool,
) -> torch.Tensor:
    num_tensors = len(tensors_list)
    if num_tensors == 0:
        raise ValueError("Empty list of tensors is not allowed.")
    dim_sizes: list[_DimMinMaxSizes] = _get_dims_min_max_size(tensors_list)
    num_variable_size_dims = sum(
        (1 if item.has_variable_sizes else 0) for item in dim_sizes
    )
    if (
        max_variable_sized_dims >= 0
        and num_variable_size_dims > max_variable_sized_dims
    ):
        raise ValueError(
            "Too many dimensions with variable size. "
            f"Got: {num_variable_size_dims} variable size dimensions. "
            f"Maximum allowed: {max_variable_sized_dims}. "
            f"Dimension sizes: {_format_dims_min_max_sizes(dim_sizes)}."
        )

    if num_variable_size_dims == 0:
        # No need to pad anything, just `stack()`.
        return torch.stack(tensors_list)
    elif num_variable_size_dims == 1 and dim_sizes[0].has_variable_sizes:
        # Use pad_sequences provided by PyTorch, which should be equivalent
        # for the common case.
        if pad_on_left_side:
            return pad_sequences_left_side(tensors_list, padding_value=padding_value)
        return pad_sequences_right_side(tensors_list, padding_value=padding_value)

    max_dim_sizes = [item.max_size for item in dim_sizes]
    result_shape = torch.Size([num_tensors] + max_dim_sizes)

    result = torch.full(
        result_shape,
        padding_value,
        dtype=tensors_list[0].dtype,
        device=tensors_list[0].device,
    )

    max_dim_sizes = torch.Size(max_dim_sizes)
    for i, input_tensor in enumerate(tensors_list):
        target = result.select(0, i)
        if max_dim_sizes == input_tensor.shape:
            target[...] = input_tensor
            continue

        target_view = target[...]

        for dim_idx, curr_size in enumerate(input_tensor.shape):
            max_size = max_dim_sizes[dim_idx]
            if curr_size < max_size:
                start_idx = (max_size - curr_size) if pad_on_left_side else 0
                target_view = target_view.narrow(
                    dim_idx, start=start_idx, length=curr_size
                )

        assert target_view.shape == input_tensor.shape
        target_view[...] = input_tensor

    return result


def pad_to_max_dim_and_stack(
    tensors_list: list[T],
    *,
    max_variable_sized_dims: int = -1,
    padding_value: float = 0,
    padding_side: Optional[str] = None,
) -> torch.Tensor:
    """Stacks variable-length tensors to a single tensor with dimension expansion.

    Some examples:
    1) Two tensors with shapes [24,8], [32,8] are combined to [2,32,8].
    2) Two tensors with shapes [24,1,8], [32,4,8] are combined to [2,32,4,8].
    3) Three tensors with shapes [7,3,5],[8,2,6],[9,1,7] are combined to [3,9,3,7].

    For 1D input tensors, the function is equivalent to `pad_sequences()`.

    If all tensors have the same shape and no padding is required, then the function
    is equivalent to `torch.stack()`.

    Args:
        tensors_list: list of tensors with potentially .
        max_variable_sized_dims: Maximum number of variable-sized dimensions.
            Negative values mean `Unlimited`.
            If you know that your tensors have a pre-defined number `N` of
            variable-sized dimensions (e.g., 1 for `sequence_length`) then
            it's a good idea to set this parameter to catch abnormal inputs
            (`ValueError` will be raised in such cases).
        padding_value: value for padded elements. Default: 0.
        padding_side: side to apply padding to. Valid values:  'right', 'left'.
            If unspecified (`None`), defaults to `right`.

    Returns:
        A tensor with shape (B, L, ...), where B is a batch size (`len(sequences)`),
        L is the longest length (`max(len(sequences[i]))`)
    """
    pad_on_left_side: bool = False
    if not padding_side or padding_side == "right":
        pad_on_left_side = False
    elif padding_side == "left":
        pad_on_left_side = True
    else:
        raise ValueError(
            f"Unsupported padding side: '{padding_side}'. "
            "Valid values: 'right', 'left'."
        )

    input_tensors = convert_to_list_of_tensors(tensors_list)

    try:
        return _pad_to_max_dim_and_stack_impl(
            input_tensors,
            max_variable_sized_dims=max_variable_sized_dims,
            padding_value=padding_value,
            pad_on_left_side=pad_on_left_side,
        )
    except RuntimeError:
        logger.error(
            "Failed to pad and stack tensors with the shapes: "
            + ", ".join([f"{t.shape}" for t in input_tensors])
        )
        raise


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


def get_shape_as_list(x: Any) -> list[int]:
    """Returns shape of an object (tensor or numpy array) as Python list."""
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return list(x.shape)

    raise ValueError(f"Unsupported type: {type(x)}. Must be numpy array, torch tensor.")


class _FreezeModelLayer:
    def __init__(self, name: str, freeze_it: bool):
        self.name: str = name
        self.freeze_it: bool = freeze_it
        self.children: list[_FreezeModelLayer] = []


def _freeze_model_layers_impl(
    module: torch.nn.Module, freeze_layers: list[_FreezeModelLayer], parent_path: str
) -> int:
    result: int = 0
    for model_layer in freeze_layers:
        full_layer_path = (
            (parent_path + "." + model_layer.name) if parent_path else model_layer.name
        )
        if hasattr(module, model_layer.name):
            child_module = getattr(module, model_layer.name)
            if model_layer.freeze_it:
                logger.info(f"Freezing layer '{full_layer_path}'...")
                for param in child_module.parameters(recurse=True):
                    param.requires_grad_(False)
                result += 1
            elif len(model_layer.children) > 0:
                result += _freeze_model_layers_impl(
                    child_module, model_layer.children, full_layer_path
                )
        else:
            logger.warning(f"Layer '{full_layer_path}' not found in model.")

    return result


def _group_freeze_model_layers(freeze_layers: list[str]) -> list[_FreezeModelLayer]:
    dummy_root: _FreezeModelLayer = _FreezeModelLayer(name="", freeze_it=False)

    # Build a tree of nested layers.
    for layer_name in freeze_layers:
        layer: _FreezeModelLayer = dummy_root
        all_parts = list(layer_name.split("."))
        for idx, curr_part in enumerate(all_parts):
            next_layer = next((x for x in layer.children if x.name == curr_part), None)
            # If it's the last part, let's freeze this layer.
            freeze_it = idx + 1 >= len(all_parts)
            if next_layer is None:
                next_layer = _FreezeModelLayer(name=curr_part, freeze_it=freeze_it)
                layer.children.append(next_layer)
            elif freeze_it:
                next_layer.freeze_it = True
            layer = next_layer
    return dummy_root.children


def freeze_model_layers(model: torch.nn.Module, freeze_layers: list[str]) -> int:
    """Recursively freezes model layers.

    Args:
        model: A model to freeze layers in.
        freeze_layers: A list of layer names to freeze.
            Nested layers can be specified using a dot ('.') separator.
            For example, "visual.child.grandchild".
            Layer names not found in the model are ignored.

    Returns:
        The total number of layers successfully frozen.
    """
    root_freeze_layers = _group_freeze_model_layers(freeze_layers)

    return _freeze_model_layers_impl(model, root_freeze_layers, "")
