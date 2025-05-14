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

"""Based on MFU from PaLM paper: https://arxiv.org/pdf/2204.02311."""

from typing import Optional

import torch

_TFLOPS = "tflops"
_DEVICE_SPECS = {
    # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "NVIDIA A100-PCIE-40GB": {
        _TFLOPS: {
            torch.float32: 19.5,
            torch.float16: 312.0,
            torch.bfloat16: 312.0,
        },
    },
    "NVIDIA A100-PCIE-80GB": {
        _TFLOPS: {
            torch.float32: 19.5,
            torch.float16: 312.0,
            torch.bfloat16: 312.0,
        },
    },
    "NVIDIA A100 80GB PCIe": {
        _TFLOPS: {
            torch.float32: 19.5,
            torch.float16: 312.0,
            torch.bfloat16: 312.0,
        },
    },
    "NVIDIA A100-SXM4-40GB": {
        _TFLOPS: {
            torch.float32: 19.5,
            torch.float16: 312.0,
            torch.bfloat16: 312.0,
        }
    },
    "NVIDIA A100-SXM4-80GB": {
        _TFLOPS: {
            torch.float32: 19.5,
            torch.float16: 312.0,
            torch.bfloat16: 312.0,
        }
    },
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
    "NVIDIA GeForce RTX 3090": {
        _TFLOPS: {
            torch.float32: 35.6,
            torch.float16: 71.0,
            torch.bfloat16: 71.0,
        },
    },
    # Only used for testing purposes
    # https://cloud.google.com/tpu/docs/v4
    "TPUv4": {
        _TFLOPS: {
            torch.float16: 275.0,
            torch.bfloat16: 275.0,
        },
    },
    # https://www.nvidia.com/en-us/data-center/l4/
    # Note that values in that page are shown with sparsity.
    "NVIDIA L4": {
        _TFLOPS: {
            torch.float32: 60.0,
            torch.float16: 121.0,
            torch.bfloat16: 121.0,
        },
    },
    # https://www.nvidia.com/en-us/data-center/tesla-t4/
    "Tesla T4": {
        _TFLOPS: {
            torch.float32: 8.1,
            torch.float16: 65.0,
            torch.bfloat16: 65.0,
        },
    },
}


def _get_device_flops(device_name: str, dtype: torch.dtype) -> float:
    """Returns peak TFLOPS for the given device name and dtype."""
    if device_name not in _DEVICE_SPECS:
        raise NotImplementedError(
            f"Unknown device name for getting hardware flops: {device_name}"
        )

    specs = _DEVICE_SPECS[device_name]
    if dtype not in specs[_TFLOPS]:
        raise NotImplementedError(f"Unknown dtype {dtype} for device {device_name}")

    return specs[_TFLOPS][dtype] * 1e12


def _get_model_flops_per_token(
    num_params: int,
    num_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    attention_head_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    add_rematerialization: bool = False,
) -> int:
    """Returns the number of FLOPs per token for the given model configuration."""
    if num_params <= 0:
        raise ValueError(f"Must have a positive number of model params: {num_params}")

    forward_flops = 2 * num_params
    backward_flops = 4 * num_params
    attention_flops = 0
    if num_layers and num_attention_heads and attention_head_size and sequence_length:
        attention_flops = (
            sequence_length
            * num_layers
            * num_attention_heads
            * attention_head_size
            * 12
        )

    rematerialization_flops = 0
    if add_rematerialization:
        # FIXME: Needs to be calculated based on checkpointing configuration
        # 73% of forward and all of attention
        # PaLM paper mentions 75%, but the calculated value requires 73%, paper error?
        rematerialization_flops = int(0.73 * forward_flops + attention_flops)

    return forward_flops + backward_flops + attention_flops + rematerialization_flops


def calculate_mfu_from_model_flops_per_second(
    device_name: str,
    num_devices: int,
    dtype: torch.dtype,
    model_flops_per_second_on_all_devices: float,
) -> float:
    """Returns the number of MFU for the given model flops per second."""
    if num_devices <= 0:
        raise ValueError(f"Must have a positive number of devices: {num_devices}")
    device_flops_per_second = _get_device_flops(device_name, dtype) * num_devices
    model_flop_utilization = (
        model_flops_per_second_on_all_devices / device_flops_per_second
    )
    return model_flop_utilization


def calculate_mfu(
    device_name: str,
    num_devices: int,
    dtype: torch.dtype,
    num_params: int,
    num_tokens: int,
    delta_time_seconds: float,
    num_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    attention_head_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    add_rematerialization: bool = False,
) -> float:
    """Returns the number of MFU for the given model configuration."""
    if num_tokens <= 0:
        raise ValueError(f"Must have a positive number of tokens: {num_tokens}")
    if delta_time_seconds <= 0:
        raise ValueError(f"Must have a positive delta time: {delta_time_seconds}")

    model_flops_per_token = _get_model_flops_per_token(
        num_params,
        num_layers,
        num_attention_heads,
        attention_head_size,
        sequence_length,
        add_rematerialization,
    )
    tokens_per_second = num_tokens / delta_time_seconds
    model_flops_per_second = model_flops_per_token * tokens_per_second

    return calculate_mfu_from_model_flops_per_second(
        device_name=device_name,
        num_devices=num_devices,
        dtype=dtype,
        model_flops_per_second_on_all_devices=model_flops_per_second,
    )
