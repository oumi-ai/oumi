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
            torch.float16: 71,
            torch.bfloat16: 71,
        },
    },
    # Only used for testing purposes
    # https://cloud.google.com/tpu/docs/v4
    "TPUv4": {
        _TFLOPS: {
            torch.float16: 275,
            torch.bfloat16: 275,
        },
    },
}


def _get_device_flops(device_name: str, dtype: torch.dtype):
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
    if num_devices <= 0:
        raise ValueError(f"Must have a positive number of devices: {num_devices}")
    if num_tokens <= 0:
        raise ValueError(f"Must have a positive number of tokens: {num_tokens}")
    if delta_time_seconds <= 0:
        raise ValueError(f"Must have a positive time: {delta_time_seconds}")

    model_flops_per_token = _get_model_flops_per_token(
        num_params,
        num_layers,
        num_attention_heads,
        attention_head_size,
        sequence_length,
        add_rematerialization,
    )
    tokens_per_second = num_tokens / delta_time_seconds
    model_flops = model_flops_per_token * tokens_per_second
    device_flops = _get_device_flops(device_name, dtype) * num_devices
    model_flop_utilization = model_flops / device_flops
    return model_flop_utilization
