"""Based on MFU from PaLM paper: https://arxiv.org/pdf/2204.02311."""

import time
from typing import Optional

import torch
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from lema.utils.torch_utils import get_device_rank_info

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
    model_flops = model_flops_per_token * tokens_per_second
    device_flops = _get_device_flops(device_name, dtype) * num_devices
    model_flop_utilization = model_flops / device_flops
    return model_flop_utilization


class MfuTrainerCallback(TrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Should be compatible with all trainers that inherit from transformers.Trainer
    """

    def __init__(
        self,
        dtype: torch.dtype,
        num_params: int,
        program_start_time: float,
        sequence_length: int,
        num_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        attention_head_size: Optional[int] = None,
        add_rematerialization: bool = False,
    ):
        """Initialize the MfuTrainerCallback.

        Args:
            dtype: The data type of the model.
            num_params: The number of parameters in the model.
            program_start_time: The start time of the program.
            sequence_length: The sequence length of the model.
            num_layers: The number of layers in the model.
            num_attention_heads: The number of attention heads in the model.
            attention_head_size: The size of each attention head in the model.
            add_rematerialization: Whether to add rematerialization to FLOPs per token.
        """
        self.dtype = dtype
        self.num_params = num_params
        self.program_start_time = program_start_time
        self.time_for_train_steps = 0.0
        self.prev_tokens_seen = 0
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.sequence_length = sequence_length
        self.add_rematerialization = add_rematerialization

        device_rank_info = get_device_rank_info()
        self.num_devices = device_rank_info.world_size
        # Assume all devices are identical
        self.device_name = torch.cuda.get_device_name(0)

        self.step_count = 0

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called at the beginning of each train step."""
        if not state.is_world_process_zero:
            return

        self.step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called at the end of each train step.

        Note that this will be called after all gradient accumulation substeps.
        """
        if not state.is_world_process_zero:
            return

        delta_time_seconds = time.time() - self.step_start_time

        # Keep track of only the training step time for "ideal" MFU
        self.time_for_train_steps += delta_time_seconds
        self.step_count += 1

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called after logging the last logs."""
        if not state.is_world_process_zero:
            return

        now = time.time()
        delta_time_seconds_actual = now - self.program_start_time
        delta_time_seconds_ideal = self.time_for_train_steps

        tokens_since_last_log = (
            args.gradient_accumulation_steps
            * args.per_device_train_batch_size
            * self.num_devices
            * self.sequence_length
            * self.step_count
        )
        total_tokens = self.prev_tokens_seen + tokens_since_last_log

        ideal_mfu = calculate_mfu(
            device_name=self.device_name,
            num_devices=self.num_devices,
            dtype=self.dtype,
            num_params=self.num_params,
            num_tokens=total_tokens,
            delta_time_seconds=delta_time_seconds_ideal,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            attention_head_size=self.attention_head_size,
            sequence_length=self.sequence_length,
            add_rematerialization=self.add_rematerialization,
        )
        actual_mfu = calculate_mfu(
            device_name=self.device_name,
            num_devices=self.num_devices,
            dtype=self.dtype,
            num_params=self.num_params,
            num_tokens=total_tokens,
            delta_time_seconds=delta_time_seconds_actual,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            attention_head_size=self.attention_head_size,
            sequence_length=self.sequence_length,
            add_rematerialization=self.add_rematerialization,
        )
        if "logs" in kwargs:
            kwargs["logs"]["Ideal MFU"] = ideal_mfu
            kwargs["logs"]["Actual MFU"] = actual_mfu
            kwargs["logs"]["Tokens Seen"] = total_tokens

        # Cleanup values
        self.prev_tokens_seen = total_tokens
        self.step_count = 0
