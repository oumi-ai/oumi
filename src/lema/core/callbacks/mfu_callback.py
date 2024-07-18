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

from lema.performance.mfu import calculate_mfu
from lema.utils.logging import logger
from lema.utils.torch_utils import get_device_rank_info


class MfuTrainerCallback(TrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Should be compatible with all trainers that inherit from transformers.Trainer.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        num_params: int,
        start_time_seconds: float,
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
            start_time_seconds: The start time of the program.
            sequence_length: The sequence length of the model.
            num_layers: The number of layers in the model.
            num_attention_heads: The number of attention heads in the model.
            attention_head_size: The size of each attention head in the model.
            add_rematerialization: Whether to add rematerialization to FLOPs per token.
        """
        self.dtype = dtype
        self.num_params = num_params
        self.start_time_seconds = start_time_seconds
        self.time_for_train_steps = 0.0
        self.tokens_seen_so_far = 0
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.sequence_length = sequence_length
        self.add_rematerialization = add_rematerialization

        device_rank_info = get_device_rank_info()
        self.num_devices = device_rank_info.world_size
        logger.info(f"MFU number of devices: {self.num_devices}")
        # Assume all devices are identical
        self.device_name = "CPU"
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)

        logger.info(f"MFU device name: {self.device_name}")
        if self.device_name == "CPU":
            logger.warning("MFU is not supported on CPU, the callback will do nothing.")

        self.steps_since_last_log = 0

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called at the beginning of each train step."""
        if not state.is_world_process_zero or self.device_name == "CPU":
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
        if not state.is_world_process_zero or self.device_name == "CPU":
            return

        delta_time_seconds = time.time() - self.step_start_time

        # Keep track of only the training step time for "ideal" MFU
        self.time_for_train_steps += delta_time_seconds
        self.steps_since_last_log += 1

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called after logging the last logs."""
        if not state.is_world_process_zero or self.device_name == "CPU":
            return

        now = time.time()
        delta_time_seconds_actual = now - self.start_time_seconds
        delta_time_seconds_ideal = self.time_for_train_steps

        tokens_since_last_log = (
            args.gradient_accumulation_steps
            * args.per_device_train_batch_size
            * self.num_devices
            * self.sequence_length
            * self.steps_since_last_log
        )
        total_tokens = self.tokens_seen_so_far + tokens_since_last_log

        # MFU using only the time spent on training steps.
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
        # MFU using the time since training started.
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

        # Cleanup values
        self.tokens_seen_so_far = total_tokens
        self.steps_since_last_log = 0
