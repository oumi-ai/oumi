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
        self._dtype = dtype
        self._num_params = num_params
        self._start_time_seconds = start_time_seconds
        self._time_for_train_steps = 0.0
        self._tokens_seen_so_far = 0
        self._sequence_length = sequence_length
        self._num_layers = num_layers
        self._num_attention_heads = num_attention_heads
        self._attention_head_size = attention_head_size
        self._sequence_length = sequence_length
        self._add_rematerialization = add_rematerialization

        device_rank_info = get_device_rank_info()
        self._num_devices = device_rank_info.world_size
        logger.info(f"MFU number of devices: {self._num_devices}")
        # Assume all devices are identical
        self._device_name = "CPU"
        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)

        logger.info(f"MFU device name: {self._device_name}")
        if self._device_name == "CPU":
            logger.warning("MFU is not supported on CPU, the callback will do nothing.")

        self.steps_since_last_log = 0

    def _callback_disabled(self, state: TrainerState) -> bool:
        """Check if the callback should be disabled."""
        return not state.is_world_process_zero or self._device_name == "CPU"

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called at the beginning of each train step."""
        if self._callback_disabled(state):
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
        if self._callback_disabled(state):
            return

        delta_time_seconds = time.time() - self.step_start_time

        # Keep track of only the training step time for "ideal" MFU
        self._time_for_train_steps += delta_time_seconds
        self.steps_since_last_log += 1

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Event called after logging the last logs."""
        if self._callback_disabled(state):
            return

        now = time.time()
        delta_time_seconds_actual = now - self._start_time_seconds
        delta_time_seconds_ideal = self._time_for_train_steps

        tokens_since_last_log = (
            args.gradient_accumulation_steps
            * args.per_device_train_batch_size
            * self._num_devices
            * self._sequence_length
            * self.steps_since_last_log
        )
        total_tokens = self._tokens_seen_so_far + tokens_since_last_log

        # MFU using only the time spent on training steps.
        ideal_mfu = calculate_mfu(
            device_name=self._device_name,
            num_devices=self._num_devices,
            dtype=self._dtype,
            num_params=self._num_params,
            num_tokens=total_tokens,
            delta_time_seconds=delta_time_seconds_ideal,
            num_layers=self._num_layers,
            num_attention_heads=self._num_attention_heads,
            attention_head_size=self._attention_head_size,
            sequence_length=self._sequence_length,
            add_rematerialization=self._add_rematerialization,
        )
        # MFU using the time since training started.
        actual_mfu = calculate_mfu(
            device_name=self._device_name,
            num_devices=self._num_devices,
            dtype=self._dtype,
            num_params=self._num_params,
            num_tokens=total_tokens,
            delta_time_seconds=delta_time_seconds_actual,
            num_layers=self._num_layers,
            num_attention_heads=self._num_attention_heads,
            attention_head_size=self._attention_head_size,
            sequence_length=self._sequence_length,
            add_rematerialization=self._add_rematerialization,
        )
        if "logs" in kwargs:
            kwargs["logs"]["Ideal MFU"] = ideal_mfu
            kwargs["logs"]["Actual MFU"] = actual_mfu

        # Cleanup values
        self._tokens_seen_so_far = total_tokens
        self.steps_since_last_log = 0
