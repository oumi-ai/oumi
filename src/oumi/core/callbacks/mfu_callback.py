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

import time
from typing import Optional, Union

import torch
import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.core.distributed import get_device_rank_info, is_world_process_zero
from oumi.performance.mfu import calculate_mfu
from oumi.utils.logging import logger
from oumi.utils.torch_utils import get_device_name

_LOGS_KWARG = "logs"

# MFU using only the time between on_step_start and on_step_end (except the first step)
_TRAIN_STEP_MFU = "train_step_mfu"
# MFU using the time since training started (except the first step)
_TRAIN_MFU = "train_mfu"


class MfuTrainerCallback(BaseTrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Should be compatible with all trainers that inherit from transformers.Trainer.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        num_params: int,
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
        self._time_of_second_step: Optional[float] = None
        self._time_for_train_steps = 0.0
        self._tokens_seen_so_far = 0
        self._sequence_length = sequence_length
        self._num_layers = num_layers
        self._num_attention_heads = num_attention_heads
        self._attention_head_size = attention_head_size
        self._add_rematerialization = add_rematerialization
        self._first_step_finished = False
        self._steps_since_last_log = 0

        device_rank_info = get_device_rank_info()
        self._num_devices = device_rank_info.world_size
        self._is_world_rank_zero = is_world_process_zero()
        logger.info(f"MFU number of devices: {self._num_devices}")
        self._device_name = get_device_name()

        logger.info(f"MFU device name: {self._device_name}")
        if self._device_name == "CPU":
            logger.warning("MFU is not supported on CPU, the callback will do nothing.")

    def _callback_disabled(self) -> bool:
        """Check if the callback should be disabled."""
        return not self._is_world_rank_zero or self._device_name == "CPU"

    def on_step_begin(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the beginning of each train step."""
        if self._callback_disabled():
            return

        self._step_start_time = time.time()
        if not self._first_step_finished:
            # Calculate the number of tokens processed per step during the first step
            self._tokens_per_step = (
                args.gradient_accumulation_steps
                * args.per_device_train_batch_size
                * self._num_devices
                * self._sequence_length
            )
            return

        if self._time_of_second_step is None:
            self._time_of_second_step = self._step_start_time

    def on_step_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of each train step.

        Note that this will be called after all gradient accumulation substeps.
        """
        if self._callback_disabled():
            return

        # Keep track of only the training step time for "ideal" MFU
        delta_time_seconds = time.time() - self._step_start_time
        if not self._first_step_finished:
            self._first_step_finished = True
            logger.info(f"First step time: {delta_time_seconds:.2f}s")
            return

        self._time_for_train_steps += delta_time_seconds
        self._steps_since_last_log += 1

    def on_log(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called after logging the last logs."""
        if self._callback_disabled():
            return

        # Avoid logging until after the first step.
        if self._time_of_second_step is None:
            return

        delta_time_seconds_train = time.time() - self._time_of_second_step
        delta_time_seconds_step = self._time_for_train_steps

        tokens_since_last_log = self._tokens_per_step * self._steps_since_last_log
        total_tokens = self._tokens_seen_so_far + tokens_since_last_log

        # MFU using only the time spent on training steps (excluding the first step).
        train_step_mfu = calculate_mfu(
            device_name=self._device_name,
            num_devices=self._num_devices,
            dtype=self._dtype,
            num_params=self._num_params,
            num_tokens=total_tokens,
            delta_time_seconds=delta_time_seconds_step,
            num_layers=self._num_layers,
            num_attention_heads=self._num_attention_heads,
            attention_head_size=self._attention_head_size,
            sequence_length=self._sequence_length,
            add_rematerialization=self._add_rematerialization,
        )
        # MFU using the time since training started (excluding the first step).
        train_mfu = calculate_mfu(
            device_name=self._device_name,
            num_devices=self._num_devices,
            dtype=self._dtype,
            num_params=self._num_params,
            num_tokens=total_tokens,
            delta_time_seconds=delta_time_seconds_train,
            num_layers=self._num_layers,
            num_attention_heads=self._num_attention_heads,
            attention_head_size=self._attention_head_size,
            sequence_length=self._sequence_length,
            add_rematerialization=self._add_rematerialization,
        )
        if _LOGS_KWARG in kwargs:
            kwargs[_LOGS_KWARG][_TRAIN_STEP_MFU] = train_step_mfu
            kwargs[_LOGS_KWARG][_TRAIN_MFU] = train_mfu

        # Cleanup values
        self._tokens_seen_so_far = total_tokens
        self._steps_since_last_log = 0
