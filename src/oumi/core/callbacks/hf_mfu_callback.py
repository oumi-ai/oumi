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

"""MFU calculator based on theoretical model flops computed by HuggingFace libraries."""

import time
from typing import Optional, Union

import torch
import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.core.distributed import get_device_rank_info, is_world_process_zero
from oumi.performance.mfu import (
    calculate_mfu_from_model_flops_per_second,
)
from oumi.utils.logging import logger
from oumi.utils.torch_utils import get_device_name

_LOGS_KWARG = "logs"

# MFU using only the time between on_step_start and on_step_end (except the first step)
# using built-in HuggingFace model's flops estimate.
_HF_TRAIN_STEP_MFU = "hf_train_step_mfu"
# MFU using the time since training started (except the first step)
# using built-in HuggingFace model's flops estimate.
_HF_TRAIN_MFU = "hf_train_mfu"


class HfMfuTrainerCallback(BaseTrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Relies on model's flops estimate computed by HuggingFace in `total_flos` metric.
    """

    def __init__(
        self,
        dtype: torch.dtype,
    ):
        """Initialize the HfMfuTrainerCallback.

        Args:
            dtype: The data type of the model.
        """
        self._dtype = dtype
        self._time_of_second_step: Optional[float] = None
        self._flops_at_second_step: Optional[float] = None
        self._time_for_train_steps = 0.0
        self._first_step_finished = False

        device_rank_info = get_device_rank_info()
        self._num_devices = device_rank_info.world_size
        self._is_world_rank_zero = is_world_process_zero()
        logger.info(f"HF MFU number of devices: {self._num_devices}")
        self._device_name = get_device_name()

        logger.info(f"HF MFU device name: {self._device_name}")
        if self._device_name == "CPU":
            logger.warning(
                "HF MFU is not supported on CPU, the callback will do nothing."
            )

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
            return

        if self._time_of_second_step is None:
            self._time_of_second_step = self._step_start_time
            if state is not None:
                self._flops_at_second_step = state.total_flos

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

        if self._flops_at_second_step is not None and (
            state is not None and state.total_flos > 0.0
        ):
            flops_since_second_step_on_all_devices = (
                state.total_flos - self._flops_at_second_step
            ) * self._num_devices
            train_step_mfu = calculate_mfu_from_model_flops_per_second(
                device_name=self._device_name,
                num_devices=self._num_devices,
                dtype=self._dtype,
                model_flops_per_second_on_all_devices=(
                    flops_since_second_step_on_all_devices / delta_time_seconds_step
                ),
            )
            train_mfu = calculate_mfu_from_model_flops_per_second(
                device_name=self._device_name,
                num_devices=self._num_devices,
                dtype=self._dtype,
                model_flops_per_second_on_all_devices=(
                    flops_since_second_step_on_all_devices / delta_time_seconds_train
                ),
            )
            if _LOGS_KWARG in kwargs:
                kwargs[_LOGS_KWARG][_HF_TRAIN_STEP_MFU] = train_step_mfu
                kwargs[_LOGS_KWARG][_HF_TRAIN_MFU] = train_mfu
