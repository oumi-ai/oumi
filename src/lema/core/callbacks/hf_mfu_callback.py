"""MFU calculator based on theoretical model flops computed by HuggingFace libraries."""

import time
from typing import Optional, Union

import torch
import transformers

from lema.core.distributed import get_device_rank_info, is_world_process_zero
from lema.core.types import TrainingParams
from lema.performance.mfu import (
    calculate_mfu_from_model_flops_per_second,
)
from lema.utils.logging import logger

_LOGS_KWARG = "logs"

# MFU using only the time between on_step_start and on_step_end (except the first step)
# using built-in HuggingFace model's flops estimate.
_HF_TRAIN_STEP_MFU = "hf_train_step_mfu"
# MFU using the time since training started (except the first step)
# using built-in HuggingFace model's flops estimate.
_HF_TRAIN_MFU = "hf_train_mfu"


class HfMfuTrainerCallback(transformers.TrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Relies on model's flops estimate computed by HuggingFace in `total_flos` metric.
    """

    def __init__(
        self,
        dtype: torch.dtype,
    ):
        """Initialize the MfuTrainerCallback.

        Args:
            dtype: The data type of the model.
        """
        self._dtype = dtype
        self._flops_at_second_step: Optional[float] = None
        self._time_for_train_steps = 0.0
        self._first_step_finished = False

        device_rank_info = get_device_rank_info()
        self._num_devices = device_rank_info.world_size
        self._is_world_rank_zero = is_world_process_zero()
        logger.info(f"HF MFU number of devices: {self._num_devices}")
        # Assume all devices are identical
        self._device_name = "CPU"
        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)

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
            flops_since_second_step = state.total_flos - self._flops_at_second_step
            train_step_mfu = calculate_mfu_from_model_flops_per_second(
                device_name=self._device_name,
                num_devices=self._num_devices,
                dtype=self._dtype,
                model_flops_per_second=(
                    flops_since_second_step / delta_time_seconds_step
                ),
            )
            train_mfu = calculate_mfu_from_model_flops_per_second(
                device_name=self._device_name,
                num_devices=self._num_devices,
                dtype=self._dtype,
                model_flops_per_second=(
                    flops_since_second_step / delta_time_seconds_train
                ),
            )
            if _LOGS_KWARG in kwargs:
                kwargs[_LOGS_KWARG][_HF_TRAIN_STEP_MFU] = train_step_mfu
                kwargs[_LOGS_KWARG][_HF_TRAIN_MFU] = train_mfu
