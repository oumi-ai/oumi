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

from typing import Optional

import torch
import transformers
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from oumi.core.configs import SchedulerType, TrainingParams
from oumi.utils.logging import logger


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_params: TrainingParams,
    num_training_steps: Optional[int] = None,
    current_epoch: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Builds a learning rate scheduler based on the provided training parameters.

    Args:
        optimizer: The optimizer for which to build the learning rate scheduler.
        training_params: The training parameters containing the scheduler args.
        num_training_steps: The total number of training steps
            (required for some schedulers).
        current_epoch (`int`, *optional*, defaults to 0):
            The index of the current epoch when resuming training.

    Optional Args (in `training_params.lr_scheduler_kwargs`):
        num_cycles (`float`, *optional*, defaults to 1.0 for cosine_with_restarts and
            0.5 for others): The number of cycles for the cosine family of schedulers.
        min_lr_rate (`float`, *optional*, defaults to 0.0): The minimum learning rate to
            decay to for cosine_with_min_lr.

    Returns:
        A learning rate scheduler.
    """
    scheduler_type = training_params.lr_scheduler_type.lower()
    scheduler_specific_kwargs = training_params.lr_scheduler_kwargs.copy()
    warmup_steps = training_params.warmup_steps
    warmup_ratio = training_params.warmup_ratio
    last_epoch = current_epoch - 1

    if warmup_steps is not None and warmup_ratio is not None:
        raise ValueError("Only one of warmup_steps and warmup_ratio should be provided")

    # Make sure num_training_steps is provided for schedulers that need it
    if (
        scheduler_type
        in (
            SchedulerType.COSINE,
            SchedulerType.COSINE_WITH_RESTARTS,
            SchedulerType.COSINE_WITH_MIN_LR,
            SchedulerType.LINEAR,
        )
        and num_training_steps is None
    ):
        raise ValueError(
            f"num_training_steps must be provided when using {scheduler_type}"
        )

    # Set warmup_steps based on warmup_ratio if provided
    if warmup_ratio is not None:
        if warmup_ratio < 0 or warmup_ratio > 1:
            raise ValueError("warmup_ratio must be in [0, 1]")

        if num_training_steps is None:
            raise ValueError(
                "num_training_steps must be provided when using warmup_ratio"
            )

        warmup_steps = int(warmup_ratio * num_training_steps)
        logger.info(
            f"Using warmup_steps={warmup_steps} based on "
            f"{warmup_ratio} warmup_ratio and {num_training_steps} max steps."
        )

    # Otherwise set warmup_steps to 0 if not provided
    if warmup_steps is None:
        warmup_steps = 0
        logger.info("No warmup steps provided. Setting warmup_steps=0.")

    if scheduler_type == SchedulerType.LINEAR:
        if scheduler_specific_kwargs:
            logger.warning(
                f"Unrecognized scheduler kwargs: {scheduler_specific_kwargs}. "
                "Ignoring them."
            )

        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type == SchedulerType.COSINE:
        assert num_training_steps is not None, "should not be none by this point"

        num_cycles = scheduler_specific_kwargs.pop("num_cycles", 0.5)

        if scheduler_specific_kwargs:
            logger.warning(
                f"Unrecognized scheduler kwargs: {scheduler_specific_kwargs}. "
                "Ignoring them."
            )

        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
            num_cycles=num_cycles,
        )
    elif scheduler_type == SchedulerType.CONSTANT:
        if scheduler_specific_kwargs:
            logger.warning(
                f"Unrecognized scheduler kwargs: {scheduler_specific_kwargs}. "
                "Ignoring them."
            )

        return transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        assert num_training_steps is not None, "should not be none by this point"

        num_cycles = scheduler_specific_kwargs.pop("num_cycles", 1)

        if scheduler_specific_kwargs:
            logger.warning(
                f"Unrecognized scheduler kwargs: {scheduler_specific_kwargs}. "
                "Ignoring them."
            )

        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
            num_cycles=num_cycles,
        )
    elif scheduler_type == SchedulerType.COSINE_WITH_MIN_LR:
        assert num_training_steps is not None, "should not be none by this point"

        num_cycles = scheduler_specific_kwargs.pop("num_cycles", 0.5)
        min_lr_rate = scheduler_specific_kwargs.pop("min_lr_rate", 0.0)

        if scheduler_specific_kwargs:
            logger.warning(
                f"Unrecognized scheduler kwargs: {scheduler_specific_kwargs}. "
                "Ignoring them."
            )

        return transformers.optimization.get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
            num_cycles=num_cycles,
            min_lr_rate=min_lr_rate,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
