from typing import Optional

import torch
import transformers

from lema.core.types import SchedulerType, TrainingParams


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_params: TrainingParams,
    num_training_steps: Optional[int] = None,
    last_epoch: int = -1,
    num_cycles: int = 1,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Builds a learning rate scheduler based on the provided training parameters.

    Args:
        optimizer: The optimizer for which to build the learning rate scheduler.
        training_params: The training parameters containing.
        num_training_steps: The total number of training steps
            (required for some schedulers).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        num_cycles (`int`, *optional*, defaults to 1): The number of cycles for the
            cosine and cosine_with_restarts schedulers.
        min_lr (`float`, *optional*, defaults to 0.0): The minimum learning rate.

    Returns:
        A learning rate scheduler or None if no scheduler is specified.
    """
    scheduler_type = training_params.lr_scheduler_type.lower()

    warmup_steps = training_params.warmup_steps
    if training_params.warmup_ratio > 0:
        if num_training_steps is None:
            raise ValueError(
                "num_training_steps must be provided when using warmup_ratio"
            )
        warmup_steps = int(training_params.warmup_ratio * num_training_steps)

    if (
        scheduler_type in ("cosine", "cosine_with_restarts")
        and num_training_steps is None
    ):
        raise ValueError(
            "num_training_steps must be provided when using "
            "cosine or cosine_with_restarts"
        )
    if scheduler_type == SchedulerType.LINEAR:
        return transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type == SchedulerType.COSINE:
        if num_training_steps is None:
            raise ValueError(
                "num_training_steps must be provided when using "
                "cosine or cosine_with_restarts"
            )

        return transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
            num_cycles=num_cycles,
        )
    elif scheduler_type == SchedulerType.CONSTANT:
        return transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        if num_training_steps is None:
            raise ValueError(
                "num_training_steps must be provided when using "
                "cosine or cosine_with_restarts"
            )

        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
            num_cycles=num_cycles,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
