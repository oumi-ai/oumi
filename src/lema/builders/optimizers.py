from typing import Optional

import torch
import transformers
from torch.optim import Optimizer
from transformers.optimization import Adafactor

from lema.core.types import TrainingParams


def build_optimizer(
    model: torch.nn.Module, config: TrainingParams
) -> torch.optim.Optimizer:
    """Builds and returns a PyTorch optimizer based on the provided configuration.

    See pytorch documentation for more information on available optimizers:
    https://pytorch.org/docs/stable/optim.html

    Args:
        model: The model whose parameters will be optimized.
        config: The configuration object containing optimizer parameters.

    Returns:
        Optimizer: The constructed PyTorch optimizer.
    """
    optimizer_name = config.optimizer.lower()

    # Get all parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    fused_available = torch.cuda.is_available()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
            fused=fused_available,
        )
    elif optimizer_name in ("adamw", "adamw_torch", "adamw_torch_fused"):
        return torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
            fused=fused_available,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            fused=fused_available,
        )
    elif optimizer_name == "adafactor":
        return Adafactor(
            trainable_params,
            lr=config.learning_rate,
            beta1=config.adam_beta1,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_lr_scheduler(
    optimizer: Optimizer,
    training_params: TrainingParams,
    num_training_steps: Optional[int] = None,
    last_epoch: int = -1,
    num_cycles: int = 1,
    min_lr: float = 0.0,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
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
    if training_params.lr_scheduler_type is None:
        return None

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
    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type == "cosine":
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
    elif scheduler_type == "constant":
        return transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type == "cosine_with_restarts":
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
