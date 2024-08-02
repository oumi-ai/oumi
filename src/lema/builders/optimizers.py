from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
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
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Builds a learning rate scheduler based on the provided training parameters.

    Args:
        optimizer: The optimizer for which to build the learning rate scheduler.
        training_params: The training parameters containing lr scheduler configuration.
        num_training_steps: The total number of training steps
            (required for some schedulers).

    Returns:
        A learning rate scheduler or None if no scheduler is specified.
    """
    # class SchedulerType(ExplicitEnum):
    #     LINEAR = "linear"
    #     COSINE = "cosine"
    #     COSINE_WITH_RESTARTS = "cosine_with_restarts"
    #     POLYNOMIAL = "polynomial"
    #     CONSTANT = "constant"
    #     CONSTANT_WITH_WARMUP = "constant_with_warmup"
    #     INVERSE_SQRT = "inverse_sqrt"
    #     REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    #     COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    #     WARMUP_STABLE_DECAY = "warmup_stable_decay"

    scheduler_type = training_params.lr_scheduler_type.lower()
    scheduler_kwargs = training_params.lr_scheduler_kwargs

    if scheduler_type == "constantlr" or scheduler_type == "constant":
        return None  # No scheduler needed for constant learning rate

    if scheduler_type == "steplr":
        return StepLR(optimizer, **scheduler_kwargs)

    if scheduler_type == "multisteplr":
        return MultiStepLR(optimizer, **scheduler_kwargs)

    if scheduler_type == "exponentiallr":
        return ExponentialLR(optimizer, **scheduler_kwargs)

    if scheduler_type == "cosineannealinglr":
        return CosineAnnealingLR(optimizer, **scheduler_kwargs)

    if scheduler_type == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    if scheduler_type == "cycliclr":
        return CyclicLR(optimizer, **scheduler_kwargs)

    if scheduler_type == "onecyclelr":
        if num_training_steps is None:
            raise ValueError(
                "num_training_steps must be provided for OneCycleLR scheduler"
            )
        return OneCycleLR(optimizer, total_steps=num_training_steps, **scheduler_kwargs)

    if scheduler_type == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(optimizer, **scheduler_kwargs)

    raise ValueError(f"Unsupported learning rate scheduler: {scheduler_type}")
