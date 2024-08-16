import bitsandbytes
import torch
from transformers.optimization import Adafactor

from lema.core.types import TrainingParams
from lema.utils.torch_naming_heuristics import group_trainable_params


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

    # Get parameters that require optimization, grouped by weight decay.
    trainable_param_groups = group_trainable_params(model, config.weight_decay)

    fused_available = torch.cuda.is_available()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            fused=fused_available,
        )
    elif optimizer_name in ("adamw", "adamw_torch", "adamw_torch_fused"):
        return torch.optim.AdamW(
            trainable_param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            fused=fused_available,
        )
    elif optimizer_name in ("adamw_8bit", "paged_adamw_8bit"):
        return bitsandbytes.optim.AdamW(
            trainable_param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
            optim_bits=8,
            is_paged=optimizer_name == "paged_adamw_8bit",
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_param_groups,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            fused=fused_available,
        )
    elif optimizer_name == "adafactor":
        return Adafactor(
            trainable_param_groups,
            lr=config.learning_rate,
            beta1=config.adam_beta1,
            relative_step=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
