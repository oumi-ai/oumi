from typing import Any, Dict, List

import bitsandbytes
import torch
from transformers.optimization import Adafactor
from transformers.trainer_pt_utils import get_parameter_names

from lema.core.types import TrainingParams


def _group_trainable_params(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Groups trainable params by weight decay for optimization.

    As a rule of thumb, we generally want to weight decay all 2d matrices, i.e.
    weight tensors for matmuls/embeddings, and not biases/layernorms.
    """
    # Exclude layernorm and bias tensors.
    decay_parameters = get_parameter_names(
        model, forbidden_layer_types=[torch.nn.LayerNorm]
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # Only include trainable params.
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Group by weight decay.
    return [
        {
            "params": [p for n, p in trainable_params if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in trainable_params if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]


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
    trainable_params = _group_trainable_params(model, config.weight_decay)

    fused_available = torch.cuda.is_available()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            fused=fused_available,
        )
    elif optimizer_name in ("adamw", "adamw_torch", "adamw_torch_fused"):
        return torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            fused=fused_available,
        )
    elif optimizer_name in ("adamw_8bit", "paged_adamw_8bit"):
        return bitsandbytes.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
            optim_bits=8,
            is_paged=optimizer_name == "paged_adamw_8bit",
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            fused=fused_available,
        )
    elif optimizer_name == "adafactor":
        return Adafactor(
            trainable_params,
            lr=config.learning_rate,
            beta1=config.adam_beta1,
            relative_step=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
