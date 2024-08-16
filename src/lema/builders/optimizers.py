from typing import Any, Dict, List

import bitsandbytes
import torch
from transformers.optimization import Adafactor

from lema.core.types import TrainingParams

_PARAMS_KEY = "params"
_WEIGHT_DECAY_KEY = "weight_decay"


def _get_parameter_names(
    model: torch.nn.Module, forbidden_layer_types: List[Any]
) -> List[str]:
    """Returns the names of the model parameters that are not inside a forbidden layer.

    Borrowed from
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in _get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in
    # any child.
    result += list(model._parameters.keys())
    return result


def _group_trainable_params(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Groups trainable params by weight decay for optimization.

    As a rule of thumb, we generally want to weight decay all 2d matrices, i.e.
    weight tensors for matmuls/embeddings, and not biases/layernorms.

    Args:
        model: The model whose parameters will be optimized.
        weight_decay: The weight decay to apply to the appropriate parameters.

    Returns:
        List[Dict[str, Any]]: A list containing two dictionaries: the first with
            parameters that should be weight decayed and the second with parameters
            that shouldn't.
    """
    # Exclude layernorm and bias tensors.
    decay_parameters = _get_parameter_names(
        model, forbidden_layer_types=[torch.nn.LayerNorm]
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # Only include trainable params.
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Group by weight decay.
    return [
        {
            _PARAMS_KEY: [p for n, p in trainable_params if n in decay_parameters],
            _WEIGHT_DECAY_KEY: weight_decay,
        },
        {
            _PARAMS_KEY: [p for n, p in trainable_params if n not in decay_parameters],
            _WEIGHT_DECAY_KEY: 0.0,
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
    trainable_param_groups = _group_trainable_params(model, config.weight_decay)

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
