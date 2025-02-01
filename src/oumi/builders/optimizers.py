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

import torch
from transformers.optimization import Adafactor

from oumi.core.configs import TrainingParams
from oumi.utils.torch_naming_heuristics import group_trainable_params


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
    elif optimizer_name in (
        "adamw_8bit",
        "paged_adamw_8bit",
        "paged_adamw",
        "paged_adamw_32bit",
    ):
        try:
            import bitsandbytes  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "bitsandbytes is not installed. "
                "Please install it with `pip install bitsandbytes` "
                "to use 8-bit or paged optimizers."
            )

        if optimizer_name in ("adamw_8bit", "paged_adamw_8bit"):
            return bitsandbytes.optim.AdamW(
                trainable_param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
                optim_bits=8,
                is_paged=optimizer_name == "paged_adamw_8bit",
            )
        else:  # paged_adamw or paged_adamw_32bit
            return bitsandbytes.optim.PagedAdamW(
                trainable_param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
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
            relative_step=False,
            scale_parameter=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
