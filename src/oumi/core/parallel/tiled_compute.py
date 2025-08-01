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

"""Tiled MLP compute functionality for memory-efficient training.

This module provides tiled MLP computation that reduces memory usage
by processing the MLP layers in smaller chunks, based on ArcticTraining's
implementation.
"""

import math

import torch
import torch.distributed as dist
from transformers import AutoConfig

from oumi.utils.logging import logger

try:
    from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

    DEEPSPEED_TILED_MLP_AVAILABLE = True
except ImportError:
    logger.warning("DeepSpeed TiledMLP not available")
    DEEPSPEED_TILED_MLP_AVAILABLE = False


def get_model_type(model_name_or_path: str) -> str:
    """Get the model type from model name or path.

    Args:
        model_name_or_path: Model name or path

    Returns:
        Model type string
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.model_type


def tiled_mlp_forward_common(self, x):
    """Common tiled MLP forward implementation for various model architectures.

    This is a monkey patch to replace modeling_llama.LlamaMLP.forward and other
    identical MLP implementations to perform tiled computation of the same.

    Args:
        self: MLP module instance
        x: Input tensor

    Returns:
        Output tensor from tiled MLP computation
    """
    if not DEEPSPEED_TILED_MLP_AVAILABLE:
        logger.warning("DeepSpeed TiledMLP not available, using standard forward")
        # Fallback to standard MLP computation
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    num_shards = "auto"

    if num_shards == "auto":
        bs, seqlen, hidden = x.shape
        num_shards = math.ceil(seqlen / hidden)

        # It's crucial that all ranks run the same number of shards, otherwise if
        # one of the ranks runs fewer shards than the rest, there will be a
        # deadlock as that rank will stop running sooner than others and will not
        # supply its ZeRO-3 weights shard to other ranks. So we will use the max
        # value across all ranks.
        #
        # XXX: but this will run on every layer - it'd be good to cache the
        # number of shards as it doesn't change during the iteration, but may
        # change between iterations if seqlen is varlen
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(num_shards, device=x.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            num_shards = tensor.item()

        logger.debug(
            f"Using {num_shards} shards for tiled MLP computation, "
            f"seqlen={seqlen}, hidden={hidden}"
        )

    compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]

    def mlp_forward(self, x):
        """Inner MLP forward function for tiled computation."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return TiledMLP.apply(
        mlp_forward,
        self,
        x,
        num_shards,
        compute_params,
    )


def enable_tiled_mlp_compute(model_name_or_path: str) -> None:
    """Enable tiled MLP computation for a model.

    Important: This monkey patching call, that overrides the original HF
    Transformers model's MLP class, has to happen before a model is instantiated.

    Currently only some models are supported, but we can easily add support for
    more model architectures if needed.

    Also beware of other packages overriding it - e.g. Liger-Kernel - you can
    tell Liger-Kernel not to override it via its `from_pretrained(..., swiglu=False)`

    Args:
        model_name_or_path: Model name or path to determine model type

    Raises:
        ValueError: If model type is not supported
    """
    if not DEEPSPEED_TILED_MLP_AVAILABLE:
        logger.warning(
            "DeepSpeed TiledMLP not available, skipping tiled MLP compute setup"
        )
        return

    model_type = get_model_type(model_name_or_path)
    logger.info(f"Enabling tiled MLP compute for model type: {model_type}")

    if model_type == "llama":
        from transformers.models.llama import modeling_llama

        modeling_llama.LlamaMLP.forward = tiled_mlp_forward_common
        logger.info("Enabled tiled MLP compute for Llama model")
    elif model_type == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2

        modeling_qwen2.Qwen2MLP.forward = tiled_mlp_forward_common
        logger.info("Enabled tiled MLP compute for Qwen2 model")
    elif model_type == "deepseek":
        try:
            from transformers.models.deepseek import modeling_deepseek

            modeling_deepseek.DeepSeekMLP.forward = tiled_mlp_forward_common
            logger.info("Enabled tiled MLP compute for DeepSeek model")
        except ImportError:
            logger.warning("DeepSeek model not available in transformers version")
    elif model_type == "phi":
        try:
            from transformers.models.phi import modeling_phi

            modeling_phi.PhiMLP.forward = tiled_mlp_forward_common
            logger.info("Enabled tiled MLP compute for Phi model")
        except ImportError:
            logger.warning("Phi model not available in transformers version")
    else:
        raise ValueError(
            f"Model type {model_type} is currently not supported for tiled MLP "
            f"compute. Please open an Issue and ask to add Tiled MLP support for "
            f"{model_type} or alternatively submit a PR."
        )


def is_tiled_mlp_compute_available() -> bool:
    """Check if tiled MLP compute is available.

    Returns:
        True if DeepSpeed TiledMLP is available, False otherwise
    """
    return DEEPSPEED_TILED_MLP_AVAILABLE
