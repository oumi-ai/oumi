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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch.distributed.fsdp as torch_fsdp

from oumi.core.configs.params.base_params import BaseParams


class ShardingStrategy(str, Enum):
    """The sharding strategies for FullyShardedDataParallel (FSDP).

    See :external:class:`torch.distributed.fsdp.ShardingStrategy`
    for more details.
    """

    FULL_SHARD = "FULL_SHARD"
    """Shards model parameters, gradients, and optimizer states.
    Provides the most memory efficiency but may impact performance."""

    SHARD_GRAD_OP = "SHARD_GRAD_OP"
    """Shards gradients and optimizer states, but not model
    parameters. Balances memory savings and performance."""

    HYBRID_SHARD = "HYBRID_SHARD"
    """Shards model parameters within a node and replicates them
    across nodes."""

    NO_SHARD = "NO_SHARD"
    """No sharding is applied. Parameters, gradients, and optimizer states
    are kept in full on each GPU."""

    HYBRID_SHARD_ZERO2 = "HYBRID_SHARD_ZERO2"
    """Apply SHARD_GRAD_OP within a node, and replicate
    parameters across nodes."""

    def to_torch(self) -> torch_fsdp.ShardingStrategy:
        """Convert the enum to the corresponding torch_fsdp.ShardingStrategy."""
        strategy_map = {
            ShardingStrategy.FULL_SHARD: torch_fsdp.ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: torch_fsdp.ShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.HYBRID_SHARD: torch_fsdp.ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy.NO_SHARD: torch_fsdp.ShardingStrategy.NO_SHARD,
            ShardingStrategy.HYBRID_SHARD_ZERO2: (
                torch_fsdp.ShardingStrategy._HYBRID_SHARD_ZERO2
            ),
        }

        if self not in strategy_map:
            raise ValueError(f"Unsupported sharding strategy: {self}")

        return strategy_map[self]


class StateDictType(str, Enum):
    """The supported state dict types for FullyShardedDataParallel (FSDP).

    This controls how the model's state dict will be saved during checkpointing, and
    how it can be consumed afterwards.
    """

    FULL_STATE_DICT = "FULL_STATE_DICT"
    """The state dict will be saved in a non-sharded, unflattened format.

    This is similar to checkpointing without FSDP.
    """

    SHARDED_STATE_DICT = "SHARDED_STATE_DICT"
    """The state dict will be saved in a sharded, unflattened format.

    This can be used by other parallel schemes.
    """

    LOCAL_STATE_DICT = "LOCAL_STATE_DICT"
    """The state dict will be saved in a sharded, flattened format.

    Since it's flattened, this can only be used by FSDP.
    """

    def to_torch(self) -> torch_fsdp.StateDictType:
        """Converts to the corresponding torch.distributed.fsdp.StateDictType."""
        state_dict_map = {
            StateDictType.FULL_STATE_DICT: torch_fsdp.StateDictType.FULL_STATE_DICT,
            StateDictType.SHARDED_STATE_DICT: (
                torch_fsdp.StateDictType.SHARDED_STATE_DICT
            ),
            StateDictType.LOCAL_STATE_DICT: torch_fsdp.StateDictType.LOCAL_STATE_DICT,
        }

        if self not in state_dict_map:
            raise ValueError(f"Unsupported state dict type: {self}")

        return state_dict_map[self]


class BackwardPrefetch(str, Enum):
    """The backward prefetch options for FullyShardedDataParallel (FSDP)."""

    BACKWARD_PRE = "BACKWARD_PRE"
    """Enables the most overlap but increases memory usage the most."""

    BACKWARD_POST = "BACKWARD_POST"
    """Enables less overlap but requires less memory usage."""

    NO_PREFETCH = "NO_PREFETCH"
    """Disables backward prefetching altogether."""

    def to_torch(self) -> Optional[torch_fsdp.BackwardPrefetch]:
        """Convert the enum to the corresponding torch_fsdp.BackwardPrefetch."""
        map = {
            BackwardPrefetch.BACKWARD_PRE: torch_fsdp.BackwardPrefetch.BACKWARD_PRE,
            BackwardPrefetch.BACKWARD_POST: torch_fsdp.BackwardPrefetch.BACKWARD_POST,
            BackwardPrefetch.NO_PREFETCH: None,
        }

        if self not in map:
            raise ValueError(f"Unsupported backward prefetch option: {self}")
        return map[self]


class AutoWrapPolicy(str, Enum):
    """The auto wrap policies for FullyShardedDataParallel (FSDP)."""

    SIZE_BASED_WRAP = "SIZE_BASED_WRAP"
    """Wraps layers based on parameter count."""

    TRANSFORMER_BASED_WRAP = "TRANSFORMER_BASED_WRAP"
    """Wraps layers based on the transformer block layer."""

    NO_WRAP = "NO_WRAP"
    """No automatic wrapping is performed."""


@dataclass
class FSDPParams(BaseParams):
    """Configuration options for Pytorch's FullyShardedDataParallel (FSDP) training."""

    enable_fsdp: bool = False
    """If True, enables FullyShardedDataParallel training.

    Allows training larger models by sharding models and gradients across multiple GPUs.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    """Determines how to shard model parameters across GPUs.

    See :external:class:`torch.distributed.fsdp.api.ShardingStrategy` for more details.


    Options:
        FULL_SHARD: Shards model parameters, gradients, and optimizer states.
            Provides the most memory efficiency but may impact performance.
        SHARD_GRAD_OP: Shards gradients and optimizer states, but not model
            parameters. Balances memory savings and performance.
        HYBRID_SHARD: Shards model parameters within a node and replicates them
            across nodes.
        NO_SHARD: No sharding is applied. Parameters, gradients, and optimizer states
            are kept in full on each GPU.
        HYBRID_SHARD_ZERO2: Apply SHARD_GRAD_OP within a node, and replicate
            parameters across nodes.

    Warning:
        NO_SHARD option is deprecated and will be removed in a future release.
            Please use DistributedDataParallel (DDP) instead.
    """

    cpu_offload: bool = False
    """If True, offloads parameters and gradients to CPU when not in use."""

    mixed_precision: Optional[str] = None
    """Enables mixed precision training.

    Options: None, "fp16", "bf16".
    """

    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
    """Determines when to prefetch the next set of parameters.

    Improves throughput by enabling communication and computation overlap
    in the backward pass at the cost of slightly increased memory usage.

    Options:
        BACKWARD_PRE: Enables the most overlap but increases memory
            usage the most. This prefetches the next set of parameters *before*
            the current set of parameters' gradient computation.
        BACKWARD_POST: Enables less overlap but requires less memory
            usage. This prefetches the next set of parameters *after* the current
            set of parameters' gradient computation.
        NO_PREFETCH: Disables backward prefetching altogether. This has no overlap and
            does not increase memory usage. This may degrade throughput significantly.
    """

    forward_prefetch: bool = False
    """If True, prefetches the forward pass results."""

    use_orig_params: Optional[bool] = None
    """If True, uses the PyTorch Module's original parameters for FSDP.

    For more information, see: https://pytorch.org/docs/stable/fsdp.html.
    If not specified, it will be automatically inferred based on other config values.
    """

    state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
    """Specifies the type of state dict to use for checkpointing."""

    auto_wrap_policy: AutoWrapPolicy = AutoWrapPolicy.NO_WRAP
    """Policy for automatically wrapping layers in FSDP."""

    min_num_params: int = 100_000
    """Minimum number of parameters for a layer to be wrapped when using
    size_based policy. This has no effect when using
    transformer_based policy.
    """

    transformer_layer_cls: Optional[str] = None
    """Class name for transformer layers when using transformer_based policy.

    This has no effect when using size_based policy.
    """

    sync_module_states: bool = True
    """If True, synchronizes module states across processes.

    When enabled, each FSDP module broadcasts parameters and buffers from rank 0
    to ensure replication across ranks.
    """
