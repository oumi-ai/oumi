from dataclasses import dataclass
from typing import Optional

from lema.core.configs.params.base_params import BaseParams


@dataclass
class FSDPParams(BaseParams):
    """Configuration options for FullyShardedDataParallel (FSDP).

    Note:
        This config is currently only used by the `LEMA` trainer. Support for other
        trainers will be added in future releases.
    """

    enable_fsdp: bool = False
    """If True, enables FullyShardedDataParallel training.

    Allows training larger models by sharding models and gradients across multiple GPUs.
    """

    sharding_strategy: str = "FULL_SHARD"
    """Determines how to shard model parameters across GPUs.

    Options:
        "FULL_SHARD": Shards model parameters, gradients, and optimizer states.
            Provides the most memory efficiency but may impact performance.
        "SHARD_GRAD_OP": Shards gradients and optimizer states, but not model
            parameters. Balances memory savings and performance.
        "HYBRID_SHARD": Shards model parameters within a node and replicates them
            across nodes.
        "NO_SHARD": No sharding is applied. Parameters, gradients, and optimizer states
            are kept in full on each GPU.

    Warning:
        "NO_SHARD" option is deprecated and will be removed in a future release.
            Please use DistributedDataParallel (DDP) instead.
    """

    cpu_offload: bool = False
    """If True, offloads parameters and gradients to CPU when not in use."""

    mixed_precision: Optional[str] = None
    """Enables mixed precision training.

    Options: None, "fp16", "bf16".
    """

    backward_prefetch: Optional[str] = "BACKWARD_PRE"
    """Determines when to prefetch the next set of parameters.

    Improves throughput by enabling communication and computation overlap
    in the backward pass at the cost of slightly increased memory usage.

    Options:
        "BACKWARD_PRE": Enables the most overlap but increases memory
            usage the most. This prefetches the next set of parameters *before*
            the current set of parameters' gradient computation.
        "BACKWARD_POST": Enables less overlap but requires less memory
            usage. This prefetches the next set of parameters *after* the current
            set of parameters' gradient computation.
        None: Disables backward prefetching altogether. This has no overlap and does not
            increase memory usage. This may degrade throughput significantly.
    """

    state_dict_type: str = "FULL_STATE_DICT"
    """Specifies the type of state dict to use for checkpointing.

    Options: "FULL_STATE_DICT", "SHARDED_STATE_DICT".
    """

    auto_wrap_policy: Optional[str] = "size_based"
    """Policy for automatically wrapping layers in FSDP.

    Options:
        "size_based": wraps layers based on parameter count.
        "transformer_based": wraps layers based on the transformer block layer.
        None: No automatic wrapping is performed.
    """

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
    """If True, synchronizes module states across processes."""

    forward_prefetch: bool = False
    """If True, prefetches the forward pass results."""
