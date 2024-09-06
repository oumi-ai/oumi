from dataclasses import dataclass
from typing import Optional

from lema.core.configs.params.base_params import BaseParams


@dataclass
class FSDPParams(BaseParams):
    """Configuration options for Fully Sharded Data Parallel (FSDP)."""

    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    mixed_precision: Optional[str] = None
    backward_prefetch: str = "BACKWARD_PRE"
    activation_checkpointing: bool = False
    enable_fsdp: bool = False
    state_dict_type: str = "FULL_STATE_DICT"
    auto_wrap_policy: str = "size_based"
    min_num_params: int = 100000
    transformer_layer_cls: Optional[str] = None
