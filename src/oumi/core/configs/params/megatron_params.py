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

from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class MegatronParams(BaseParams):
    """Configuration for Megatron-Core parallelism via mcore_adapter.

    These parameters control tensor, pipeline, and expert parallelism
    for distributed training with mcore_adapter (MCA).

    Requires mcore_adapter to be installed::

        pip install "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"
    """

    tensor_model_parallel_size: int = 1
    """Number of GPUs for tensor parallelism."""

    pipeline_model_parallel_size: int = 1
    """Number of GPUs for pipeline parallelism."""

    expert_model_parallel_size: int = 1
    """Number of GPUs for expert parallelism (MoE models)."""

    virtual_pipeline_model_parallel_size: int | None = None
    """Number of virtual pipeline stages (interleaved scheduling)."""

    sequence_parallel: bool = False
    """Enable sequence parallelism (requires TP > 1)."""

    use_distributed_optimizer: bool = False
    """Shard optimizer state across data-parallel ranks."""

    recompute_granularity: str | None = None
    """Activation recomputation: 'full' or 'selective'."""

    transformer_impl: str = "transformer_engine"
    """Transformer implementation: 'transformer_engine' or 'local'."""

    auto_convert_to_hf: bool = True
    """Automatically convert MCA checkpoint to HF format on final save."""

    mca_config_overrides: dict[str, Any] = field(default_factory=dict)
    """Additional overrides passed to McaTrainingArguments."""

    def is_enabled(self) -> bool:
        """Returns True if any Megatron parallelism is configured."""
        return (
            self.tensor_model_parallel_size > 1
            or self.pipeline_model_parallel_size > 1
            or self.expert_model_parallel_size > 1
        )
