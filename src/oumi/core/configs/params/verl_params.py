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

"""Parameters for VERL PPO training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from oumi.core.configs.params.base_params import BaseParams


class AdvantageEstimator(str, Enum):
    """Advantage estimation methods supported by VERL."""

    GAE = "gae"
    """Generalized Advantage Estimation."""

    GRPO = "grpo"
    """Group Relative Policy Optimization."""

    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    """Reinforce++ algorithm."""

    REMAX = "remax"
    """ReMax algorithm."""

    RLOO = "rloo"
    """RLOO algorithm."""

    def __str__(self) -> str:
        """Return the raw string value of the enum."""
        return self.value


class TrainingStrategy(str, Enum):
    """Training strategy for VERL."""

    FSDP = "fsdp"
    """Use PyTorch FSDP for distributed training."""

    MEGATRON = "megatron"
    """Use Megatron-LM for distributed training."""

    def __str__(self) -> str:
        """Return the raw string value of the enum."""
        return self.value


class RolloutEngine(str, Enum):
    """Rollout engine for VERL."""

    VLLM = "vllm"
    """Use vLLM for fast generation."""

    SGLANG = "sglang"
    """Use SGLang for generation."""

    TRANSFORMERS = "transformers"
    """Use HuggingFace Transformers for generation."""

    def __str__(self) -> str:
        """Return the raw string value of the enum."""
        return self.value


@dataclass
class VerlParams(BaseParams):
    """Parameters for VERL PPO training."""

    adv_estimator: AdvantageEstimator = AdvantageEstimator.GAE
    """The advantage estimation method to use.
    
    Options include GAE (standard PPO), GRPO (Group Relative Policy Optimization),
    REINFORCE_PLUS_PLUS, REMAX, and RLOO.
    """

    training_strategy: TrainingStrategy = TrainingStrategy.FSDP
    """The distributed training strategy to use.
    
    FSDP uses PyTorch's native fully sharded data parallel, while MEGATRON uses
    Megatron-LM's model parallelism approach.
    """

    rollout_engine: RolloutEngine = RolloutEngine.VLLM
    """The engine to use for generating rollouts during PPO training.
    
    vLLM is the recommended option for best performance.
    """

    n_gpus_per_node: int = 1
    """Number of GPUs per node for training."""

    nnodes: int = 1
    """Number of nodes for distributed training."""

    use_reward_model: bool = False
    """Whether to use a reward model instead of a function-based reward."""

    reward_model_path: Optional[str] = None
    """Path to the reward model checkpoint.
    
    Only used if use_reward_model is True.
    """

    kl_ctrl_type: str = "fixed"
    """KL divergence controller type.
    
    "fixed" uses a constant KL penalty, "adaptive" adjusts the KL penalty during training.
    """

    kl_coef: float = 0.1
    """KL divergence coefficient for PPO loss."""

    target_kl: float = 0.1
    """Target KL divergence for adaptive KL controller."""

    kl_horizon: int = 10000
    """Horizon for adaptive KL controller adjustments."""

    extra_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to VERL configuration."""

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.use_reward_model and not self.reward_model_path:
            raise ValueError(
                "reward_model_path must be provided when use_reward_model is True"
            )
