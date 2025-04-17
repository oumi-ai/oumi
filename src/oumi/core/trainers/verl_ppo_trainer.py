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

"""VERL PPO Trainer integration for Oumi.

This trainer adapts the Volcano Engine Reinforcement Learning (VERL) PPO implementation
for use with Oumi. VERL is a flexible, efficient and production-ready RL training
library for LLMs that supports efficient multi-GPU training scenarios.

Note:
    The current implementation depends on VERL's `RayPPOTrainer`, which may not be fully
    implemented or stable in all VERL releases. If you encounter a NotImplementedError,
    consider:

    1. Checking that your VERL installation is complete and up-to-date
    2. Using a different trainer type until VERL integration is more stable
    3. Consulting the VERL documentation or contacting the VERL team for support
"""

import copy
from pathlib import Path
from typing import Callable, Optional

import ray
from datasets import Dataset
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
)
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    CriticWorker,
)
from verl.workers.reward_manager import NaiveRewardManager

from oumi.core.configs import TrainingConfig, TrainingParams
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


@ray.remote(num_gpus=2)
class VerlPpoTrainer(BaseTrainer):
    """VERL PPO Trainer."""

    def __init__(
        self,
        processing_class: Optional[BaseTokenizer],
        args: TrainingParams,
        reward_funcs: list[Callable],
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        """Initialize the VERL PPO trainer.

        Args:
            processing_class: The tokenizer for the model
            args: Training parameters
            reward_funcs: List of reward functions to use
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional keyword arguments
        """
        self.processing_class = processing_class
        self.params = copy.deepcopy(args)
        assert len(reward_funcs) <= 1, "We only support up to one reward function."
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_filepath = None
        self.val_filepath = None
        logger.info(f"Potato available resources: {ray.available_resources()}")

        # Initialize Ray if not already initialized
        # if not ray.is_initialized():
        #     logger.info("Initializing Ray cluster...")
        #     ray.init(
        #         runtime_env={
        #             "env_vars": {
        #                 "TOKENIZERS_PARALLELISM": "true",
        #                 "NCCL_DEBUG": "WARN",
        #                 "VLLM_LOGGING_LEVEL": "WARN",
        #             }
        #         }
        #     )

        self._create_dataset_files()
        self._setup_verl_trainer()

    def _create_dataset_files(self) -> None:
        """Create dataset files for VERL in Parquet format.

        Args:
            num_examples: Number of examples to include in the dummy datasets

        Returns:
            Tuple of (train_file_path, val_file_path)
        """
        # TODO: Add Subfolder for dataset
        self.cache_dir = Path.home() / ".cache" / "oumi" / "verl_datasets"

        train_file = self.cache_dir / "train.parquet"
        self.train_dataset.to_parquet(train_file)
        self.train_filepath = str(train_file)

        if self.eval_dataset:
            val_file = self.cache_dir / "val.parquet"
            self.eval_dataset.to_parquet(val_file)
            self.val_filepath = str(val_file)
        else:
            self.val_filepath = []

    def _create_config(self):
        yaml_path = Path(__file__).parent / "verl_ppo_trainer.yaml"
        config = OmegaConf.load(yaml_path)
        # TODO: Fill in the config with the actual parameters
        config.algorithm.adv_estimator = "grpo"
        config.data.train_files = self.train_filepath
        config.data.val_files = self.val_filepath
        config.data.train_batch_size = 64
        config.data.val_batch_size = 640
        config.data.max_prompt_length = 256
        config.data.max_response_length = 1024
        config.actor_rollout_ref.model.path = (
            "d1shs0ap/cognitive-behaviors-Llama-3.2-3B"
        )
        config.actor_rollout_ref.actor.optim.lr = 1e-6
        config.actor_rollout_ref.model.use_remove_padding = True
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 16
        config.actor_rollout_ref.actor.ppo_micro_batch_size = 4
        config.actor_rollout_ref.actor.use_kl_loss = True
        config.actor_rollout_ref.actor.kl_loss_coef = 0.001
        config.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.actor.fsdp_config.param_offload = False
        config.actor_rollout_ref.actor.fsdp_config.grad_offload = False
        config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size = 4
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.4
        config.actor_rollout_ref.rollout.n = 16
        config.actor_rollout_ref.ref.log_prob_micro_batch_size = 2
        config.actor_rollout_ref.ref.fsdp_config.param_offload = True
        config.algorithm.kl_ctrl.kl_coef = 0.001
        config.trainer.critic_warmup = 0
        config.trainer.logger = ["wandb"]
        config.trainer.val_before_train = False
        config.trainer.n_gpus_per_node = 2
        config.trainer.nnodes = 1
        config.trainer.save_freq = -1
        config.trainer.test_freq = 50
        config.trainer.default_local_dir = "output"
        config.trainer.project_name = "Countdown-cognitive-behaviors"
        config.trainer.experiment_name = "oumi-verl-test"
        config.trainer.total_epochs = 1

        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        return config

    def _setup_verl_trainer(self):
        """Set up the VERL PPO trainer."""
        self.verl_config = self._create_config()
        logger.info(f"VERL config: {self.verl_config}")

        tokenizer = self.processing_class

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Create resource pool manager
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [self.verl_config.trainer.n_gpus_per_node]
            * self.verl_config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if (
            self.verl_config.algorithm.use_kl_in_reward
            or self.verl_config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Create reward function manager
        compute_score = self.reward_funcs[0] if self.reward_funcs else None
        reward_fn = NaiveRewardManager(
            tokenizer=tokenizer, num_examine=0, compute_score=compute_score
        )
        # TODO: Different reward calculation for validation?
        # Could use TinyZero's RewardManager instead.
        val_reward_fn = NaiveRewardManager(
            tokenizer=tokenizer, num_examine=1, compute_score=compute_score
        )

        self.verl_trainer = RayPPOTrainer(
            config=self.verl_config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Train the model using VERL PPO.

        Args:
            resume_from_checkpoint: Optional path to a checkpoint to resume from.
        """
        # TODO: Support resuming from checkpoint. May need to pass this parameter
        # into RayPPOTrainer creation.

        self.verl_trainer.init_workers()
        self.verl_trainer.fit()

    def save_state(self) -> None:
        """Save the Trainer state using VERL's checkpoint handling."""
        # TODO: Implement

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Save the model to the specified output directory.

        Args:
            config: The Oumi training config.
            final: Whether this is the final model being saved during training.
        """
        # TODO: Implement
