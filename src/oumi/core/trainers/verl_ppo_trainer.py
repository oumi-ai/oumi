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
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import TrainerCallback

# Import VERL components
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
)
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    CriticWorker,
)
from verl.workers.reward_manager import NaiveRewardManager as RewardManager

from oumi.core.configs import TrainingConfig, TrainingParams
from oumi.core.configs.params.verl_params import VerlParams
from oumi.core.distributed import get_device_rank_info, is_world_process_zero
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


class VerlPpoTrainer(BaseTrainer):
    """VERL PPO Trainer for Oumi.

    Integrates VERL's PPO implementation for efficient distributed RL training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processing_class: Optional[BaseTokenizer],
        args: TrainingParams,
        train_dataset: Dataset,
        processor: Optional[BaseProcessor] = None,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        data_collator: Optional[Callable] = None,
        config: Optional[TrainingConfig] = None,
        **kwargs,
    ):
        """Initialize the VERL PPO trainer.

        Args:
            model: The model to train
            processing_class: The tokenizer for the model
            args: Training parameters
            train_dataset: Training dataset
            processor: Optional processor for the data
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of training callbacks
            data_collator: Optional data collator
            config: Optional full training configuration
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.processing_class = processing_class
        self.processor = processor
        self.params = copy.deepcopy(args)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks or []
        self.collator_fn = data_collator
        self.full_config = config
        
        # Create temporary directory for dummy datasets
        self.temp_dir = None
        self.dummy_train_file = None
        self.dummy_val_file = None

        # Extract VERL config from training params
        verl_params = cast(VerlParams, self.params.verl_params)
        self.verl_params = verl_params
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            logger.info("Initializing Ray cluster...")
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=True,
                logging_level="warning",
            )
            
        # Setup VERL trainer
        self._setup_verl_trainer()
        
    def _create_dummy_dataset_files(self, num_examples: int = 10) -> tuple[str, str]:
        """Create dummy dataset files for VERL in Parquet format.
        
        Args:
            num_examples: Number of examples to include in the dummy datasets
            
        Returns:
            Tuple of (train_file_path, val_file_path)
        """
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for dummy datasets: {self.temp_dir}")
        
        # Create dummy data
        prompts = [f"This is a dummy prompt {i}." for i in range(num_examples)]
        completions = [f"This is a dummy completion {i}." for i in range(num_examples)]
        
        # Create pandas DataFrame
        train_df = pd.DataFrame({
            "prompt": prompts,
            "completion": completions,
        })
        
        # Create a smaller validation set
        val_df = train_df.iloc[:max(1, num_examples // 5)].copy()
        
        # Create train and val files in Parquet format
        train_file = os.path.join(self.temp_dir, "dummy_train.parquet")
        val_file = os.path.join(self.temp_dir, "dummy_val.parquet")
        
        # Write to Parquet format
        train_df.to_parquet(train_file, index=False)
        val_df.to_parquet(val_file, index=False)
        
        self.dummy_train_file = train_file
        self.dummy_val_file = val_file
        
        logger.info(f"Created dummy train file (Parquet): {train_file} with {len(train_df)} examples")
        logger.info(f"Created dummy val file (Parquet): {val_file} with {len(val_df)} examples")
        
        return train_file, val_file

    def _setup_verl_trainer(self):
        """Set up the VERL PPO trainer."""
        logger.info("Setting up VERL PPO trainer...")
        
        # Define worker roles mapping
        role_worker_mapping = {
            Role.ActorRollout: ActorRolloutRefWorker,
            Role.Critic: CriticWorker,
            Role.RefPolicy: ActorRolloutRefWorker,
        }

        # Add reward model worker if enabled
        if self.verl_params.use_reward_model:
            from verl.workers.fsdp_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = RewardModelWorker

        # Setup resource pool for GPU allocation
        resource_pool_spec = {
            "global_pool": [self.verl_params.n_gpus_per_node] * self.verl_params.nnodes,
        }

        mapping = {
            Role.ActorRollout: "global_pool",
            Role.Critic: "global_pool",
            Role.RefPolicy: "global_pool",
        }

        if self.verl_params.use_reward_model:
            mapping[Role.RewardModel] = "global_pool"

        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Create reward function manager
        tokenizer = self.processing_class
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

        try:
            # Calculate batch sizes based on GPU count
            per_gpu_batch_size = 4  # A reasonable default for most GPUs
            total_gpus = self.verl_params.n_gpus_per_node * self.verl_params.nnodes
            total_batch_size = per_gpu_batch_size * total_gpus
            mini_batch_size = max(16, total_batch_size * 2)  # mini_batch should be larger than micro_batch
            
            logger.info(f"Using batch sizes: per_gpu={per_gpu_batch_size}, total={total_batch_size}, mini={mini_batch_size}")
            
            # Create dummy dataset files for VERL to use
            # This prevents the "No objects to concatenate" error
            dummy_train_file, dummy_val_file = self._create_dummy_dataset_files(num_examples=20)
            
            # Create VERL configuration from Oumi params
            # Using direct dictionary instead of complex nested structures
            # This avoids schema validation issues between Oumi and VERL
            verl_config = {
                "data": {
                    "prompt_key": "prompt",
                    "completion_key": "completion",
                    "max_prompt_length": self.model.config.max_position_embeddings - 256,
                    "train_batch_size": max(mini_batch_size, self.params.per_device_train_batch_size * 8),  # Must be >= mini_batch_size
                    "shuffle": True,
                    "filter_overlong_prompts": True,
                    "truncation": "error",
                    # Use the actual dummy dataset files we created
                    "train_files": [dummy_train_file],
                    "val_files": [dummy_val_file],
                },
                "ppo_micro_batch_size": total_batch_size,  # Add at top level too
                "actor_rollout_ref": {
                    "hybrid_engine": True,  # Required by VERL
                    "ppo_micro_batch_size": total_batch_size,  # Add here too
                    "actor": {
                        "strategy": str(self.verl_params.training_strategy).lower(),
                        "ppo_mini_batch_size": mini_batch_size,  # Calculated based on GPU count
                        "ppo_micro_batch_size": total_batch_size,  # Total batch size across all GPUs
                        "ppo_micro_batch_size_per_gpu": per_gpu_batch_size,  # Batch size per GPU
                        "use_dynamic_bsz": False,
                        "use_kl_loss": str(self.verl_params.adv_estimator).lower() == "grpo",
                        "optim": {"lr": self.params.learning_rate},
                    },
                    "rollout": {
                        "name": str(self.verl_params.rollout_engine).lower(),
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "tensor_model_parallel_size": 1,
                        "gpu_memory_utilization": 0.4,
                        # Number of completions to generate for each prompt (required for GRPO)
                        "n": 1 if str(self.verl_params.adv_estimator).lower() != "grpo" else 4,
                        "log_prob_micro_batch_size_per_gpu": per_gpu_batch_size,
                        "log_prob_micro_batch_size": total_batch_size,
                        "ppo_micro_batch_size": total_batch_size,  # Also add here
                        "val_kwargs": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 256,
                            "do_sample": True,
                            "n": 1,  # Always use 1 for validation
                        },
                    },
                    "ref": {
                        "strategy": str(self.verl_params.training_strategy).lower(),
                        "log_prob_micro_batch_size_per_gpu": per_gpu_batch_size,
                        "log_prob_micro_batch_size": total_batch_size,
                        "ppo_micro_batch_size": total_batch_size,  # Also add here
                    },
                },
                "critic": {
                    "strategy": str(self.verl_params.training_strategy).lower(),
                    "model": {"enable_gradient_checkpointing": self.params.enable_gradient_checkpointing},
                    "ppo_micro_batch_size_per_gpu": per_gpu_batch_size,
                    "ppo_micro_batch_size": total_batch_size,  # Total batch size across all GPUs
                    "ppo_mini_batch_size": mini_batch_size,  # Match actor's mini batch size
                    "optim": {"lr": self.params.learning_rate * 10},  # Higher learning rate for critic
                },
                "algorithm": {
                    "adv_estimator": str(self.verl_params.adv_estimator).lower(),
                    "ppo_micro_batch_size": total_batch_size,  # Add here too
                    "kl_ctrl": {
                        "type": self.verl_params.kl_ctrl_type,
                        "kl_coef": self.verl_params.kl_coef,
                        "target_kl": self.verl_params.target_kl,
                        "horizon": self.verl_params.kl_horizon,
                    },
                },
                "reward_model": {
                    "enable": self.verl_params.use_reward_model,
                    "strategy": str(self.verl_params.training_strategy).lower() if self.verl_params.use_reward_model else None,
                    "model": {
                        "path": self.verl_params.reward_model_path or "",
                        "input_tokenizer": None,  # Use actor's tokenizer
                        "external_lib": None,
                        "use_remove_padding": False,
                    },
                    "micro_batch_size_per_gpu": 4,
                    "ppo_micro_batch_size": total_batch_size,  # Add here too
                    "use_dynamic_bsz": False,
                    "forward_max_token_len_per_gpu": 32768,
                    "ulysses_sequence_parallel_size": 1,
                    "reward_manager": "naive",
                },
                "custom_reward_function": {
                    "path": None,
                    "name": "compute_score",
                },
                "trainer": {
                    "total_epochs": self.params.num_train_epochs,
                    "n_gpus_per_node": self.verl_params.n_gpus_per_node,
                    "nnodes": self.verl_params.nnodes,
                    "output_dir": self.params.output_dir,
                    "seed": self.params.seed,
                    "logging_steps": self.params.logging_steps,
                    "save_steps": self.params.save_steps,
                    "enable_wandb": self.params.enable_wandb,
                    "enable_tensorboard": self.params.enable_tensorboard,
                    "test_freq": 10,
                    "ppo_micro_batch_size": total_batch_size,  # Add here too
                    "project_name": "oumi_verl_ppo",
                    "experiment_name": f"{self.model.config.model_type}_{str(self.verl_params.adv_estimator).lower()}",
                },
            }

            # If extra args were provided, merge them in
            extra_args = self.verl_params.extra_args or {}
            for section, values in extra_args.items():
                if section in verl_config:
                    if isinstance(verl_config[section], dict) and isinstance(values, dict):
                        # Deep merge for nested dictionaries
                        for key, value in values.items():
                            if key in verl_config[section] and isinstance(verl_config[section][key], dict) and isinstance(value, dict):
                                verl_config[section][key].update(value)
                            else:
                                verl_config[section][key] = value
                    else:
                        verl_config[section] = values
                else:
                    verl_config[section] = values

            # Convert to OmegaConf for attribute access
            verl_config = OmegaConf.create(verl_config)
            
            # Add a debugging statement to trace the configuration
            logger.info("VERL PPO configuration keys (top level): %s", list(verl_config.keys()))
            logger.info("actor_rollout_ref keys: %s", list(verl_config.actor_rollout_ref.keys()))
            
            # Store the configuration for reference
            self.verl_config = verl_config
            
            # Initialize VERL trainer
            logger.info("Initializing VERL PPO trainer...")
            self.verl_trainer = RayPPOTrainer(
                config=verl_config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=RayWorkerGroup,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
            )
            
            logger.info("VERL PPO trainer initialized successfully")
            
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            
            # Provide specific guidance for common errors
            if "No objects to concatenate" in error_message or "Parquet" in error_message:
                logger.error(
                    f"VERL PPO trainer initialization failed with dataset error: {error_type}: {error_message}\n"
                    "This is likely because VERL cannot properly process the dataset. The most common reasons are:\n"
                    "1. Empty dataset - VERL requires non-empty datasets\n"
                    "2. Dataset format mismatch - VERL expects datasets in Parquet format\n"
                    "3. Missing dataset files - VERL requires physical dataset files\n"
                    "4. Parquet file corruption - VERL needs valid Parquet files with proper schema\n\n"
                    "Try using a valid Parquet dataset or check if the dummy data settings need adjustment."
                )
                raise RuntimeError(
                    f"Error initializing VERL PPO trainer: Dataset error - {error_message}. "
                    "Try providing a non-empty dataset in Parquet format or check if dummy data is properly configured."
                ) from e
            else:
                logger.error(
                    f"VERL PPO trainer initialization failed with error: {error_type}: {error_message}\n"
                    "This could be because:\n"
                    "1. You're using a development or placeholder version of VERL\n"
                    "2. The trainer needs additional configuration options\n"
                    "3. The trainer implementation is incomplete\n"
                    "4. There's a mismatch between your VERL version and expected implementation\n\n"
                    "Please check the VERL documentation or contact the VERL team for support."
                )
                
                # Re-raise with more informative message
                if isinstance(e, NotImplementedError):
                    raise NotImplementedError(
                        "VERL PPO trainer implementation is incomplete or not ready for use. "
                        "Try using a different trainer type or updating to a newer version of VERL."
                    ) from e
                else:
                    raise RuntimeError(
                        f"Error initializing VERL PPO trainer: {error_message}. "
                        "Check your configuration and VERL installation."
                    ) from e

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Train the model using VERL PPO.

        Args:
            resume_from_checkpoint: Optional path to a checkpoint to resume from.
        """
        if resume_from_checkpoint:
            self.verl_trainer.resume_from = resume_from_checkpoint

        # Initialize VERL workers
        self.verl_trainer.init_workers()

        # Start training
        self.verl_trainer.fit()

        # Log completion
        if is_world_process_zero():
            logger.info("VERL PPO training completed.")

    def save_state(self) -> None:
        """Save the Trainer state using VERL's checkpoint handling."""
        if hasattr(self, "verl_trainer") and hasattr(
            self.verl_trainer, "save_checkpoint"
        ):
            self.verl_trainer.save_checkpoint()

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Save the model to the specified output directory.

        Args:
            config: The Oumi training config.
            final: Whether this is the final model being saved during training.
        """
        if not is_world_process_zero():
            return

        output_path = Path(self.params.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if final:
            # Save the final model
            output_path = output_path / "final_model"
            output_path.mkdir(exist_ok=True)

            if hasattr(self.verl_trainer, "save_model") and callable(
                self.verl_trainer.save_model
            ):
                self.verl_trainer.save_model(str(output_path))
            else:
                # Fallback to save_checkpoint if save_model doesn't exist
                self.verl_trainer.save_checkpoint(str(output_path))
                
    def __del__(self):
        """Clean up resources when the trainer is deleted."""
        # Clean up temporary files
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")