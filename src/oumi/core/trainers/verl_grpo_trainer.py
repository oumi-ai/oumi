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

"""Volcano Engine Reinforcement Learning (verl) GRPO Trainer."""

import copy
import os
from pathlib import Path
from pprint import pformat
from typing import Callable, Optional, Union, cast

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from oumi.core.types.conversation import Conversation
from oumi.core.types.conversation import Role as ConversationRole

try:
    import ray  # pyright: ignore[reportMissingImports]
    import verl  # pyright: ignore[reportMissingImports]
    from verl.trainer.ppo.ray_trainer import (  # pyright: ignore[reportMissingImports]
        RayPPOTrainer,
        ResourcePoolManager,
        Role,
    )
    from verl.workers.fsdp_workers import (  # pyright: ignore[reportMissingImports]
        ActorRolloutRefWorker,
        CriticWorker,
    )
    from verl.workers.reward_manager import (  # pyright: ignore[reportMissingImports]
        NaiveRewardManager,
    )
except ModuleNotFoundError:
    verl = None
    ray = None


from oumi.core.configs import TrainingConfig
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


class VerlGrpoTrainer(BaseTrainer):
    """verl GRPO Trainer.

    This class wraps verl's RayPPOTrainer. This class' name is misleading as it supports
    other RL algorithms as well, including GRPO, which we use here.

    For documentation on the underlying verl RayPPOTrainer, see
    https://verl.readthedocs.io/en/latest/examples/config.html.
    """

    def __init__(
        self,
        processing_class: Optional[BaseTokenizer],
        config: TrainingConfig,
        reward_funcs: list[Callable],
        train_dataset: Dataset,
        eval_dataset: Dataset,
        cache_dir: Union[str, Path] = Path.home() / ".cache" / "oumi" / "verl_datasets",
        **kwargs,
    ):
        """Initializes the verl trainer.

        Args:
            processing_class: The tokenizer for the model.
            config: Training config.
            reward_funcs: List of reward functions to use.
            train_dataset: Training dataset.
            eval_dataset: Validation dataset. This is required by verl.
            cache_dir: Directory to cache verl Parquet datasets.
            **kwargs: Additional keyword arguments.
        """
        if verl is None:
            raise RuntimeError(
                "verl is not installed. "
                "Please install it with 'pip install `oumi[gpu]`'."
            )
        logger.warning(
            "VerlGrpoTrainer is experimental, and the interface is subject to change."
        )
        self._processing_class = processing_class
        self._oumi_config = copy.deepcopy(config)
        # TODO: OPE-1192 - Support multiple reward functions.
        if len(reward_funcs) > 1:
            raise ValueError("We only support up to one reward function.")
        self._reward_funcs = reward_funcs

        self._cache_dir = Path(cache_dir)
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        # Sets self._train_filepath and self._val_filepath.
        self._create_dataset_files(
            VerlGrpoTrainer._create_verl_data_entry_from_simple_conversation
        )

        self._setup_verl_trainer()

    @staticmethod
    def _extract_question_images_answer_from_simple_conversation(
        example: dict,
    ) -> tuple[str, list, str]:
        if "conversation_json" not in example:
            raise ValueError(
                f"Example doesn't contain 'conversation_json' key. "
                f"Available keys: {example.keys()}"
            )

        conversation_json = example["conversation_json"]
        conversation = Conversation.from_json(conversation_json)

        user_messages = conversation.filter_messages(role=ConversationRole.USER)
        if len(user_messages) != 1:
            raise ValueError(f"Expected 1 user message, but got {len(user_messages)}.")

        assistant_messages = conversation.filter_messages(
            role=ConversationRole.ASSISTANT
        )
        if len(assistant_messages) != 1:
            raise ValueError(
                f"Expected 1 assistant message, but got {len(assistant_messages)}."
            )

        user_message = user_messages[0]
        assistant_message = assistant_messages[0]

        prompt: str = user_message.text_content_items[-1].content or ""
        images = [{"bytes": item.binary} for item in user_message.image_content_items]
        answer: str = assistant_message.text_content_items[-1].content or ""
        return (prompt, images, answer)

    @staticmethod
    def _create_verl_data_entry_from_simple_conversation(
        example: dict, idx: int, data_source: str, split: str
    ) -> dict:
        prompt, images, answer = (
            VerlGrpoTrainer._extract_question_images_answer_from_simple_conversation(
                example
            )
        )
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "images": images,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": prompt,  # TODO: extract problem
            },
        }
        return data

    def _create_dataset_files(
        self, process_fn: Optional[Callable[[dict, int, str, str], dict]] = None
    ) -> None:
        """Creates dataset files for verl in Parquet format.

        The Parquet files are saved to the Oumi cache directory.
        """
        data_source: str = "hiyouga/geometry3k"
        train_file = self._cache_dir / "train.parquet"
        train_dataset = self._train_dataset
        if process_fn is not None:
            train_dataset = train_dataset.map(
                function=lambda example, idx: process_fn(
                    example, idx, data_source, "train"
                ),
                with_indices=True,
                num_proc=8,
            )

        train_dataset.to_parquet(train_file)
        self._train_filepath = str(train_file)

        val_file = self._cache_dir / "val.parquet"
        eval_dataset = self._eval_dataset
        if process_fn is not None:
            eval_dataset = eval_dataset.map(
                function=lambda example, idx: process_fn(
                    example, idx, data_source, "validation"
                ),
                with_indices=True,
                num_proc=8,
            )
        eval_dataset.to_parquet(val_file)
        self._val_filepath = str(val_file)

    def _create_config(self) -> DictConfig:
        """Creates a verl config."""
        model_params = self._oumi_config.model
        model_name = model_params.model_name

        # 1. Read verl default dict config from YAML.
        yaml_path = Path(__file__).parent / "verl_trainer_config.yaml"
        config = OmegaConf.load(yaml_path)
        config = cast(DictConfig, config)

        # 2. Set config values, ex. from Oumi config values
        config.algorithm.adv_estimator = "grpo"
        config.data.train_files = self._train_filepath
        config.data.val_files = self._val_filepath

        ### NEW
        config.data.tokenizer = model_name
        config.critic.model.path = model_name
        config.reward_model.model.path = model_name
        ###

        grpo_params = self._oumi_config.training.grpo
        training_params = self._oumi_config.training

        config.data.max_response_length = grpo_params.max_completion_length
        config.actor_rollout_ref.model.path = model_name
        config.actor_rollout_ref.actor.optim.lr = training_params.learning_rate
        config.actor_rollout_ref.model.enable_gradient_checkpointing = (
            training_params.enable_gradient_checkpointing
        )
        if grpo_params.use_vllm:
            config.actor_rollout_ref.rollout.name = "vllm"
        else:
            config.actor_rollout_ref.rollout.name = "hf"
        config.actor_rollout_ref.rollout.temperature = grpo_params.temperature
        config.actor_rollout_ref.rollout.gpu_memory_utilization = (
            grpo_params.vllm_gpu_memory_utilization
        )

        # Normally, training steps is determined by the number of epochs.
        # If max_steps is set, it will override this.
        config.trainer.total_epochs = training_params.num_train_epochs
        if training_params.max_steps != -1:
            config.trainer.total_training_steps = training_params.max_steps

        if training_params.eval_strategy == "steps":
            config.trainer.test_freq = training_params.eval_steps
        if not training_params.save_epoch:
            config.trainer.save_freq = training_params.save_steps

        config.trainer.logger = []
        if training_params.logging_strategy != "no":
            config.trainer.logger.append("console")
        if training_params.enable_wandb:
            config.trainer.logger.append("wandb")
        config.trainer.project_name = os.environ.get("WANDB_PROJECT", "oumi_verl")
        config.trainer.experiment_name = training_params.run_name
        config.trainer.default_local_dir = training_params.output_dir

        # 3. Apply user overrides
        overrides_config = OmegaConf.create(training_params.verl_config_overrides)
        config = cast(DictConfig, OmegaConf.merge(config, overrides_config))

        # 4. Finalize and validate config.

        # Resolves the value of all interpolation fields in the config.
        # ex. `prompt_length: ${data.max_prompt_length}`
        # https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#omegaconf-resolve
        OmegaConf.resolve(config)

        if (
            config.actor_rollout_ref.actor.strategy == "fsdp"
            and config.actor_rollout_ref.actor.strategy != config.critic.strategy
        ):
            raise ValueError(
                "Actor and critic must use the same strategy when using FSDP."
            )
        logger.info(f"VERL config:\n\n{pformat(config)}\n\n")
        return config

    def _setup_verl_trainer(self):
        """Sets up verl's RayPPOTrainer."""
        if ray is None:
            raise RuntimeError(
                "ray is not installed. "
                "Please install it with 'pip install `oumi[gpu]`'."
            )
        self._verl_config = self._create_config()
        logger.info(f"verl config: {self._verl_config}")

        tokenizer = self._processing_class

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        # Create resource pool manager
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [self._verl_config.trainer.n_gpus_per_node]
            * self._verl_config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Create reward function manager
        compute_score = self._reward_funcs[0] if len(self._reward_funcs) > 0 else None
        reward_fn = NaiveRewardManager(
            tokenizer=tokenizer, num_examine=0, compute_score=compute_score
        )
        # num_examine=1 means to print 1 example per batch for analysis.
        val_reward_fn = NaiveRewardManager(
            tokenizer=tokenizer, num_examine=1, compute_score=compute_score
        )

        self._verl_trainer = RayPPOTrainer(
            config=self._verl_config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Trains the model using verl's RayPPOTrainer.

        Args:
            resume_from_checkpoint: Optional path to a checkpoint to resume from.
        """
        if resume_from_checkpoint:
            raise NotImplementedError("Resuming from checkpoint is not implemented.")

        logger.info("Initializing verl trainer workers...")
        self._verl_trainer.init_workers()
        logger.info("Starting verl training...")
        self._verl_trainer.fit()

    # TODO: OPE-1192 - Implement saving model/trainer state. verl training should
    # already handle saving models, including the final checkpoint.

    def save_state(self) -> None:
        """Saves the training state."""
        pass

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model.

        Args:
            config: The Oumi training config.
            final: Whether this is the final model being saved during training.
        """
        pass
