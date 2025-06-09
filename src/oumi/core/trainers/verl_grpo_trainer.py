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
from oumi.utils.grpo_utils import (
    extract_prompt_images_completion_from_single_turn_conversation,
)

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


from oumi.core.configs import DatasetSplitParams, TrainingConfig
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger
from oumi.utils.verl_model_merger import FSDPModelMerger, ModelMergerConfig

# Dataset processing function type. This function takes the following arguments:
# 1. a dataset sample.
# 2. index of the sample.
# 3. data source name
# 4. split name (train, validation, etc.)
# Returns an example converted to verl format.
_DatasetProcessFn = Callable[[dict, int, str, str], dict]


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
        processor: Optional[BaseProcessor] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initializes the verl trainer.

        Args:
            processing_class: The tokenizer for the model.
            config: Training config.
            reward_funcs: List of reward functions to use.
            train_dataset: Training dataset.
            eval_dataset: Validation dataset. This is required by verl.
            processor: Optional processor for the dataset. Required for VLM-s.
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
        self._final_output_dir: Optional[Path] = (
            Path(self._oumi_config.training.output_dir).absolute().resolve()
            if self._oumi_config.training.output_dir
            else None
        )
        self._temp_output_dir: Optional[Path] = (
            self._final_output_dir / "verl_output" if self._final_output_dir else None
        )

        if not self._final_output_dir and config.training.save_final_model:
            raise ValueError(
                "Output directory must be specified when saving final model is enabled."
            )

        # TODO: OPE-1192 - Support multiple reward functions.
        if len(reward_funcs) > 1:
            raise ValueError("We only support up to one reward function.")
        self._reward_funcs = reward_funcs

        self._cache_dir: Path = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".cache" / "oumi" / "verl_datasets"
        )
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        # verl trainer uses private methods and properties of `transformers`
        # processor, so we need to pass the raw processor here.
        self._processor = processor.raw_processor if processor is not None else None
        # Detect what dataset post-processing function to use (if any).
        process_fn = self._detect_dataset_process_fn()
        # Generate files and set self._train_filepath and self._val_filepath.
        self._create_dataset_files(process_fn)
        self._setup_verl_trainer()

    def _detect_dataset_process_fn(
        self,
    ) -> Optional[_DatasetProcessFn]:
        """Returns a post-processing function to convert data to verl format.

        Examines dataset samples to determine what post-processing function to use.

        Returns:
            A post-processing function to convert data to verl format.
            If no post-processing is needed, returns `None`.
        """
        first_train_sample = next(iter(self._train_dataset))
        first_eval_sample = next(iter(self._eval_dataset))

        if not isinstance(first_train_sample, dict):
            raise ValueError(
                "Element type of training dataset must be a dictionary. "
                f"Got {type(first_train_sample)} instead."
            )
        if not isinstance(first_eval_sample, dict):
            raise ValueError(
                "Element type of validation dataset must be a dictionary. "
                f"Got {type(first_eval_sample)} instead."
            )

        # Detect datasets containing Conversation-s.
        if "conversation_json" in first_train_sample:
            if "conversation_json" not in first_eval_sample:
                raise ValueError(
                    "Training and validation datasets must both have the same key: "
                    "'conversation_json'."
                )
            try:
                # Check if the conversation_json is valid.
                _ = Conversation.from_json(first_train_sample["conversation_json"])
                _ = Conversation.from_json(first_eval_sample["conversation_json"])
            except Exception as e:
                raise ValueError(
                    "Invalid conversation_json in training or validation dataset."
                ) from e
            return VerlGrpoTrainer._create_verl_data_entry_from_single_turn_conversation
        return None

    @staticmethod
    def _get_data_source_name(params: DatasetSplitParams) -> str:
        """Returns the verl data source name."""
        dataset_names = list({ds.dataset_name for ds in params.datasets})
        if len(dataset_names) != 1:
            if len(dataset_names) > 1:
                raise ValueError(
                    f"Multiple dataset names found: {dataset_names}. "
                    f"Please specify a single dataset name."
                )
            else:
                raise ValueError(
                    "No dataset names found. Please check the dataset split parameters."
                )
        return dataset_names[0]

    @staticmethod
    def _extract_question_images_answer_from_single_turn_conversation(
        example: dict,
    ) -> tuple[str, list, str]:
        """Finds question, answer, and optional images in a single-turn conversation.

        Args:
            example: A dictionary containing the conversation JSON.

        Returns:
            A tuple containing the question, images, and answer.
            The list of images is empty for text-only conversations.
        """
        prompt, images, answer = (
            extract_prompt_images_completion_from_single_turn_conversation(example)
        )

        if len(images) > 0:
            # TODO: Generalize. This only works for QwenVL 2.5, which is the only
            # VLM supported by verl as of 2025-05-15.
            if not prompt.startswith("<image>"):
                prompt = "<image>" + prompt
        return (prompt, images, answer)

    @staticmethod
    def _create_verl_data_entry_from_single_turn_conversation(
        example: dict, idx: int, data_source: str, split: str
    ) -> dict:
        prompt, images, answer = (
            VerlGrpoTrainer._extract_question_images_answer_from_single_turn_conversation(
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
        self, process_fn: Optional[_DatasetProcessFn] = None
    ) -> None:
        """Creates dataset files for verl in Parquet format.

        The Parquet files are saved to the Oumi cache directory.

        Args:
            process_fn: Optional function to convert the dataset samples to verl format.
        """
        train_file = self._cache_dir / "train.parquet"
        train_dataset = self._train_dataset

        # Limit the max number of sub-processes to 8 to avoid overloading the system
        # with too many processes.
        # TODO: Make this configurable.
        num_proc = min(8, os.cpu_count() or 1)

        if process_fn is not None:
            train_data_source = self._get_data_source_name(self._oumi_config.data.train)
            train_dataset = train_dataset.map(
                function=lambda example, idx: process_fn(
                    example,
                    idx,
                    train_data_source,
                    "train",
                ),
                with_indices=True,
                num_proc=num_proc,
            )

        train_dataset.to_parquet(train_file)
        self._train_filepath = str(train_file)

        val_file = self._cache_dir / "val.parquet"
        eval_dataset = self._eval_dataset
        if process_fn is not None:
            validation_data_source = self._get_data_source_name(
                self._oumi_config.data.validation
            )
            eval_dataset = eval_dataset.map(
                function=lambda example, idx: process_fn(
                    example,
                    idx,
                    validation_data_source,
                    "validation",
                ),
                with_indices=True,
                num_proc=num_proc,
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
        config.trainer.default_local_dir = str(self._temp_output_dir or "")

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
        return config

    def _setup_verl_trainer(self):
        """Sets up verl's RayPPOTrainer."""
        if ray is None:
            raise RuntimeError(
                "ray is not installed. "
                "Please install it with 'pip install `oumi[gpu]`'."
            )
        self._verl_config = self._create_config()
        logger.info(f"verl config: {pformat(self._verl_config)}")

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
            processor=self._processor,
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
        if final:
            self._export_hf_model()

    def _export_hf_model(self) -> bool:
        """Exports the tuned model to HF format.

        This method is called after training is complete.

        Returns:
            True if the model is exported successfully, False otherwise.
        """
        if not (self._final_output_dir and self._temp_output_dir):
            return False
        final_dir = Path(self._final_output_dir)
        temp_dir = Path(self._temp_output_dir)
        all_checkpoint_dirs: list[Path] = [
            f.absolute()
            for f in temp_dir.iterdir()
            if f.is_dir()
            and f.name.startswith("global_step_")
            and (f / "actor").exists()
            and (f / "actor").is_dir()
        ]

        # Find sub-directory named `global_step_NNN` with the largest NNN.
        latest_checkpoint_step = -1
        latest_checkpoint_dir: Optional[Path] = None
        for d in all_checkpoint_dirs:
            step_str = str(d.name.removeprefix("global_step_"))
            try:
                step = int(step_str)
            except Exception as e:
                raise RuntimeError(f"Failed to extract step number from {d}") from e
            if step > latest_checkpoint_step:
                latest_checkpoint_dir = d
                latest_checkpoint_step = step

        if not latest_checkpoint_dir:
            logger.warning(f"No checkpoints found under {temp_dir}")
            return False

        logger.info(
            f"Merging and exporting model from '{latest_checkpoint_dir}' "
            f"to '{final_dir}' ..."
        )

        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            # TODO: Detect if tie-word-embedding is enabled, or add a config parameter.
            tie_word_embedding=False,
            local_dir=str(latest_checkpoint_dir / "actor"),
            hf_model_config_path=str(latest_checkpoint_dir / "actor" / "huggingface"),
            target_dir=str(final_dir),
        )
        merger = FSDPModelMerger(config)
        merger.merge_and_save()
        logger.info(f"Successfully exported model to '{final_dir}'!")
        return True
