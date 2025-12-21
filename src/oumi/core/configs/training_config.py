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

import itertools
from dataclasses import dataclass, field
from typing import Final

import torch

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.data_params import DataParams
from oumi.core.configs.params.deepspeed_params import DeepSpeedParams
from oumi.core.configs.params.fsdp_params import FSDPParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.peft_params import PeftParams
from oumi.core.configs.params.training_params import (
    MixedPrecisionDtype,
    TrainerType,
    TrainingParams,
)
from oumi.utils.logging import logger


@dataclass
class TrainingConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    """Parameters for the dataset.

    This field contains all the necessary settings for data processing and loading.
    It includes options for train and evaluation datasets and preprocessing steps.

    For more details, see the :class:`oumi.core.configs.params.data_params.DataParams`
    class.
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model.

    This field defines the model architecture, size, and other model-specific settings.
    It includes options for model type, pretrained weights, and tokenizer configuration.

    For more details, see :class:`oumi.core.configs.params.model_params.ModelParams`
    class.
    """

    training: TrainingParams = field(default_factory=TrainingParams)
    """Parameters for the training process.

    This field contains all settings related to the training loop,
    including learning rate, batch size, number of epochs, and optimization parameters.

    For more details, see
    :class:`oumi.core.configs.params.training_params.TrainingParams`.
    """

    peft: PeftParams = field(default_factory=PeftParams)
    """Parameters for Parameter-Efficient Fine-Tuning (PEFT).

    This field defines settings for various PEFT methods such as LoRA, or Prefix Tuning.
    It includes options for rank, alpha values, and other PEFT-specific parameters.

    For more details, see :class:`oumi.core.configs.params.peft_params.PeftParams`.
    """

    fsdp: FSDPParams = field(default_factory=FSDPParams)
    """Parameters for FSDP."""

    deepspeed: DeepSpeedParams = field(default_factory=DeepSpeedParams)
    """Parameters for DeepSpeed distributed training.

    This field contains configuration options for DeepSpeed ZeRO optimization
    stages, memory offloading, and other DeepSpeed-specific settings.

    For more details, see
    :class:`oumi.core.configs.params.deepspeed_params.DeepSpeedParams`.
    """

    def __post_init__(self):
        """Verifies/populates params."""
        if self.model.compile:
            raise ValueError(
                "Use `training.compile` instead of `model.compile` to "
                "enable model compilation during training."
            )
        if self.training.compile and (
            self.fsdp.use_orig_params is not None and not self.fsdp.use_orig_params
        ):
            raise ValueError(
                "`fsdp.use_orig_params` must be True for model compilation."
            )

        # Validate distributed training configurations
        if self.fsdp.enable_fsdp and self.deepspeed.enable_deepspeed:
            raise ValueError(
                "Cannot enable both FSDP and DeepSpeed simultaneously. "
                "Please enable only one distributed training method."
            )

        # Validate DeepSpeed batch size configuration for TRL trainers
        trainer_type: Final[TrainerType] = self.training.trainer_type
        if (
            self.deepspeed.enable_deepspeed
            and trainer_type
            in (
                TrainerType.TRL_SFT,
                TrainerType.TRL_DPO,
                TrainerType.TRL_KTO,
                TrainerType.TRL_GRPO,
                TrainerType.TRL_GKD,
                TrainerType.TRL_GOLD,
            )
            and self.deepspeed.train_batch_size != "auto"
        ):
            raise ValueError(
                f"When using TRL trainer ({trainer_type}) with DeepSpeed, "
                "train_batch_size must be set to 'auto' to allow proper batch size "
                "management. "
                f"Current value: {self.deepspeed.train_batch_size}"
            )

        # Verify values for model dtype and mixed precision training.
        if self.training.mixed_precision_dtype in [
            MixedPrecisionDtype.FP16,
            MixedPrecisionDtype.BF16,
        ]:
            if self.model.torch_dtype != torch.float32:
                raise ValueError(
                    "Model must be loaded in fp32 to enable mixed precision training."
                )

        # Check values for model sequence length.
        if self.model.model_max_length and self.model.model_max_length > 0:
            max_seq_length_value = int(self.model.model_max_length)
            max_seq_length_key = None
            if trainer_type in (
                TrainerType.TRL_SFT,
                TrainerType.TRL_DPO,
                TrainerType.TRL_GKD,
                TrainerType.TRL_GOLD,
            ):
                # TODO: DPOTrainer also defines "max_prompt_length" and
                # "max_target_length". How to handle them?
                max_seq_length_key = "max_length"
            else:
                logger.warning(
                    f"Ignored model.model_max_length={max_seq_length_value} "
                    f"parameter for trainer {self.training.trainer_type}."
                )

            if max_seq_length_key:
                existing_max_seq_length = self.training.trainer_kwargs.get(
                    max_seq_length_key
                )
                if (existing_max_seq_length is not None) and (
                    existing_max_seq_length != max_seq_length_value
                ):
                    logger.warning(
                        f"Overriding existing '{max_seq_length_key}' value "
                        f"'{existing_max_seq_length}' with '{max_seq_length_value}'"
                    )
                self.training.trainer_kwargs[max_seq_length_key] = max_seq_length_value

        # Set Liger kernel flags if using a HF trainer, and if so, don't do Liger
        # patch ourselves.
        if self.model.enable_liger_kernel:
            if trainer_type in (
                TrainerType.TRL_SFT,
                TrainerType.TRL_DPO,
                TrainerType.TRL_GRPO,
                TrainerType.TRL_GKD,
                TrainerType.TRL_GOLD,
                TrainerType.HF,
            ):
                self.training.trainer_kwargs["use_liger_kernel"] = True
                self.model.enable_liger_kernel = False
            elif trainer_type == TrainerType.OUMI:
                # We need to Liger patch ourselves for our own training loop.
                pass
            else:
                raise ValueError("Unrecognized trainer type!")

        # Setup and validate params for "vision_language_sft" collator.
        # The collator expects VLM SFT dataset to only produce just
        # one column: 'conversation_json' (JSON-encoded `Conversation`)!
        collator_name: Final[str] = self.data.train.collator_name or ""
        if collator_name == "vision_language_sft":
            for dataset_params in itertools.chain(
                self.data.train.datasets,
                self.data.validation.datasets,
                self.data.test.datasets,
            ):
                if not dataset_params.dataset_kwargs.get("return_conversations", True):
                    raise ValueError(
                        "`return_conversations` must be True "
                        f"for the dataset '{dataset_params.dataset_name}' "
                        f"when using '{collator_name}' collator!"
                    )
                dataset_params.dataset_kwargs["return_conversations"] = True
            # Extra setup for TRL_SFT.
            if trainer_type == TrainerType.TRL_SFT:
                if self.training.trainer_kwargs.get("remove_unused_columns", False):
                    raise ValueError(
                        "`remove_unused_columns` must be False "
                        f"when using '{collator_name}' collator! "
                        'The "unused" columns are consumed by the collator, '
                        "not by a model."
                    )
                self.training.trainer_kwargs["remove_unused_columns"] = False

                # `trl` shouldn't be preparing the dataset, as we do it in Oumi.
                dataset_kwargs = self.training.trainer_kwargs.get("dataset_kwargs", {})
                dataset_kwargs["skip_prepare_dataset"] = True
                self.training.trainer_kwargs["dataset_kwargs"] = dataset_kwargs

        if len(self.model.processor_kwargs) > 0:
            model_processor_name: Final[str] = (
                self.model.tokenizer_name or self.model.model_name
            )
            for dataset_params in itertools.chain(
                self.data.train.datasets,
                self.data.validation.datasets,
                self.data.test.datasets,
            ):
                if (
                    "processor_name" not in dataset_params.dataset_kwargs
                    or "processor_kwargs" in dataset_params.dataset_kwargs
                ):
                    continue
                dataset_processor_name: str = dataset_params.dataset_kwargs[
                    "processor_name"
                ]
                if dataset_processor_name == model_processor_name:
                    # Copy processor kwargs from the model if processor names match
                    # and the dataset doesn't override them.
                    dataset_params.dataset_kwargs["processor_kwargs"] = {
                        **self.model.processor_kwargs
                    }

        # Verl will error without a validation dataset.
        if (
            self.training.trainer_type == TrainerType.VERL_GRPO
            and not self.data.validation.datasets
        ):
            raise ValueError(
                "At least one validation dataset is required for VERL_GRPO training."
            )

    @classmethod
    def for_method(
        cls,
        method: str,
        model: str,
        dataset: str,
        output_dir: str = "./output",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        use_peft: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        save_steps: int = 500,
        logging_steps: int = 10,
        **kwargs,
    ) -> "TrainingConfig":
        """Create a TrainingConfig for a specific training method with sensible defaults.

        This factory method provides a simplified way to create training configurations
        without manually constructing all the nested parameter objects.

        Args:
            method: Training method - "sft", "dpo", "kto", "grpo", "gkd", "gold".
            model: HuggingFace model name or path (e.g., "meta-llama/Llama-3.1-8B").
            dataset: Dataset name or path (e.g., "tatsu-lab/alpaca").
            output_dir: Output directory for checkpoints and logs.
            epochs: Number of training epochs.
            batch_size: Per-device batch size.
            learning_rate: Learning rate.
            use_peft: Whether to use LoRA/PEFT for efficient fine-tuning.
            lora_r: LoRA rank (only used if use_peft=True).
            lora_alpha: LoRA alpha (only used if use_peft=True).
            gradient_accumulation_steps: Gradient accumulation steps.
            warmup_ratio: Warmup ratio for learning rate scheduler.
            save_steps: Save checkpoint every N steps.
            logging_steps: Log metrics every N steps.
            **kwargs: Additional overrides for any config parameter.

        Returns:
            A configured TrainingConfig instance.

        Raises:
            ValueError: If the training method is unknown.

        Examples:
            Create SFT config:
                >>> config = TrainingConfig.for_method(
                ...     method="sft",
                ...     model="meta-llama/Llama-3.1-8B",
                ...     dataset="tatsu-lab/alpaca",
                ... )

            Create DPO config with custom settings:
                >>> config = TrainingConfig.for_method(
                ...     method="dpo",
                ...     model="meta-llama/Llama-3.1-8B",
                ...     dataset="my-preference-dataset",
                ...     learning_rate=1e-5,
                ...     epochs=1,
                ... )
        """
        from oumi.core.configs.params.data_params import (
            DataParams,
            DatasetParams,
            DatasetSplitParams,
        )

        # Map method string to trainer type
        trainer_type_map = {
            "sft": TrainerType.TRL_SFT,
            "dpo": TrainerType.TRL_DPO,
            "kto": TrainerType.TRL_KTO,
            "grpo": TrainerType.TRL_GRPO,
            "gkd": TrainerType.TRL_GKD,
            "gold": TrainerType.TRL_GOLD,
        }

        trainer_type = trainer_type_map.get(method.lower())
        if trainer_type is None:
            raise ValueError(
                f"Unknown training method: '{method}'. "
                f"Supported methods: {list(trainer_type_map.keys())}"
            )

        # Build data params
        data_params = DataParams(
            train=DatasetSplitParams(datasets=[DatasetParams(dataset_name=dataset)])
        )

        # Build model params
        model_params = ModelParams(model_name=model)

        # Build training params
        training_params = TrainingParams(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            logging_steps=logging_steps,
            trainer_type=trainer_type,
            use_peft=use_peft,
        )

        # Build PEFT params if using LoRA
        peft_params = PeftParams(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        ) if use_peft else PeftParams()

        return cls(
            data=data_params,
            model=model_params,
            training=training_params,
            peft=peft_params,
            **kwargs,
        )

    @classmethod
    def for_sft(
        cls,
        model: str,
        dataset: str,
        **kwargs,
    ) -> "TrainingConfig":
        """Create a TrainingConfig for SFT (Supervised Fine-Tuning).

        Convenience method equivalent to for_method(method="sft", ...).

        Args:
            model: HuggingFace model name or path.
            dataset: Dataset name or path.
            **kwargs: Additional parameters passed to for_method().

        Returns:
            A configured TrainingConfig for SFT.

        Examples:
            >>> config = TrainingConfig.for_sft(
            ...     model="meta-llama/Llama-3.1-8B",
            ...     dataset="tatsu-lab/alpaca",
            ... )
        """
        return cls.for_method(method="sft", model=model, dataset=dataset, **kwargs)

    @classmethod
    def for_dpo(
        cls,
        model: str,
        dataset: str,
        **kwargs,
    ) -> "TrainingConfig":
        """Create a TrainingConfig for DPO (Direct Preference Optimization).

        Convenience method equivalent to for_method(method="dpo", ...).

        Args:
            model: HuggingFace model name or path.
            dataset: Dataset name or path (should be a preference dataset).
            **kwargs: Additional parameters passed to for_method().

        Returns:
            A configured TrainingConfig for DPO.

        Examples:
            >>> config = TrainingConfig.for_dpo(
            ...     model="meta-llama/Llama-3.1-8B",
            ...     dataset="my-preference-dataset",
            ... )
        """
        return cls.for_method(method="dpo", model=model, dataset=dataset, **kwargs)

    @classmethod
    def for_grpo(
        cls,
        model: str,
        dataset: str,
        **kwargs,
    ) -> "TrainingConfig":
        """Create a TrainingConfig for GRPO (Group Relative Policy Optimization).

        Convenience method equivalent to for_method(method="grpo", ...).

        Args:
            model: HuggingFace model name or path.
            dataset: Dataset name or path.
            **kwargs: Additional parameters passed to for_method().

        Returns:
            A configured TrainingConfig for GRPO.

        Examples:
            >>> config = TrainingConfig.for_grpo(
            ...     model="meta-llama/Llama-3.1-8B",
            ...     dataset="my-grpo-dataset",
            ... )
        """
        return cls.for_method(method="grpo", model=model, dataset=dataset, **kwargs)
