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
from typing import Any, Final

import torch

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.data_params import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
)
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
    """Configuration for model training.

    Example:
        Creating config from a YAML file::

            config = TrainingConfig.from_yaml("my_config.yaml")

        Creating config programmatically::

            config = TrainingConfig(
                model=ModelParams(model_name="gpt2"),
                data=DataParams(...),
                training=TrainingParams(output_dir="./output"),
            )

        Using the convenience constructor::

            config = TrainingConfig.for_model(
                "meta-llama/Llama-3.2-1B",
                dataset="tatsu-lab/alpaca",
                output_dir="./output",
            )
    """
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
    def for_model(
        cls,
        model_name: str,
        dataset: str | list[str],
        output_dir: str,
        *,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        torch_dtype: str = "auto",
        trainer_type: TrainerType = TrainerType.TRL_SFT,
        use_peft: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        dataset_split: str = "train",
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "TrainingConfig":
        """Create a TrainingConfig with sensible defaults for a specific model.

        This is a convenience factory method for quickly setting up training
        without manually constructing nested configuration objects.

        Args:
            model_name: Name or path of the model (HuggingFace model ID or local path).
            dataset: Dataset name(s) to use for training. Can be a single string or
                a list of dataset names.
            output_dir: Directory to save model checkpoints and outputs.
            epochs: Number of training epochs.
            batch_size: Per-device training batch size.
            learning_rate: Learning rate for the optimizer.
            torch_dtype: Data type for model parameters ("auto", "float16", "bfloat16").
            trainer_type: Type of trainer to use (default: TRL_SFT).
            use_peft: Whether to use PEFT/LoRA for parameter-efficient fine-tuning.
            lora_r: LoRA rank (only used if use_peft=True).
            lora_alpha: LoRA alpha scaling factor (only used if use_peft=True).
            dataset_split: Which split of the dataset to use for training.
            trust_remote_code: Whether to trust remote code when loading model/dataset.
            **kwargs: Additional keyword arguments for advanced configuration.

        Returns:
            A configured TrainingConfig instance.

        Example:
            Basic SFT training::

                config = TrainingConfig.for_model(
                    "meta-llama/Llama-3.2-1B",
                    dataset="tatsu-lab/alpaca",
                    output_dir="./output",
                )

            Training with LoRA::

                config = TrainingConfig.for_model(
                    "meta-llama/Llama-3.2-1B",
                    dataset="tatsu-lab/alpaca",
                    output_dir="./output",
                    use_peft=True,
                    lora_r=16,
                )

            Training with multiple datasets::

                config = TrainingConfig.for_model(
                    "gpt2",
                    dataset=["dataset1", "dataset2"],
                    output_dir="./output",
                    epochs=5,
                )
        """
        # Build dataset params
        if isinstance(dataset, str):
            dataset_list = [dataset]
        else:
            dataset_list = list(dataset)

        datasets = [
            DatasetParams(
                dataset_name=ds_name,
                split=dataset_split,
                trust_remote_code=trust_remote_code,
            )
            for ds_name in dataset_list
        ]

        data_params = DataParams(
            train=DatasetSplitParams(datasets=datasets),
        )

        # Build model params
        model_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ModelParams.__dataclass_fields__
        }
        model_params = ModelParams(
            model_name=model_name,
            torch_dtype_str=torch_dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        # Build training params
        training_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in TrainingParams.__dataclass_fields__
        }
        training_params = TrainingParams(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            trainer_type=trainer_type,
            use_peft=use_peft,
            **training_kwargs,
        )

        # Build PEFT params if using LoRA
        peft_params = PeftParams()
        if use_peft:
            peft_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in PeftParams.__dataclass_fields__
            }
            peft_params = PeftParams(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                **peft_kwargs,
            )

        return cls(
            model=model_params,
            data=data_params,
            training=training_params,
            peft=peft_params,
        )
