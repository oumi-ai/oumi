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

        # Verify values for model dtype and mixed precision training.
        if self.training.mixed_precision_dtype in [
            MixedPrecisionDtype.FP16,
            MixedPrecisionDtype.BF16,
        ]:
            if self.model.torch_dtype != torch.float32:
                raise ValueError(
                    "Model must be loaded in fp32 to enable mixed precision training."
                )

        trainer_type: Final[TrainerType] = self.training.trainer_type

        # Check values for model sequence length.
        if self.model.model_max_length and self.model.model_max_length > 0:
            max_seq_length_value = int(self.model.model_max_length)
            max_seq_length_key = None
            if trainer_type == TrainerType.TRL_SFT:
                max_seq_length_key = "max_seq_length"
            elif trainer_type == TrainerType.TRL_DPO:
                max_seq_length_key = "max_length"
                # TODO: DPOTrainer also defines "max_prompt_length" and
                # "max_target_length". How to handle them?
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
        # TODO(OPE-1117): Clean up this logic after upgrading to trl 0.16.
        if self.model.enable_liger_kernel:
            if trainer_type == TrainerType.TRL_SFT:
                self.training.trainer_kwargs["use_liger"] = True
                self.training.trainer_kwargs["use_liger_kernel"] = True
                self.model.enable_liger_kernel = False
            elif trainer_type in (TrainerType.TRL_DPO, TrainerType.HF):
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
