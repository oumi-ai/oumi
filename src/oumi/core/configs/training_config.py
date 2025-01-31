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

        # Check values for model sequence length.
        if self.model.model_max_length and self.model.model_max_length > 0:
            max_seq_length_value = int(self.model.model_max_length)
            max_seq_length_key = None
            if self.training.trainer_type == TrainerType.TRL_SFT:
                max_seq_length_key = "max_seq_length"
            elif self.training.trainer_type == TrainerType.TRL_DPO:
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
