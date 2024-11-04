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

    def get_accelerate_env_vars(self) -> dict[str, str]:
        """Gets environment variables for HF Accelerate.

        `training.enable_gradient_checkpointing` needs to be disabled if setting these
        environment variables, as FSDP gradient checkpointing is handled by Accelerate.

        This mimics the environment variables set here:
        https://github.com/huggingface/accelerate/blob/bf4572b6ce0a534a9d73537485a0edf1d68144b8/src/accelerate/utils/launch.py#L260-L285
        Note how they lowercase all boolean values, except for
        `ACCELERATE_DYNAMO_USE_FULLGRAPH` and `ACCELERATE_DYNAMO_USE_DYNAMIC`, which we
        also do. It's worth pointing out that `ACCELERATE_USE_FSDP` must be lowercase:
        https://github.com/huggingface/accelerate/blob/bf4572b6ce0a534a9d73537485a0edf1d68144b8/src/accelerate/accelerator.py#L341
        """
        env_vars = {}

        # These environment variables are set by default in HF Accelerate.
        env_vars["ACCELERATE_DYNAMO_BACKEND"] = "NO"
        env_vars["ACCELERATE_DYNAMO_MODE"] = "default"
        env_vars["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = "False"
        env_vars["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "False"

        # We generally don't need these values to be configurable, and usually have
        # them set to True.
        env_vars["FSDP_USE_ORIG_PARAMS"] = "True"
        # https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/modeling_utils.py#L146
        env_vars["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"

        # These env vars are set based on FSDPParams.
        env_vars["ACCELERATE_USE_FSDP"] = str(self.fsdp.enable_fsdp).lower()
        env_vars["FSDP_SHARDING_STRATEGY"] = self.fsdp.sharding_strategy.value
        env_vars["FSDP_OFFLOAD_PARAMS"] = str(self.fsdp.cpu_offload).lower()
        if self.fsdp.mixed_precision:
            env_vars["ACCELERATE_MIXED_PRECISION"] = self.fsdp.mixed_precision
        env_vars["FSDP_BACKWARD_PREFETCH"] = self.fsdp.backward_prefetch.value
        env_vars["FSDP_FORWARD_PREFETCH"] = str(self.fsdp.forward_prefetch).lower()
        env_vars["FSDP_STATE_DICT_TYPE"] = self.fsdp.state_dict_type.value
        env_vars["FSDP_AUTO_WRAP_POLICY"] = self.fsdp.auto_wrap_policy.value
        env_vars["FSDP_MIN_NUM_PARAMS"] = str(self.fsdp.min_num_params)
        if self.fsdp.transformer_layer_cls:
            env_vars["FSDP_TRANSFORMER_CLS_TO_WRAP"] = self.fsdp.transformer_layer_cls
        env_vars["FSDP_SYNC_MODULE_STATES"] = str(self.fsdp.sync_module_states).lower()

        # This is set from TrainingParams.
        env_vars["FSDP_ACTIVATION_CHECKPOINTING"] = str(
            self.training.enable_gradient_checkpointing
        ).lower()
        return env_vars

    def __post_init__(self):
        """Verifies/populates params."""
        if self.model.compile:
            raise ValueError(
                "Use `training.compile` instead of `model.compile` to "
                "enable model compilation during training."
            )

        # Verify dataset-related params for TRL_SFT.
        if self.training.trainer_type == TrainerType.TRL_SFT:
            if not self.data.train.target_col:
                raise ValueError("`target_col` must be specified for TRL_SFT Trainer.")

            # Set `dataset_text_field` in `trainer_kwargs` since it's required for
            # `SFTTrainer`, and warn users if their value will be overridden.
            existing_dataset_text_field = self.training.trainer_kwargs.get(
                "dataset_text_field"
            )
            if (existing_dataset_text_field is not None) and (
                existing_dataset_text_field != self.data.train.target_col
            ):
                logger.warning(
                    "Overriding existing `dataset_text_field` value "
                    f"'{existing_dataset_text_field}' with "
                    f"'{self.data.train.target_col}'"
                )
            self.training.trainer_kwargs["dataset_text_field"] = (
                self.data.train.target_col
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
