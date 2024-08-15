from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import torch
from omegaconf import MISSING

from lema.core.types.base_config import BaseConfig
from lema.core.types.params.data_params import DataParams, DatasetSplitParams
from lema.core.types.params.job_resources import JobResources, StorageMount
from lema.core.types.params.model_params import ModelParams
from lema.core.types.params.peft_params import PeftParams
from lema.core.types.params.training_params import (
    MixedPrecisionDtype,
    TrainerType,
    TrainingParams,
)
from lema.utils.logging import logger


class EvaluationFramework(Enum):
    """Enum representing the evaluation framework to use."""

    LEMA = "lema"
    LM_HARNESS = "lm_harness"


@dataclass
class TrainingConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    peft: PeftParams = field(default_factory=PeftParams)

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
            if self.model.torch_dtype() != torch.float32:
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
                    f"Ignored model.model_max_length={max_seq_length_value} config "
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


@dataclass
class GenerationConfig(BaseConfig):
    # TODO: Add more parameters to control text generation.
    max_new_tokens: int = 256
    batch_size: int = 2
    input_filepath: Optional[str] = None
    output_filepath: Optional[str] = None


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class EvaluationConfig(BaseConfig):
    data: DatasetSplitParams = field(default_factory=DatasetSplitParams)
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation_framework: EvaluationFramework = EvaluationFramework.LM_HARNESS
    #: Number of few-shot examples (with responses) to add in the prompt, in order to
    #: teach the model how to respond to the specific dataset's prompts.
    #: If not set (None): LM Harness will decide the value.
    #: If set to 0: no few-shot examples will be added in the prompt.
    num_shots: Optional[int] = None
    #: Number of samples/examples to evaluate from this dataset. Mostly for debugging,
    #: in order to reduce the runtime. If not set (None): the entire dataset is
    #: evaluated. If set, this must be a positive integer.
    num_samples: Optional[int] = None
    #: Where to write computed evaluations.
    output_dir: str = "output"

    def __post_init__(self):
        """Verifies params."""
        if not isinstance(self.evaluation_framework, EvaluationFramework):
            raise ValueError(
                "`evaluation_framework` must belong to class `EvaluationFramework`."
            )
        if self.evaluation_framework not in list(EvaluationFramework):
            raise ValueError(
                f"Unknown `evaluation_framework` value: {self.evaluation_framework}."
            )
        if self.num_shots and self.num_shots < 0:
            raise ValueError("`num_shots` must be non-negative.")
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("`num_samples` must be None or a positive integer.")


@dataclass
class AsyncEvaluationConfig(BaseConfig):
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    #: The directory to poll for new checkpoints.
    checkpoints_dir: str = MISSING

    #: The time in seconds between the end of the previous evaluation and the start of
    #: the next polling attempt. Cannot be negative.
    polling_interval: float = MISSING

    #: The number of times to retry polling before exiting the current job.
    #: A retry occurs when the job reads the target directory but cannot find a new
    #: model checkpoint to evaluate. Defaults to 5. Cannot be negative.
    num_retries: int = 5

    def __post_init__(self):
        """Verifies/populates params."""
        if self.polling_interval < 0:
            raise ValueError("`polling_interval` must be non-negative.")
        if self.num_retries < 0:
            raise ValueError("`num_retries` must be non-negative.")


@dataclass
class JobConfig(BaseConfig):
    """Configuration for launching jobs on a cluster."""

    #: Job name (optional). Only used for display purposes.
    name: Optional[str] = None

    #: The user that the job will run as (optional). Required only for Polaris.
    user: Optional[str] = None

    #: The local directory containing the scripts required to execute this job.
    #: This directory will be copied to the remote node before the job is executed.
    working_dir: str = MISSING

    #: The number of nodes to use for the job. Defaults to 1.
    num_nodes: int = 1

    #: The resources required for each node in the job.
    resources: JobResources = field(default_factory=JobResources)

    #: The environment variables to set on the node.
    envs: Dict[str, str] = field(default_factory=dict)

    #: File mounts to attach to the node.
    #: For mounting (copying) local directories, the key is the file path on the remote
    #: and the value is the local path.
    #: The keys of `file_mounts` cannot be shared with `storage_mounts`.
    file_mounts: Dict[str, str] = field(default_factory=dict)

    #: Storage system mounts to attach to the node.
    #: For mounting remote storage solutions, the key is the file path on the remote
    #: and the value is a StorageMount.
    #: The keys of `storage_mounts` cannot be shared with `file_mounts`.
    storage_mounts: Dict[str, StorageMount] = field(default_factory=dict)

    #: The setup script to run on every node. Optional.
    #: `setup` will always be executed before `run`.
    #: ex) pip install -r requirements.txt
    setup: Optional[str] = None

    #: The script to run on every node. Required. Runs after `setup`.
    run: str = MISSING
