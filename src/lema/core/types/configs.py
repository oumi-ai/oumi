from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from omegaconf import MISSING

from lema.core.types.base_config import BaseConfig
from lema.core.types.params.data_params import DataParams
from lema.core.types.params.model_params import ModelParams
from lema.core.types.params.peft_params import PeftParams
from lema.core.types.params.training_params import TrainerType, TrainingParams
from lema.logging import logger


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
        if self.training.trainer_type == TrainerType.TRL_SFT:
            if not self.data.train.target_col:
                raise ValueError("`target_col` must be specified for TRL_SFT Trainer.")

            # Set `dataset_text_field` in `trainer_kwargs` since it's requried for
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
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation_framework: EvaluationFramework = EvaluationFramework.LM_HARNESS
    # Number of few-shot examples (with responses) to add in the prompt, in order to
    # teach the model how to respond to the specific dataset's prompts.
    num_shots: Optional[int] = 0
    # Number of samples/examples to evaluate from this dataset. Mostly for debugging, in
    # order to reduce the runtime. If set to `0`: the entire dataset is evaluated.
    num_samples: Optional[int] = 0
    # Where to write computed evaluations.
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
        if self.num_samples and self.num_samples < 0:
            raise ValueError("`num_samples` must be non-negative.")


@dataclass
class AsyncEvaluationConfig(BaseConfig):
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # The directory to poll for new checkpoints.
    checkpoints_dir: str = MISSING

    # The time in seconds between the end of the previous evaluation and the start of
    # the next polling attempt. Cannot be negative.
    polling_interval: float = MISSING

    # The number of times to retry polling before exiting the current job.
    # A retry occurs when the job reads the target directory but cannot find a new
    # model checkpoint to evaluate. Defaults to 5. Cannot be negative.
    num_retries: int = 5

    def __post_init__(self):
        """Verifies/populates params."""
        if self.polling_interval < 0:
            raise ValueError("`polling_interval` must be non-negative.")
        if self.num_retries < 0:
            raise ValueError("`num_retries` must be non-negative.")
