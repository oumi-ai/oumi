from dataclasses import dataclass, field
from typing import Optional

from lema.core.types.base_config import BaseConfig
from lema.core.types.params.data_params import DataParams, DatasetSplitParams
from lema.core.types.params.model_params import ModelParams
from lema.core.types.params.peft_params import PeftParams
from lema.core.types.params.training_params import (
    IntervalSchedule,
    TrainerType,
    TrainingParams,
)
from lema.logging import logger


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

        # Set to `logging_steps` the `eval_steps` if not specified.
        if (
            self.training.eval_strategy == IntervalSchedule.STEPS.value
            and self.training.eval_steps is None
        ):
            self.training.eval_steps = self.training.logging_steps

        if self.training.should_do_eval and len(self.data.validation.datasets) == 0:
            raise ValueError(
                "You must specify a validation dataset when you request "
                f"the '{self.training.eval_strategy}' eval strategy."
            )


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
