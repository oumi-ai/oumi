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
from enum import Enum
from pathlib import Path
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.peft_params import PeftParams
from oumi.core.configs.params.telemetry_params import TelemetryParams
from oumi.core.configs.params.training_params import (
    TrainerType,
    TrainingParams,
)
from oumi.utils.logging import logger


class TunerType(Enum):
    """Enum representing the supported tuners."""

    OPTUNA = "optuna"


class ParamType(Enum):
    """Enum representing the type of parameter to tune."""

    CATEGORICAL = "categorical"
    INT = "int"
    FLOAT = "float"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"

    @staticmethod
    def verify_param_spec(
        param_name: str,
        param_spec: dict,
        valid_training_params: set[str],
        type_name: str = "",
    ) -> None:
        """Verifies that a parameter specification is valid."""
        if param_name not in valid_training_params:
            raise ValueError(
                f"Invalid tunable parameter: {param_name}. "
                f"Must be a valid `{type_name}` field."
            )
        elif isinstance(param_spec, dict):
            # Validate required keys
            if "type" not in param_spec:
                raise ValueError(
                    f"Tunable parameter '{param_name}' must have 'type' key"
                )

            param_type_str = param_spec["type"]

            # Validate type is a valid ParamType
            try:
                param_type = ParamType(param_type_str)
            except ValueError:
                valid_types = [t.value for t in ParamType]
                raise ValueError(
                    f"Invalid type '{param_type_str}' for parameter"
                    f" '{param_name}'. Must be one of: {valid_types}"
                )
            # Validate based on parameter type
            if param_type == ParamType.CATEGORICAL:
                if "choices" not in param_spec:
                    raise ValueError(
                        f"Categorical parameter '{param_name}' must have 'choices' key"
                    )
                if (
                    not isinstance(param_spec["choices"], list)
                    or len(param_spec["choices"]) == 0
                ):
                    raise ValueError(
                        f"Categorical parameter '{param_name}' must have"
                        " non-empty choices list"
                    )
            else:
                # All other types need low and high
                required_keys = {"low", "high"}
                if not required_keys.issubset(param_spec.keys()):
                    raise ValueError(
                        f"Parameter '{param_name}' must have 'low' and 'high' keys"
                    )
        else:
            raise ValueError(f"Tunable parameter '{param_name}' must be a dict")


@dataclass
class TuningParams(BaseParams):
    tunable_training_params: dict[str, dict] = field(default_factory=dict)
    """Dictionary mapping parameter names to their search spaces.

    Each value should be:
    - A dict specifying the parameter type and range:
      {
          "type": "float",  # or ParamType enum value
          "low": 1e-5,
          "high": 1e-2
      }

    Supported types from ParamType enum:
    - CATEGORICAL: List of discrete choices
    - INT: Integer range
    - FLOAT: Float range (uniform sampling)
    - LOGUNIFORM: Float range with log-scale sampling
    - UNIFORM: Float range with uniform sampling
    """

    fixed_training_params: dict = field(default_factory=dict)
    """A dictionary containing the fixed parameters for the training process.

    These parameters will remain constant throughout the tuning process and will not be
    modified. This allows you to set certain training parameters that you do not wish to
    tune, while still allowing other parameters to be optimized.
    """

    tunable_peft_params: dict[str, dict] = field(default_factory=dict)
    """Dictionary mapping PEFT parameter names to their search spaces.

    Each value should be:
    - A dict specifying the parameter type and range:
        {
            "type": "categorical",  # or ParamType enum value
            "choices": ["8", "16", "32"]
        }
    Supported types from ParamType enum:
    - CATEGORICAL: List of discrete choices
    - INT: Integer range
    - FLOAT: Float range (uniform sampling)
    - LOGUNIFORM: Float range with log-scale sampling
    - UNIFORM: Float range with uniform sampling
    """

    fixed_peft_params: dict = field(default_factory=dict)
    """A dictionary containing the fixed parameters for the PEFT configuration.

    These parameters will remain constant throughout the tuning process and will not be
    modified. This allows you to set certain PEFT parameters that you do not wish to
    tune, while still allowing other parameters to be optimized.
    """

    output_dir: str = "output"
    """Directory where the output files will be saved.

    This includes all trained models, evaluation results, and any other artifacts
    produced during the tuning process.
    """

    tuning_study_name: str = "tuning-study"
    """A unique identifier for the current tuning run.

    This name is used to identify the tuning study in logging outputs, saved model
    checkpoints, and experiment tracking tools like Weights & Biases or
    TensorBoard.
    """

    evaluation_metrics: list[str] = field(default_factory=lambda: ["eval_loss"])
    """The metrics used to evaluate the performance of the model during tuning.

    These metrics are used to compare different hyperparameter configurations and select
    the best-performing parameter configuration.
    """

    evaluation_direction: list[str] = field(default_factory=lambda: ["minimize"])
    """The direction of optimization for each evaluation metric.

    This can be either "minimize" or "maximize", depending on whether lower or higher
    values of the evaluation metric indicate better performance. If only one value is
    provided, it is applied to all metrics.
    """

    log_level: str = "info"
    """The logging level for the main Oumi logger.

    Possible values are "debug", "info", "warning", "error", "critical".
    """

    logging_strategy: str = "trials"
    """The strategy to use for logging during the tuning process.

    Possible values are:
    - "trials": Log every `logging_steps` steps.
    - "epoch": Log at the end of each epoch for each trail configuration.
    - "no": Disable logging.
    """

    logging_dir: Optional[str] = None
    """The directory where training logs will be saved.

    If not specified, defaults to a "logs" subdirectory within the `output_dir`.
    """

    log_examples: bool = False
    """Whether to log an example of the data in the first step for debugging purposes.

    If True, the example will be logged to the console.
    """

    telemetry: TelemetryParams = field(default_factory=TelemetryParams)
    """Parameters for telemetry.

    This field contains telemetry configuration options.
    """

    tuner_type: TunerType = TunerType.OPTUNA
    """The type of tuner to use for hyperparameter optimization.

    Possible values are:
        - OPTUNA: Optuna tuner.
    """

    # # # # # # # # # # # # # # # # # #
    # Training specific parameters
    # # # # # # # # # # # # # # # # # #

    trainer_type: TrainerType = TrainerType.TRL_SFT
    """The type of trainer to use for model training.
    Possible values are:
        - TRL_SFT: TRL's SFT Trainer
        TODO: Add more options in the future.
    """

    def __post_init__(self):
        """Verifies params."""
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"

        # Validate logging strategy
        valid_logging_strategies = {"trials", "epoch", "no"}
        if self.logging_strategy not in valid_logging_strategies:
            raise ValueError(
                f"Invalid logging_strategy: {self.logging_strategy}. "
                f"Choose from {valid_logging_strategies}."
            )

        # Validate evaluation metrics and directions
        if len(self.evaluation_metrics) != len(self.evaluation_direction):
            if len(self.evaluation_direction) == 1:
                self.evaluation_direction *= len(self.evaluation_metrics)
                logger.warning(
                    "Single evaluation_direction provided. "
                    "Applying it to all evaluation_metrics."
                )
            else:
                raise ValueError(
                    "Length of evaluation_metrics must match length of "
                    "evaluation_direction, or evaluation_direction must be of length 1."
                )

        # Validate each evaluation direction
        for direction in self.evaluation_direction:
            if direction not in {"minimize", "maximize"}:
                raise ValueError(
                    f"Invalid evaluation_direction: {direction}. "
                    'Choose either "minimize" or "maximize".'
                )

        # Warn if using multiple metrics with incompatible logging strategy
        if len(self.evaluation_metrics) > 1 and self.logging_strategy == "epoch":
            logger.warning(
                "Using multiple evaluation_metrics with 'epoch' logging_strategy "
                "may lead to ambiguous logging. Consider using 'trials' instead."
            )

        # Validate trainer type
        # TODO: Add more options in the future.
        if self.trainer_type != TrainerType.TRL_SFT:
            raise ValueError(
                f"Invalid trainer_type: {self.trainer_type}. "
                f"Choose from {[t.value for t in [TrainerType.TRL_SFT]]}."
            )

        # Validate that the params keys are valid TrainingParams fields
        valid_training_params = {
            field.name for field in TrainingParams.__dataclass_fields__.values()
        }

        # Verify fixed training params keys are valid TrainingParams fields
        for param_name in self.fixed_training_params.keys():
            if param_name not in valid_training_params:
                raise ValueError(
                    f"Invalid fixed parameter: {param_name}. "
                    f"Must be a valid `TrainingParams` field."
                )

        # Ensure tunable_training_params values are valid
        for param_name, param_spec in self.tunable_training_params.items():
            ParamType.verify_param_spec(
                param_name, param_spec, valid_training_params, "TrainingParams"
            )

        # Validate that the params keys are valid PEFTParams fields
        valid_training_params = {
            field.name for field in PeftParams.__dataclass_fields__.values()
        }
        # Verify fixed training params keys are valid PEFT fields
        for param_name in self.fixed_peft_params.keys():
            if param_name not in valid_training_params:
                raise ValueError(
                    f"Invalid fixed parameter: {param_name}. "
                    f"Must be a valid `PeftParams` field."
                )

        # Ensure tunable_training_params values are valid
        for param_name, param_spec in self.tunable_peft_params.items():
            ParamType.verify_param_spec(
                param_name, param_spec, valid_training_params, "PeftParams"
            )

    @property
    def telemetry_dir(self) -> Optional[Path]:
        """Returns the telemetry stats output directory."""
        result: Optional[Path] = None
        if self.telemetry.telemetry_dir:
            result = Path(self.telemetry.telemetry_dir)

        if self.output_dir:
            output_dir = Path(self.output_dir)
            # If `telemetry.telemetry_dir` is relative, then treat it
            # as a sub-directory of `output_dir`.
            if result and not result.is_absolute():
                result = output_dir / result

        return result
