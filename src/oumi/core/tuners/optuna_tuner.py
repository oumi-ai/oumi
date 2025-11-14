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

from pathlib import Path
from typing import Any, Callable, Union

try:
    import optuna
except ImportError:
    raise ImportError(
        "Optuna is not installed. Please install"
        " oumi with the 'tuning' extra to use the OptunaTuner"
    )

from oumi.core.configs import TuningConfig
from oumi.core.configs.params.tuning_params import ParamType, TuningParams
from oumi.core.tuners.base_tuner import BaseTuner


class OptunaTuner(BaseTuner):
    """Optuna-based hyperparameter tuner implementation."""

    def __init__(self, tuning_params: TuningParams) -> None:
        """Initializes the Optuna based hyperparameter tuner.

        Args:
            tuning_params (TuningParams): the tuning parameters to be used
        """
        super().__init__(tuning_params)
        self._study: optuna.Study
        self._sampler: optuna.samplers.BaseSampler

    def create_study(self) -> None:
        """Create an Optuna study with multi-objective optimization support."""
        # Determine optimization directions
        directions = []
        for direction in self.tuning_params.evaluation_direction:
            if direction == "minimize":
                directions.append(optuna.study.StudyDirection.MINIMIZE)
            elif direction == "maximize":
                directions.append(optuna.study.StudyDirection.MAXIMIZE)
            elif direction == "not set":
                directions.append(optuna.study.StudyDirection.NOT_SET)
            else:
                raise ValueError(f"Unsupported optimization direction: {direction}")

        # Initialize sampler
        if self.tuning_params.tuner_sampler is not None:
            if self.tuning_params.tuner_sampler == "TPESampler":
                self._sampler = optuna.samplers.TPESampler()
            elif self.tuning_params.tuner_sampler == "RandomSampler":
                self._sampler = optuna.samplers.RandomSampler()
            else:
                raise ValueError(
                    f"Unsupported sampler: {self.tuning_params.tuner_sampler}"
                )
        else:
            self._sampler = optuna.samplers.TPESampler()

        # Create study or load existing one
        self._study = optuna.create_study(
            directions=directions,
            storage=self.tuning_params.storage,
            study_name=self.tuning_params.tuning_study_name,
            load_if_exists=self.tuning_params.load_if_exists,
            sampler=self._sampler,
        )

    def suggest_parameters(
        self, trial: optuna.Trial
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Suggest parameters using Optuna's suggest methods."""

        def _suggest_single_param(
            param_name: str, param_spec: Any, params: dict[str, Any]
        ) -> None:
            if isinstance(param_spec, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_spec)
            elif isinstance(param_spec, dict):
                param_type = ParamType(param_spec["type"])

                if param_type == ParamType.CATEGORICAL:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_spec["choices"]
                    )
                elif param_type == ParamType.INT:
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                    )
                elif param_type in [ParamType.FLOAT, ParamType.UNIFORM]:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                    )
                elif param_type == ParamType.LOGUNIFORM:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                        log=True,
                    )
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
            else:
                raise ValueError(
                    f"Parameter specification for {param_name} is invalid."
                )

        suggested_train_params: dict[str, Any] = {}
        for (
            param_name,
            param_spec,
        ) in self.tuning_params.tunable_training_params.items():
            _suggest_single_param(param_name, param_spec, params=suggested_train_params)

        suggested_peft_params: dict[str, Any] = {}
        for param_name, param_spec in self.tuning_params.tunable_peft_params.items():
            _suggest_single_param(param_name, param_spec, params=suggested_peft_params)

        return suggested_train_params, suggested_peft_params

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any], dict[str, Any], int], dict[str, float]],
        n_trials: int,
    ) -> None:
        """Run Optuna optimization."""
        if self._study is None:
            self.create_study()

        def _objective(trial: optuna.Trial) -> Union[float, list[float]]:
            # Get suggested parameters
            train_params, peft_params = self.suggest_parameters(trial)

            # Run objective function (training + evaluation)
            metrics = objective_fn(
                train_params,
                peft_params,
                trial.number,
            )

            # Return metric values in the order specified
            metric_values = [
                metrics[metric_name]
                for metric_name in self.tuning_params.evaluation_metrics
            ]

            # Return single value or list for multi-objective
            return metric_values[0] if len(metric_values) == 1 else metric_values

        # Run optimization
        self._study.optimize(_objective, n_trials=n_trials)

    def get_best_trials(self) -> list[dict[str, Any]]:
        """Get the best trials from the Optuna study."""
        if self._study is None:
            raise RuntimeError("Study not created. Call create_study() first.")
        return [
            {
                "trial_number": best_trial.number,
                "params": best_trial.params,
                "values": best_trial.values,
                "number": best_trial.number,
            }
            for best_trial in self._study.best_trials
        ]

    def get_best_trial(self) -> dict[str, Any]:
        """Get the best trial from the Optuna study if only one objective."""
        if self._study is None:
            raise RuntimeError("Study not created. Call create_study() first.")

        best_trial = self._study.best_trial
        return {
            "trial_number": best_trial.number,
            "params": best_trial.params,
            "values": best_trial.values,
            "number": best_trial.number,
        }

    def save_study(self, config: TuningConfig) -> None:
        """Saves the study results in a csv file."""
        assert self._study
        self._study.trials_dataframe().to_csv(
            Path(config.tuning.output_dir, "trials_results.csv"), index=False
        )
