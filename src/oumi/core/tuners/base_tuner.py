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


from abc import ABC, abstractmethod
from typing import Any, Callable

from oumi.core.configs.tuning_config import TuningConfig, TuningParams


class BaseTuner(ABC):
    """Abstract base class for hyperparameter tuners.

    This class defines the interface that all tuner implementations must follow,
    allowing for different optimization backends (Optuna, Ray Tune, etc.) while
    maintaining a consistent API.
    """

    def __init__(self, tuning_params: TuningParams):
        """Initialize the tuner with configuration parameters.

        Args:
            tuning_params: Configuration for the tuning process.
        """
        self.tuning_params = tuning_params
        self._study = None

    @abstractmethod
    def create_study(self) -> None:
        """Create a new optimization study.

        This method should initialize the tuner's internal study object
        with the appropriate configuration (study name, direction, etc.).
        """
        pass

    @abstractmethod
    def suggest_parameters(self, trial: Any) -> dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: The trial object from the underlying tuner backend.

        Returns:
            Dictionary mapping parameter names to suggested values.
        """
        pass

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[..., Any],
        n_trials: int,
    ) -> None:
        """Run the optimization process.

        Args:
            objective_fn: Function that takes suggested parameters and returns
                         a dictionary of metric values.
            n_trials: Number of trials to run.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_best_trial(self) -> dict[str, Any]:
        """Get the best trial from the study, if only one objective is being optimized.

        Returns:
            Dictionary containing best parameters and their metric values.
        """
        pass

    @abstractmethod
    def get_best_trials(self) -> list[dict[str, Any]]:
        """Get the best trials from the study, for multiple objectives.

        Returns:
            Dictionary containing best parameters and their metric values for the best
            trials.
        """
        pass

    @abstractmethod
    def save_study(self, config: TuningConfig) -> None:
        """Saves the study object to the specified output directory.

        Args:
            config (TrainingConfig): The Oumi training config.

        Returns:
            None
        """
        pass
