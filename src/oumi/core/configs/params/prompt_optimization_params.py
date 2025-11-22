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
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class PromptOptimizationParams(BaseParams):
    """Parameters for prompt optimization."""

    optimizer: str = "mipro"
    """The optimization algorithm to use.

    Supported optimizers:
        - mipro: MIPROv2 optimizer for generating instructions and few-shot examples
        - gepa: GEPA (Genetic-Pareto) optimizer using reflective prompt evolution
        - bootstrap: BootstrapFewShot optimizer for simple few-shot example selection
        - optuna: BootstrapFewShot with Bayesian optimization (Optuna) for
          hyperparameters

    Deprecated optimizers (will raise an error):
        - evolutionary: DEPRECATED - use mipro, gepa, or bootstrap instead
    """

    num_trials: int = 50
    """Number of optimization trials to run."""

    max_bootstrapped_demos: int = 4
    """Maximum number of bootstrapped demonstrations (few-shot examples) to use."""

    max_labeled_demos: int = 16
    """Maximum number of labeled demonstrations available in the training set."""

    metric_threshold: Optional[float] = None
    """Optional threshold for the evaluation metric.

    If set, optimization will stop early if this threshold is reached.
    """

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    optimize_instructions: bool = True
    """Whether to optimize prompt instructions."""

    optimize_demos: bool = True
    """Whether to optimize few-shot demonstrations."""

    optimize_hyperparameters: bool = False
    """Whether to optimize generation hyperparameters (temperature, top_p, etc.)."""

    hyperparameter_ranges: dict[str, list[Any]] = field(default_factory=dict)
    """Ranges for hyperparameters to optimize.

    Each range is a list of [min, max] values.

    Example:
        {
            "temperature": [0.0, 1.0],
            "top_p": [0.5, 1.0],
            "max_new_tokens": [50, 500]
        }
    """

    save_intermediate_results: bool = True
    """Whether to save intermediate optimization results."""

    enable_checkpointing: bool = True
    """Whether to enable checkpointing for resuming interrupted runs."""

    checkpoint_interval: int = 300
    """Interval in seconds between checkpoints (default: 5 minutes)."""

    resume_from_checkpoint: bool = True
    """Whether to automatically resume from checkpoint if found."""

    skip_final_eval: bool = False
    """Whether to skip the final evaluation on the full validation set.

    DSPy optimizers already evaluate during optimization, so this final evaluation
    provides additional verification but doubles inference costs for evaluation.
    Set to True to save time/cost if you trust DSPy's internal evaluation.
    """

    max_errors: Optional[int] = None
    """Maximum number of errors allowed during optimization before failing.

    If None, uses DSPy's default error handling (typically strict).
    If set to a positive integer, optimization will tolerate up to that many
    errors before failing. This improves fault tolerance for flaky models or
    network issues.

    Example: max_errors=10 allows up to 10 failed examples before stopping.
    """

    verbose: bool = False
    """Whether to print verbose optimization progress."""

    def __finalize_and_validate__(self) -> None:
        """Validates the prompt optimization parameters."""
        valid_optimizers = {"mipro", "gepa", "bootstrap", "optuna"}
        deprecated_optimizers = {"evolutionary"}

        if self.optimizer in deprecated_optimizers:
            raise ValueError(
                f"Optimizer '{self.optimizer}' is deprecated. "
                f"Please use one of: {', '.join(valid_optimizers)}"
            )

        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. "
                f"Must be one of {valid_optimizers}"
            )

        if self.num_trials <= 0:
            raise ValueError(f"num_trials must be positive, got {self.num_trials}")

        if self.max_bootstrapped_demos < 0:
            raise ValueError(
                f"max_bootstrapped_demos must be non-negative, "
                f"got {self.max_bootstrapped_demos}"
            )

        if self.max_labeled_demos < 0:
            raise ValueError(
                f"max_labeled_demos must be non-negative, got {self.max_labeled_demos}"
            )

        if (
            self.metric_threshold is not None
            and not 0.0 <= self.metric_threshold <= 1.0
        ):
            raise ValueError(
                f"metric_threshold must be between 0 and 1, got {self.metric_threshold}"
            )

        if self.max_errors is not None and self.max_errors < 0:
            raise ValueError(
                f"max_errors must be non-negative or None, got {self.max_errors}"
            )
