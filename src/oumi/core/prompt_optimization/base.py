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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization.progress import OptimizationStats, ProgressTracker
from oumi.utils.logging import logger


@dataclass
class OptimizationResult:
    """Result from a prompt optimization run."""

    optimized_prompt: str
    """The optimized prompt/instruction."""

    optimized_demos: list[dict[str, Any]]
    """List of optimized few-shot demonstrations."""

    optimized_hyperparameters: dict[str, Any]
    """Optimized generation hyperparameters."""

    final_score: float
    """Final evaluation score on validation set."""

    training_history: list[dict[str, Any]]
    """History of optimization trials and scores."""

    num_trials: int
    """Number of optimization trials performed."""

    metadata: dict[str, Any]
    """Additional metadata about the optimization run."""

    optimization_stats: OptimizationStats | None = None
    """Statistics from the optimization run (timing, inference calls, etc)."""

    candidate_programs: list[dict[str, Any]] | None = None
    """List of candidate programs explored during optimization.

    For MIPROv2: Contains all candidate programs with their scores.
    Format: [{"program": str, "score": float}, ...]
    """

    detailed_results: dict[str, Any] | None = None
    """Detailed results from the optimizer.

    For GEPA: Contains DspyGEPAResult with Pareto frontiers, lineage info, etc.
    For other optimizers: May contain optimizer-specific metadata.
    """


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers."""

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize the optimizer.

        Args:
            config: Configuration for prompt optimization.
            metric_fn: Optional custom metric function that takes predictions
                and references and returns a score between 0 and 1.
        """
        self.config = config
        self.metric_fn = metric_fn

    @abstractmethod
    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: str | None = None,
    ) -> OptimizationResult:
        """Optimize prompts using the training data.

        Args:
            train_data: Training dataset as list of examples.
                Each example should have 'input' and 'output' fields.
            val_data: Validation dataset for evaluating optimized prompts.
            initial_prompt: Optional initial prompt to start optimization from.

        Returns:
            OptimizationResult containing the optimized prompt and metadata.
        """
        pass

    @abstractmethod
    def get_optimizer_name(self) -> str:
        """Get the name of the optimizer."""
        pass

    def _log_progress(self, message: str) -> None:
        """Log optimization progress if verbose mode is enabled.

        Args:
            message: Message to log.
        """
        if self.config.optimization.verbose:
            print(f"[{self.get_optimizer_name()}] {message}")

    def _create_progress_tracker(
        self, total_trials: int, description: str | None = None
    ) -> ProgressTracker:
        """Create a progress tracker for optimization.

        Args:
            total_trials: Total number of trials to run.
            description: Optional description for progress bar.

        Returns:
            ProgressTracker instance.
        """
        desc = description or f"{self.get_optimizer_name()} Optimization"
        # Disable progress bar if not verbose
        disable = not self.config.optimization.verbose
        return ProgressTracker(total_trials, description=desc, disable=disable)

    def _evaluate_program_on_dataset(
        self,
        program: Any,
        dataset: list[Any],
        metric_fn: Callable,
        description: str = "Evaluating on validation set",
        skip: bool = False,
        estimated_score: float | None = None,
    ) -> tuple[float, list[float], int, "OptimizationStats"]:
        """Evaluate a DSPy program on a dataset.

        Args:
            program: The DSPy program to evaluate.
            dataset: List of DSPy examples to evaluate on.
            metric_fn: The metric function to use for evaluation.
            description: Description for the progress bar.
            skip: If True, skip evaluation and return estimated_score.
            estimated_score: Score to return if skip=True.

        Returns:
            Tuple of (average_score, all_scores, failed_examples, stats).
        """
        from oumi.core.prompt_optimization.progress import OptimizationStats

        # If skipping evaluation, return the estimated score
        if skip:
            if estimated_score is None:
                estimated_score = 0.0
                logger.warning(
                    "Final evaluation skipped but no estimated score provided. "
                    "Returning score of 0.0. Enable final evaluation for "
                    "accurate scores."
                )

            self._log_progress(
                f"Skipping final evaluation (skip_final_eval=True). "
                f"Using estimated score: {estimated_score:.4f}"
            )

            # Create dummy stats
            stats = OptimizationStats()
            stats.end_time = stats.start_time
            return estimated_score, [estimated_score], 0, stats

        eval_tracker = self._create_progress_tracker(
            len(dataset), description=description
        )

        scores = []
        failed_examples = 0

        with eval_tracker:
            for i, example in enumerate(dataset):
                try:
                    # Run the program on the example
                    prediction = program(question=example.question)
                    score = metric_fn(example, prediction)
                    scores.append(score)

                    # Update progress
                    current_avg = sum(scores) / len(scores)
                    eval_tracker.update(
                        n=1,
                        score=current_avg,
                        examples_processed=1,
                        inference_calls=1,
                    )
                except Exception as e:
                    self._log_progress(
                        f"Warning: Evaluation failed for example "
                        f"{i + 1}/{len(dataset)}: {e}"
                    )
                    failed_examples += 1
                    scores.append(0.0)
                    eval_tracker.update(
                        n=1, examples_processed=1, inference_calls=1, failed_calls=1
                    )

        if failed_examples > 0:
            self._log_progress(
                f"Warning: {failed_examples}/{len(dataset)} examples failed "
                "during evaluation"
            )

        if not scores:
            raise RuntimeError(
                "All validation examples failed during evaluation. "
                "Please check your model and metric configuration."
            )

        final_score = sum(scores) / len(scores)
        stats = eval_tracker.get_stats()

        return final_score, scores, failed_examples, stats
