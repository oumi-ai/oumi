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

import copy
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from oumi.core.configs import EvaluationConfig, EvaluationTaskParams
from oumi.core.evaluators.evaluation_result import EvaluationResult
from oumi.evaluation.save_utils import save_evaluation_output


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def evaluate(self, config: EvaluationConfig, **kwargs) -> list[EvaluationResult]:
        """Evaluates a model using the provided evaluation configuration.

        Args:
            config: The desired configuration for evaluation.
            kwargs: Additional keyword arguments for an evaluator subclassing the base.

        Returns:
            List of evaluation results (one per task, in the same order with `tasks`).
        """
        # Create a copy of the evaluation config, without tasks, so that there is no
        # redundant information in the `config` input parameter of `self.evaluate_task`.
        config_without_tasks = copy.deepcopy(config)
        config_without_tasks.tasks = []

        # Evaluate on each task included in the configuration, serially.
        evaluation_results = []
        for task in config.tasks:
            evaluation_result = self.evaluate_task(
                task_params=task, config=config_without_tasks, **kwargs
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

    def evaluate_task(
        self,
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        **kwargs,
    ) -> EvaluationResult:
        """Evaluates a model using the provided configuration on a specific task.

        Args:
            task_params: The task parameters for evaluation.
            config: The desired evaluation configuration for evaluation.
            kwargs: Additional keyword arguments for an evaluator subclassing the base.
                These are directly passed into the `_evaluate_task` abstract method,
                which is implemented by the subclass.

        Returns:
            The results for evaluating on the task.
        """
        # Ensure the task prerequisites are satisfied; fast-fail if not.
        self._check_task_prerequisites(task_params)

        # Get a timestamp at the beginning of the current run.
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        # Evaluate the task.
        evaluation_result = self._evaluate_task(
            task_params=task_params,
            config=config,
            **kwargs,
        )
        elapsed_time_sec = time.time() - start_time

        # Save the output, if an output directory has been provided.
        if config.output_dir:
            self.save_output(
                task_params=task_params,
                evaluation_result=evaluation_result,
                base_output_dir=config.output_dir,
                config=config,
                start_time_str=start_time_str,
                elapsed_time_sec=int(elapsed_time_sec),
            )
        return evaluation_result

    def save_output(
        self,
        task_params: EvaluationTaskParams,
        evaluation_result: EvaluationResult,
        base_output_dir: str,
        config: Optional[EvaluationConfig],
        start_time_str: Optional[str],
        elapsed_time_sec: Optional[int],
    ) -> None:
        """Saves the evaluation's output to the specified output directory.

        Args:
            task_params: The task parameters used for this evaluation.
            evaluation_result: The evaluation result.
            base_output_dir: The directory where the evaluation results will be saved.
            config: The evaluation configuration.
            start_time_str: Human-readable timestamp, indicating when the run started.
            elapsed_time_sec: Duration (in seconds) of the evaluation run.

        Returns:
            None
        """
        save_evaluation_output(
            backend_name=task_params.evaluation_platform,
            task_params=task_params,
            evaluation_result=evaluation_result,
            base_output_dir=base_output_dir,
            config=config,
            start_time_str=start_time_str,
            elapsed_time_sec=elapsed_time_sec,
        )

    # Internal methods (to be implemented by subclasses).
    def _check_task_prerequisites(self, task_params: EvaluationTaskParams) -> None:
        pass

    @abstractmethod
    def _evaluate_task(
        self,
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        **kwargs,
    ) -> EvaluationResult:
        """Evaluates a model using the provided configuration for a specific task.

        Args:
            task_params: The task parameters for evaluation.
            config: The desired evaluation configuration for evaluation.
            kwargs: Additional keyword arguments for an evaluator subclassing the base.

        Returns:
            The results for evaluating on the task.
        """
        raise NotImplementedError("Subclasses should implement this method.")
