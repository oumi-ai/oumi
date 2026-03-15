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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oumi.core.configs.params.aide_params import AideParams


@dataclass
class AideResult:
    """Result of an AIDE agentic optimization run.

    Attributes:
        best_code: The Python source code of the best solution found.
        best_metric: The metric value achieved by the best solution,
            or None if no successful solution was found.
        total_steps: Total number of search steps executed.
        good_solutions: Number of solutions that ran without errors.
        buggy_solutions: Number of solutions that had bugs or crashed.
        journal_path: Path to the serialized solution tree (JSON).
        best_solution_path: Path to the best solution's source file.
    """

    best_code: str
    best_metric: float | None
    total_steps: int
    good_solutions: int
    buggy_solutions: int
    journal_path: str
    best_solution_path: str


# Type for execution callbacks: takes (code: str, save: bool) -> result.
ExecCallbackType = Callable[[str, bool], Any]


class BaseAgenticOptimizer(ABC):
    """Abstract base class for agentic code optimizers.

    This class defines the interface that all agentic optimizer implementations
    must follow, allowing for different LLM-powered optimization backends
    (AIDE, future alternatives) while maintaining a consistent API.

    This parallels :class:`~oumi.core.tuners.base_tuner.BaseTuner` for
    traditional hyperparameter tuning.
    """

    def __init__(self, aide_params: AideParams):
        """Initialize the optimizer with configuration parameters.

        Args:
            aide_params: Configuration for the agentic optimization process.
        """
        self.aide_params = aide_params

    @abstractmethod
    def step(self, exec_callback: ExecCallbackType | None = None) -> None:
        """Execute one search step (draft, debug, or improve).

        Each step generates a candidate solution, executes it via
        the callback, evaluates the result, and updates the search tree.

        Args:
            exec_callback: Optional function that executes generated code in a
                sandbox and returns the execution result. If None, the optimizer
                uses its built-in execution mechanism.
        """
        pass

    @abstractmethod
    def get_best_solution(self) -> AideResult:
        """Get the best solution found so far.

        Returns:
            AideResult containing the best code, metric, and metadata.
        """
        pass

    @abstractmethod
    def get_search_summary(self) -> dict[str, Any]:
        """Get a summary of the search progress.

        Returns:
            Dictionary with keys like 'total_nodes', 'good_nodes',
            'buggy_nodes', 'best_metric', 'best_metric_history'.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (interpreter sessions, temp files, etc.)."""
        pass
