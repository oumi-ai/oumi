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

"""AIDE-based agentic code optimizer.

Uses the AIDE ML library's tree-search algorithm to iteratively generate,
test, and refine code solutions optimized against a user-defined metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import aide  # type: ignore[reportMissingImports]
    from aide.agent import Agent as AideAgent
    from aide.interpreter import Interpreter as AideInterpreter
    from aide.journal import Journal as AideJournal
    from aide.utils.config import save_run as aide_save_run
except ImportError:
    aide = None  # type: ignore[assignment]

from oumi.core.agentic.base_agentic_optimizer import (
    AideResult,
    BaseAgenticOptimizer,
    ExecCallbackType,
)
from oumi.core.configs.params.aide_params import AideOptimizationSurface, AideParams
from oumi.utils.logging import logger


def _build_oumi_task_desc(
    goal: str,
    surface: AideOptimizationSurface,
    target_metric: str,
    target_direction: str,
    base_config_yaml: str | None = None,
    mutable_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build an AIDE-compatible task description dict from Oumi config."""
    desc: dict[str, Any] = {
        "Task goal": goal,
        "Framework": (
            "You are optimizing a model using the Oumi ML framework. "
            "Oumi provides train(), evaluate(), and infer() functions. "
            "All configs use Python dataclasses that can be constructed "
            "programmatically or loaded from YAML."
        ),
        "Evaluation": (
            f"The target metric is '{target_metric}' "
            f"which should be {target_direction}d. "
            f"Print the metric as: METRIC:{target_metric}=<value>"
        ),
    }

    if surface == AideOptimizationSurface.CONFIG_SEARCH:
        desc["Surface"] = (
            "Generate a Python script that creates a TrainingConfig, "
            "trains a model using oumi.train.train(), evaluates it, "
            "and prints the target metric."
        )
        if mutable_paths:
            desc["Constraints"] = (
                f"You may only modify these config paths: {mutable_paths}"
            )
    elif surface == AideOptimizationSurface.REWARD_FUNCTION:
        desc["Surface"] = (
            "Generate a reward function compatible with Oumi's GRPO trainer. "
            "The function signature must be: "
            "def reward_fn(completions: list[str], "
            "prompts: list[str] | None = None, **kwargs) -> list[float]"
        )
    elif surface == AideOptimizationSurface.EVAL_FUNCTION:
        desc["Surface"] = (
            "Generate a custom evaluation function. "
            "Use @register_evaluation_function to register it. "
            "The function should return a dict of metric names to values."
        )
    elif surface == AideOptimizationSurface.FULL_PIPELINE:
        desc["Surface"] = (
            "Generate a complete training pipeline script using Oumi's API. "
            "The script must print the final metric as: "
            f"METRIC:{target_metric}=<value>"
        )

    if base_config_yaml:
        desc["Base Training Config"] = base_config_yaml

    return desc


def _build_aide_omegaconf(aide_params: AideParams, workspace_dir: Path) -> Any:
    """Convert AideParams to AIDE's internal OmegaConf config."""
    from omegaconf import OmegaConf

    cfg_dict = {
        "agent": {
            "steps": aide_params.steps,
            "k_fold_validation": 5,
            "expose_prediction": False,
            "data_preview": False,
            "code": {
                "model": aide_params.code_llm.model,
                "temp": aide_params.code_llm.temperature,
            },
            "feedback": {
                "model": aide_params.feedback_llm.model,
                "temp": aide_params.feedback_llm.temperature,
            },
            "search": {
                "num_drafts": aide_params.search.num_drafts,
                "debug_prob": aide_params.search.debug_prob,
                "max_debug_depth": aide_params.search.max_debug_depth,
            },
        },
        "exec": {
            "timeout": aide_params.execution.timeout,
            "format_tb_ipython": aide_params.execution.format_tb_ipython,
        },
        "workspace_dir": str(workspace_dir),
        "log_dir": str(Path(aide_params.output_dir) / "logs"),
        "preprocess_dir": str(workspace_dir / "preprocess"),
        "generate_report": aide_params.generate_report,
    }

    if aide_params.generate_report:
        cfg_dict["report"] = {
            "model": aide_params.report_llm.model,
            "temp": aide_params.report_llm.temperature,
        }

    return OmegaConf.create(cfg_dict)


class AideOptimizer(BaseAgenticOptimizer):
    """AIDE-based agentic code optimizer.

    Uses AIDE ML's tree-search algorithm with draft/debug/improve phases
    to iteratively generate and refine code solutions. This parallels
    :class:`~oumi.core.tuners.optuna_tuner.OptunaTuner` for traditional
    hyperparameter tuning.

    Raises:
        ImportError: If the ``aideml`` package is not installed.
    """

    def __init__(
        self,
        aide_params: AideParams,
        task_desc: dict[str, Any],
        workspace_dir: Path,
    ) -> None:
        """Initialize the AIDE optimizer.

        Args:
            aide_params: AIDE configuration parameters.
            task_desc: Task description dict for the AIDE agent.
            workspace_dir: Working directory for generated scripts.

        Raises:
            ImportError: If aideml is not installed.
        """
        if aide is None:
            raise ImportError(
                "AIDE ML is not installed. Please install"
                " oumi with the 'aide' extra to use the AideOptimizer:"
                " pip install oumi[aide]"
            )
        super().__init__(aide_params)
        assert aide is not None  # Type guard

        self._workspace_dir = workspace_dir
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        # Build AIDE's internal config
        self._aide_cfg = _build_aide_omegaconf(aide_params, workspace_dir)

        # Initialize AIDE components
        # AIDE's Agent expects task_desc as a string; convert dict to markdown.
        if isinstance(task_desc, dict):
            task_desc_str = "\n".join(f"## {k}\n{v}" for k, v in task_desc.items())
        else:
            task_desc_str = str(task_desc)

        self._journal = AideJournal()
        self._agent = AideAgent(
            task_desc=task_desc_str,
            cfg=self._aide_cfg,
            journal=self._journal,
        )
        self._interpreter = AideInterpreter(
            workspace_dir,
            timeout=aide_params.execution.timeout,
            format_tb_ipython=aide_params.execution.format_tb_ipython,
        )

        logger.info(
            f"AideOptimizer initialized: {aide_params.steps} steps, "
            f"surface={aide_params.surface.value}, "
            f"metric={aide_params.target_metric} ({aide_params.target_direction})"
        )

    def step(self, exec_callback: ExecCallbackType | None = None) -> None:
        """Execute one search step (draft, debug, or improve).

        Args:
            exec_callback: Optional custom execution callback. If None,
                uses the built-in AIDE interpreter.
        """
        callback = exec_callback or self._interpreter.run
        self._agent.step(exec_callback=callback)
        aide_save_run(self._aide_cfg, self._journal)

        # Log progress
        best = self._journal.get_best_node(only_good=True)
        metric_str = f"{best.metric.value:.6f}" if best and best.metric else "N/A"
        logger.info(
            f"Step {len(self._journal)} complete. "
            f"Best metric: {metric_str}, "
            f"Good: {len(self._journal.good_nodes)}, "
            f"Buggy: {len(self._journal.buggy_nodes)}"
        )

    def get_best_solution(self) -> AideResult:
        """Get the best solution found so far.

        Returns:
            AideResult with the best code, metric, and paths.
        """
        # Handle empty journal (no steps run yet)
        if not self._journal.nodes:
            best_node = None
        else:
            try:
                best_node = self._journal.get_best_node(only_good=False)
            except (ValueError, IndexError):
                best_node = None

        # Save best solution to file
        output_dir = Path(self.aide_params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_solution_path = output_dir / "best_solution.py"
        journal_path = output_dir / "journal.json"

        if best_node and best_node.code:
            best_solution_path.write_text(best_node.code)

        # Save journal
        try:
            journal_data = [
                {
                    "id": str(node.id),
                    "plan": node.plan,
                    "metric": (
                        node.metric.value
                        if node.metric and hasattr(node.metric, "value")
                        else None
                    ),
                    "is_buggy": node.is_buggy,
                    "stage": node.stage_name,
                }
                for node in self._journal.nodes
            ]
            journal_path.write_text(json.dumps(journal_data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save journal: {e}")

        best_metric_value: float | None = None
        if best_node and best_node.metric and hasattr(best_node.metric, "value"):
            raw = best_node.metric.value
            best_metric_value = float(raw) if raw is not None else None

        return AideResult(
            best_code=best_node.code if best_node else "",
            best_metric=best_metric_value,
            total_steps=len(self._journal),
            good_solutions=len(self._journal.good_nodes),
            buggy_solutions=len(self._journal.buggy_nodes),
            journal_path=str(journal_path),
            best_solution_path=str(best_solution_path),
        )

    def get_search_summary(self) -> dict[str, Any]:
        """Get a summary of the search progress.

        Returns:
            Dictionary with total_nodes, good_nodes, buggy_nodes,
            draft_nodes, and best_metric.
        """
        best_metric_value = None
        if self._journal.nodes and self._journal.good_nodes:
            try:
                best = self._journal.get_best_node(only_good=True)
                if best and best.metric and hasattr(best.metric, "value"):
                    raw = best.metric.value
                    best_metric_value = float(raw) if raw is not None else None
            except (ValueError, IndexError):
                pass

        return {
            "total_nodes": len(self._journal),
            "good_nodes": len(self._journal.good_nodes),
            "buggy_nodes": len(self._journal.buggy_nodes),
            "draft_nodes": len(self._journal.draft_nodes),
            "best_metric": (best_metric_value),
        }

    def cleanup(self) -> None:
        """Clean up the interpreter session."""
        self._interpreter.cleanup_session()
