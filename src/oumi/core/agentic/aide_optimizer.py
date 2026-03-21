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

from omegaconf import OmegaConf

try:
    import aide  # type: ignore[reportMissingImports]
    from aide.agent import Agent as AideAgent
    from aide.interpreter import Interpreter as AideInterpreter
    from aide.journal import Journal as AideJournal
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
            f"Print the metric value to stdout so it can be extracted."
        ),
    }

    # Tell the agent about the oumi_helper.py available in the workspace.
    helper_note = (
        "IMPORTANT: An `oumi_helper` module is available in your workspace. "
        "Always use it instead of calling Oumi's API directly. "
        "It handles all configuration loading, environment setup, "
        "and metric extraction.\n"
    )

    if surface == AideOptimizationSurface.CONFIG_SEARCH:
        # Use just the filename — the file is copied into the workspace.
        base_path = Path(base_config_yaml).name if base_config_yaml else "train.yaml"
        desc["Surface"] = (
            "Generate a Python script that optimizes training hyperparameters."
        )
        desc["How to use oumi_helper"] = (
            f"{helper_note}\n"
            "```python\n"
            "from oumi_helper import run_trial, get_config_fields\n"
            "\n"
            "# See what fields are available:\n"
            "# fields = get_config_fields(base_config_path)\n"
            "\n"
            "# Run a trial with specific overrides:\n"
            "metric = run_trial(\n"
            f'    base_config_path="{base_path}",\n'
            "    overrides={\n"
            '        "training.learning_rate": 3e-4,\n'
            '        "training.optimizer": "adafactor",\n'
            '        "training.warmup_ratio": 0.1,\n'
            "    },\n"
            f'    target_metric="{target_metric}",\n'
            ")\n"
            f'print(f"{target_metric}={{metric}}")\n'
            "```\n"
        )
        if mutable_paths:
            desc["Constraints"] = (
                f"You may only modify these config paths: {mutable_paths}"
            )
    elif surface == AideOptimizationSurface.REWARD_FUNCTION:
        base_path = Path(base_config_yaml).name if base_config_yaml else "train.yaml"
        desc["Surface"] = "Generate a reward function for GRPO/RLHF training."
        desc["How to use oumi_helper"] = (
            f"{helper_note}\n"
            "```python\n"
            "from oumi_helper import test_reward\n"
            "\n"
            "def my_reward_fn(\n"
            "    completions: list[str],\n"
            "    prompts: list[str] | None = None,\n"
            "    **kwargs,\n"
            ") -> list[float]:\n"
            "    rewards = []\n"
            "    for completion in completions:\n"
            "        # Your reward logic here.\n"
            "        score = 1.0 if 'correct' in completion else 0.0\n"
            "        rewards.append(score)\n"
            "    return rewards\n"
            "\n"
            "metric = test_reward(\n"
            "    reward_fn=my_reward_fn,\n"
            f'    base_config_path="{base_path}",\n'
            f'    target_metric="{target_metric}",\n'
            ")\n"
            f'print(f"{target_metric}={{metric}}")\n'
            "```\n"
        )
    elif surface == AideOptimizationSurface.EVAL_FUNCTION:
        desc["Surface"] = "Generate a custom evaluation function."
        desc["How to use oumi_helper"] = (
            f"{helper_note}\n"
            "```python\n"
            "from oumi_helper import test_eval\n"
            "\n"
            "def my_eval_fn(task_params, config, inference_engine):\n"
            "    # Your evaluation logic here.\n"
            "    # Return dict of metric_name -> float value.\n"
            "    return {'accuracy': 0.95}\n"
            "\n"
            "metric = test_eval(\n"
            "    eval_fn=my_eval_fn,\n"
            f'    base_config_path="eval_config.yaml",\n'
            f'    target_metric="{target_metric}",\n'
            ")\n"
            f'print(f"{target_metric}={{metric}}")\n'
            "```\n"
        )
    elif surface == AideOptimizationSurface.FULL_PIPELINE:
        desc["Surface"] = "Generate a complete training pipeline script."
        desc["How to use oumi_helper"] = (
            f"{helper_note}\n"
            "```python\n"
            "from oumi_helper import run_trial\n"
            "\n"
            "# You can use run_trial for simple overrides,\n"
            "# or call Oumi's API directly for full control:\n"
            "# from oumi.core.configs import TrainingConfig\n"
            "# from oumi.train import train\n"
            "# config = TrainingConfig.from_yaml('train.yaml')\n"
            "# eval_results = train(config)\n"
            "```\n"
        )

    if base_config_yaml:
        desc["Base Training Config"] = base_config_yaml

    return desc


def _build_aide_omegaconf(aide_params: AideParams, workspace_dir: Path) -> Any:
    """Convert AideParams to AIDE's internal OmegaConf config."""
    import coolname
    from omegaconf import OmegaConf

    exp_name = coolname.generate_slug(3)

    cfg_dict = {
        "data_dir": str(workspace_dir / "input"),
        "desc_file": None,
        "goal": None,
        "eval": None,
        "exp_name": exp_name,
        "preprocess_data": False,
        "copy_data": False,
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
            "agent_file_name": aide_params.execution.agent_file_name,
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
        task_desc: dict[str, Any] | str,
        workspace_dir: Path,
        base_training_config: str | None = None,
    ) -> None:
        """Initialize the AIDE optimizer.

        Args:
            aide_params: AIDE configuration parameters.
            task_desc: Task description as dict (keys become markdown headers)
                or string (used as-is).
            workspace_dir: Working directory for generated scripts.
            base_training_config: Path to a base Oumi training config YAML.

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

        # AIDE's interpreter redirects stdout to a queue, which breaks
        # Oumi's rich logging (sys.stdout.isatty() fails). Disable it
        # for the duration of the AIDE run; restored in cleanup().
        import os

        self._prev_rich_logging = os.environ.get("OUMI_DISABLE_RICH_LOGGING")
        os.environ["OUMI_DISABLE_RICH_LOGGING"] = "true"

        # Write helper script into workspace so agent can import it.
        from oumi.core.agentic.workspace_helper import write_workspace_helper

        write_workspace_helper(workspace_dir)

        # Copy base training config into workspace so AIDE subprocess can
        # find it. Resolve the path to absolute so it works regardless of
        # what directory the notebook/script runs from.
        if base_training_config:
            import shutil

            src = Path(base_training_config)
            if not src.is_absolute():
                # Search common locations for the config file.
                for root in [Path.cwd(), Path.cwd().parent, Path(__file__).parents[3]]:
                    candidate = root / src
                    if candidate.exists():
                        src = candidate
                        break

            if src.exists():
                dst = workspace_dir / src.name
                shutil.copy2(src, dst)
                logger.info(f"Copied base config -> {dst}")
            else:
                logger.warning(f"Base config not found: {base_training_config}")

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

        # Log the latest node's details for debugging.
        latest_node = self._journal.nodes[-1] if self._journal.nodes else None
        if latest_node:
            stage = latest_node.stage_name
            logger.info(f"  Action: {stage}")
            if latest_node.plan:
                # Truncate plan to first 200 chars for readability
                plan_preview = latest_node.plan[:200].replace("\n", " ")
                logger.info(f"  Plan: {plan_preview}")
            if latest_node.is_buggy and latest_node.term_out:
                # Show last 5 lines of output for buggy nodes
                out_lines = latest_node.term_out.strip().split("\n")
                error_preview = "\n".join(out_lines[-5:])
                logger.info(f"  Error output:\n{error_preview}")
            if latest_node.analysis:
                logger.info(f"  Analysis: {latest_node.analysis[:200]}")

        # Save run state (journal, config, tree visualization).
        # aide_save_run expects cfg.log_dir to be a Path, but our OmegaConf
        # config stores strings. Handle this by saving manually.
        log_dir = Path(self._aide_cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            from aide.utils import serialize, tree_export

            serialize.dump_json(self._journal, log_dir / "journal.json")
            OmegaConf.save(config=self._aide_cfg, f=log_dir / "config.yaml")
            tree_export.generate(
                self._aide_cfg, self._journal, log_dir / "tree_plot.html"
            )
            best = self._journal.get_best_node(only_good=False)
            if best and best.code:
                (log_dir / "best_solution.py").write_text(best.code)
        except Exception as e:
            logger.warning(f"Failed to save run state: {e}")

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
            logger.warning("No solutions were generated (empty journal).")
            best_node = None
        else:
            try:
                best_node = self._journal.get_best_node(only_good=False)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not extract best node: {e}")
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
            "best_metric": best_metric_value,
        }

    def cleanup(self) -> None:
        """Clean up the interpreter session and restore environment."""
        self._interpreter.cleanup_session()

        # Restore the rich logging environment variable.
        import os

        if self._prev_rich_logging is None:
            os.environ.pop("OUMI_DISABLE_RICH_LOGGING", None)
        else:
            os.environ["OUMI_DISABLE_RICH_LOGGING"] = self._prev_rich_logging
