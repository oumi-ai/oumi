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

"""Tests for the AIDE optimizer."""

import pytest

try:
    import aide  # type: ignore[reportMissingImports] # noqa: F401
except ImportError:
    pytest.skip("AIDE ML not installed", allow_module_level=True)

from oumi.core.agentic.aide_optimizer import (
    AideOptimizer,
    _build_aide_omegaconf,
    _build_oumi_task_desc,
)
from oumi.core.agentic.base_agentic_optimizer import AideResult, BaseAgenticOptimizer
from oumi.core.configs.params.aide_params import (
    AideExecParams,
    AideLLMParams,
    AideOptimizationSurface,
    AideParams,
    AideSearchParams,
)

# --- Fixtures ---


@pytest.fixture
def aide_params():
    """Create AideParams for testing."""
    return AideParams(
        steps=3,
        surface=AideOptimizationSurface.CONFIG_SEARCH,
        target_metric="eval_loss",
        target_direction="minimize",
        output_dir="/tmp/aide_test_output",
        workspace_dir="/tmp/aide_test_workspace",
        code_llm=AideLLMParams(model="o4-mini", temperature=0.5),
        feedback_llm=AideLLMParams(model="gpt-5-mini", temperature=0.5),
        search=AideSearchParams(num_drafts=2, debug_prob=0.5, max_debug_depth=2),
        execution=AideExecParams(timeout=60),
    )


# --- Task Description Tests ---


def test_build_oumi_task_desc_config_search():
    desc = _build_oumi_task_desc(
        goal="Minimize eval_loss",
        surface=AideOptimizationSurface.CONFIG_SEARCH,
        target_metric="eval_loss",
        target_direction="minimize",
    )
    assert "Minimize eval_loss" in desc["Task goal"]
    assert "Oumi ML framework" in desc["Framework"]
    assert "eval_loss" in desc["Evaluation"]
    assert "hyperparameters" in desc["Surface"]
    assert "oumi_helper" in desc["How to use oumi_helper"]


def test_build_oumi_task_desc_reward_function():
    desc = _build_oumi_task_desc(
        goal="Design math reward",
        surface=AideOptimizationSurface.REWARD_FUNCTION,
        target_metric="reward_accuracy",
        target_direction="maximize",
    )
    assert "reward function" in desc["Surface"].lower()
    assert "GRPO" in desc["Surface"]


def test_build_oumi_task_desc_eval_function():
    desc = _build_oumi_task_desc(
        goal="Create eval",
        surface=AideOptimizationSurface.EVAL_FUNCTION,
        target_metric="f1",
        target_direction="maximize",
    )
    assert "evaluation function" in desc["Surface"].lower()


def test_build_oumi_task_desc_full_pipeline():
    desc = _build_oumi_task_desc(
        goal="Full pipeline",
        surface=AideOptimizationSurface.FULL_PIPELINE,
        target_metric="accuracy",
        target_direction="maximize",
    )
    assert "complete training pipeline" in desc["Surface"].lower()


def test_build_oumi_task_desc_with_mutable_paths():
    desc = _build_oumi_task_desc(
        goal="Test",
        surface=AideOptimizationSurface.CONFIG_SEARCH,
        target_metric="eval_loss",
        target_direction="minimize",
        mutable_paths=["training.learning_rate", "peft.lora_r"],
    )
    assert "Constraints" in desc
    assert "training.learning_rate" in desc["Constraints"]


def test_build_oumi_task_desc_with_base_config():
    desc = _build_oumi_task_desc(
        goal="Test",
        surface=AideOptimizationSurface.CONFIG_SEARCH,
        target_metric="eval_loss",
        target_direction="minimize",
        base_config_yaml="model:\n  model_name: test\n",
    )
    assert "Base Training Config" in desc
    assert "model_name: test" in desc["Base Training Config"]


# --- OmegaConf Config Tests ---


def test_build_aide_omegaconf(aide_params, tmp_path):
    cfg = _build_aide_omegaconf(aide_params, tmp_path)
    assert cfg.agent.steps == 3
    assert cfg.agent.code.model == "o4-mini"
    assert cfg.agent.code.temp == 0.5
    assert cfg.agent.feedback.model == "gpt-5-mini"
    assert cfg.agent.search.num_drafts == 2
    assert cfg.agent.search.debug_prob == 0.5
    assert cfg.exec.timeout == 60


# --- AideOptimizer Tests ---


def test_aide_optimizer_is_base_optimizer_subclass():
    assert issubclass(AideOptimizer, BaseAgenticOptimizer)


def test_aide_optimizer_init(aide_params, tmp_path):
    task_desc = {"Task goal": "Test", "Framework": "Oumi"}
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=tmp_path,
    )
    assert optimizer.aide_params == aide_params
    optimizer.cleanup()


def test_aide_optimizer_get_search_summary(aide_params, tmp_path):
    task_desc = {"Task goal": "Test", "Framework": "Oumi"}
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=tmp_path,
    )
    summary = optimizer.get_search_summary()
    assert summary["total_nodes"] == 0
    assert summary["good_nodes"] == 0
    assert summary["buggy_nodes"] == 0
    optimizer.cleanup()


def test_aide_optimizer_get_best_solution_empty(aide_params, tmp_path):
    task_desc = {"Task goal": "Test", "Framework": "Oumi"}
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=tmp_path,
    )
    result = optimizer.get_best_solution()
    assert isinstance(result, AideResult)
    assert result.best_code == ""
    assert result.best_metric is None
    assert result.total_steps == 0
    optimizer.cleanup()


# --- AideResult Tests ---


def test_aide_result_construction():
    result = AideResult(
        best_code="print('hello')",
        best_metric=0.95,
        total_steps=20,
        good_solutions=15,
        buggy_solutions=5,
        journal_path="/tmp/journal.json",
        best_solution_path="/tmp/best.py",
    )
    assert result.best_code == "print('hello')"
    assert result.best_metric == 0.95
    assert result.total_steps == 20
    assert result.good_solutions == 15
    assert result.buggy_solutions == 5


def test_aide_result_none_metric():
    result = AideResult(
        best_code="",
        best_metric=None,
        total_steps=5,
        good_solutions=0,
        buggy_solutions=5,
        journal_path="/tmp/journal.json",
        best_solution_path="/tmp/best.py",
    )
    assert result.best_metric is None
    assert result.good_solutions == 0


# --- Additional Coverage Tests ---


def test_build_aide_omegaconf_report_disabled(tmp_path):
    """Test OmegaConf config when report generation is disabled."""
    params = AideParams(
        steps=5,
        generate_report=False,
        output_dir=str(tmp_path / "output"),
    )
    cfg = _build_aide_omegaconf(params, tmp_path)
    assert cfg.generate_report is False
    assert "report" not in cfg


def test_build_aide_omegaconf_custom_models(tmp_path):
    """Test OmegaConf config with custom LLM models."""
    params = AideParams(
        steps=5,
        code_llm=AideLLMParams(model="claude-sonnet-4-20250514", temperature=0.8),
        feedback_llm=AideLLMParams(model="gpt-4.1", temperature=0.2),
    )
    cfg = _build_aide_omegaconf(params, tmp_path)
    assert cfg.agent.code.model == "claude-sonnet-4-20250514"
    assert cfg.agent.code.temp == 0.8
    assert cfg.agent.feedback.model == "gpt-4.1"
    assert cfg.agent.feedback.temp == 0.2


def test_build_oumi_task_desc_no_constraints():
    """Config search without mutable_paths has no Constraints key."""
    desc = _build_oumi_task_desc(
        goal="Test",
        surface=AideOptimizationSurface.CONFIG_SEARCH,
        target_metric="eval_loss",
        target_direction="minimize",
        mutable_paths=None,
    )
    assert "Constraints" not in desc


def test_build_oumi_task_desc_no_base_config():
    """Task desc without base config has no Base Training Config key."""
    desc = _build_oumi_task_desc(
        goal="Test",
        surface=AideOptimizationSurface.CONFIG_SEARCH,
        target_metric="eval_loss",
        target_direction="minimize",
        base_config_yaml=None,
    )
    assert "Base Training Config" not in desc


def test_build_oumi_task_desc_metric_direction_in_eval():
    """Evaluation section includes both metric name and direction."""
    desc = _build_oumi_task_desc(
        goal="Test",
        surface=AideOptimizationSurface.FULL_PIPELINE,
        target_metric="accuracy",
        target_direction="maximize",
    )
    assert "accuracy" in desc["Evaluation"]
    assert "maximize" in desc["Evaluation"]


def test_aide_optimizer_get_search_summary_empty_journal(aide_params, tmp_path):
    """get_search_summary returns None metric on empty journal."""
    task_desc = {"Task goal": "Test", "Framework": "Oumi"}
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=tmp_path,
    )
    summary = optimizer.get_search_summary()
    assert summary["best_metric"] is None
    assert summary["total_nodes"] == 0
    assert summary["draft_nodes"] == 0
    optimizer.cleanup()


def test_aide_optimizer_get_best_solution_saves_files(aide_params, tmp_path):
    """get_best_solution creates output files even with empty journal."""
    aide_params.output_dir = str(tmp_path / "output")
    task_desc = {"Task goal": "Test", "Framework": "Oumi"}
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=tmp_path / "workspace",
    )
    result = optimizer.get_best_solution()

    from pathlib import Path

    assert Path(result.journal_path).exists()
    assert result.best_solution_path.endswith("best_solution.py")
    optimizer.cleanup()


def test_aide_optimizer_init_with_string_task_desc(aide_params, tmp_path):
    """AideOptimizer accepts string task_desc (not just dict)."""
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc="Simple string goal",
        workspace_dir=tmp_path,
    )
    assert optimizer.aide_params == aide_params
    optimizer.cleanup()


def test_aide_optimizer_cleanup_is_idempotent(aide_params, tmp_path):
    """Calling cleanup() multiple times should not raise."""
    task_desc = {"Task goal": "Test"}
    optimizer = AideOptimizer(
        aide_params=aide_params,
        task_desc=task_desc,
        workspace_dir=tmp_path,
    )
    optimizer.cleanup()
    # Second cleanup should not raise
    optimizer.cleanup()
