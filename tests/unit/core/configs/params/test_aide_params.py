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

import pytest

from oumi.core.configs.aide_config import AideConfig
from oumi.core.configs.params.aide_params import (
    AideExecParams,
    AideLLMParams,
    AideOptimizationSurface,
    AideParams,
    AideSearchParams,
)

# --- AideLLMParams ---


def test_aide_llm_params_defaults():
    params = AideLLMParams()
    assert params.model == "o4-mini"
    assert params.temperature == 0.5


def test_aide_llm_params_custom():
    params = AideLLMParams(model="claude-sonnet-4-20250514", temperature=0.8)
    assert params.model == "claude-sonnet-4-20250514"
    assert params.temperature == 0.8


# --- AideSearchParams ---


def test_aide_search_params_defaults():
    params = AideSearchParams()
    assert params.num_drafts == 5
    assert params.debug_prob == 0.5
    assert params.max_debug_depth == 3


def test_aide_search_params_invalid_num_drafts():
    with pytest.raises(ValueError, match="num_drafts must be >= 1"):
        AideSearchParams(num_drafts=0)


def test_aide_search_params_invalid_debug_prob_low():
    with pytest.raises(ValueError, match="debug_prob must be between"):
        AideSearchParams(debug_prob=-0.1)


def test_aide_search_params_invalid_debug_prob_high():
    with pytest.raises(ValueError, match="debug_prob must be between"):
        AideSearchParams(debug_prob=1.5)


def test_aide_search_params_invalid_max_debug_depth():
    with pytest.raises(ValueError, match="max_debug_depth must be >= 0"):
        AideSearchParams(max_debug_depth=-1)


def test_aide_search_params_edge_values():
    params = AideSearchParams(num_drafts=1, debug_prob=0.0, max_debug_depth=0)
    assert params.num_drafts == 1
    assert params.debug_prob == 0.0
    assert params.max_debug_depth == 0


def test_aide_search_params_max_debug_prob():
    params = AideSearchParams(debug_prob=1.0)
    assert params.debug_prob == 1.0


# --- AideExecParams ---


def test_aide_exec_params_defaults():
    params = AideExecParams()
    assert params.timeout == 3600
    assert params.format_tb_ipython is False
    assert params.agent_file_name == "runfile.py"


def test_aide_exec_params_custom():
    params = AideExecParams(timeout=600, agent_file_name="experiment.py")
    assert params.timeout == 600
    assert params.agent_file_name == "experiment.py"


# --- AideParams ---


def test_aide_params_defaults():
    params = AideParams()
    assert params.steps == 20
    assert params.surface == AideOptimizationSurface.CONFIG_SEARCH
    assert params.target_metric == "eval_loss"
    assert params.target_direction == "minimize"
    assert params.generate_report is True
    assert params.log_level == "info"


def test_aide_params_invalid_direction():
    with pytest.raises(ValueError, match="target_direction must be"):
        AideParams(target_direction="unknown")


def test_aide_params_invalid_steps():
    with pytest.raises(ValueError, match="steps must be >= 1"):
        AideParams(steps=0)


def test_aide_params_invalid_log_level():
    with pytest.raises(ValueError, match="Invalid log_level"):
        AideParams(log_level="verbose")


def test_aide_params_maximize_direction():
    params = AideParams(target_metric="accuracy", target_direction="maximize")
    assert params.target_direction == "maximize"


def test_aide_params_all_surfaces():
    for surface in AideOptimizationSurface:
        params = AideParams(surface=surface)
        assert params.surface == surface


def test_aide_params_nested_defaults():
    params = AideParams()
    assert isinstance(params.code_llm, AideLLMParams)
    assert isinstance(params.feedback_llm, AideLLMParams)
    assert isinstance(params.search, AideSearchParams)
    assert isinstance(params.execution, AideExecParams)
    assert params.code_llm.model == "o4-mini"
    assert params.feedback_llm.model == "gpt-4.1-mini"


def test_aide_params_telemetry_dir_default():
    """TelemetryParams defaults to 'telemetry' subdir, resolved under output_dir."""
    params = AideParams()
    # TelemetryParams.telemetry_dir defaults to "telemetry", which is relative
    # and gets resolved under output_dir.
    assert params.telemetry_dir == Path("output/aide/telemetry")


def test_aide_params_telemetry_dir_relative():
    params = AideParams(output_dir="output/aide")
    params.telemetry.telemetry_dir = "stats"
    assert params.telemetry_dir == Path("output/aide/stats")


# --- AideOptimizationSurface ---


def test_aide_optimization_surface_values():
    assert AideOptimizationSurface.CONFIG_SEARCH.value == "config_search"
    assert AideOptimizationSurface.REWARD_FUNCTION.value == "reward_function"
    assert AideOptimizationSurface.EVAL_FUNCTION.value == "eval_function"
    assert AideOptimizationSurface.FULL_PIPELINE.value == "full_pipeline"


def test_aide_optimization_surface_enum_values_are_strings():
    for surface in AideOptimizationSurface:
        assert isinstance(surface.value, str)


# --- AideConfig ---


def test_aide_config_defaults():
    config = AideConfig()
    assert config.goal == ""
    assert config.base_training_config is None
    assert config.mutable_config_paths == []
    assert config.eval_task_names == []
    assert isinstance(config.aide, AideParams)


def test_aide_config_with_goal():
    config = AideConfig(goal="Minimize eval_loss for SmolLM")
    assert config.goal == "Minimize eval_loss for SmolLM"


def test_aide_config_with_mutable_paths():
    paths = ["training.learning_rate", "peft.lora_r"]
    config = AideConfig(mutable_config_paths=paths)
    assert config.mutable_config_paths == paths


def test_aide_config_yaml_roundtrip(tmp_path: Path):
    yaml_content = """\
model:
  model_name: "test-model"

goal: "Test goal"

aide:
  steps: 10
  target_metric: "accuracy"
  target_direction: "maximize"
  code_llm:
    model: "gpt-4.1"
  search:
    num_drafts: 3

mutable_config_paths:
  - "training.learning_rate"
"""
    yaml_path = tmp_path / "roundtrip.yaml"
    yaml_path.write_text(yaml_content)

    loaded = AideConfig.from_yaml_and_arg_list(str(yaml_path), [])

    assert loaded.goal == "Test goal"
    assert loaded.aide.steps == 10
    assert loaded.aide.target_metric == "accuracy"
    assert loaded.aide.target_direction == "maximize"
    assert loaded.aide.code_llm.model == "gpt-4.1"
    assert loaded.aide.search.num_drafts == 3
    assert loaded.mutable_config_paths == ["training.learning_rate"]


def test_aide_config_yaml_with_cli_overrides(tmp_path: Path):
    yaml_content = """\
model:
  model_name: "test-model"

goal: "Test goal"

aide:
  steps: 10
"""
    yaml_path = tmp_path / "overrides.yaml"
    yaml_path.write_text(yaml_content)

    loaded = AideConfig.from_yaml_and_arg_list(
        str(yaml_path),
        ["aide.steps=30", "aide.target_metric=f1_score"],
    )

    assert loaded.aide.steps == 30
    assert loaded.aide.target_metric == "f1_score"


def test_aide_config_finalize_and_validate(tmp_path: Path):
    yaml_content = """\
model:
  model_name: "test-model"

goal: "Minimize eval loss"

aide:
  steps: 5
  target_direction: "minimize"
"""
    yaml_path = tmp_path / "validate.yaml"
    yaml_path.write_text(yaml_content)

    config = AideConfig.from_yaml_and_arg_list(str(yaml_path), [])
    config.finalize_and_validate()  # Should not raise


def test_aide_config_finalize_and_validate_requires_goal(tmp_path: Path):
    yaml_content = """\
model:
  model_name: "test-model"

aide:
  steps: 5
"""
    yaml_path = tmp_path / "no_goal.yaml"
    yaml_path.write_text(yaml_content)

    config = AideConfig.from_yaml_and_arg_list(str(yaml_path), [])
    with pytest.raises(ValueError, match="goal.*required"):
        config.finalize_and_validate()


def test_aide_config_full_yaml(tmp_path: Path):
    yaml_content = """\
model:
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  torch_dtype_str: "bfloat16"

goal: "Optimize eval_loss for SmolLM"

aide:
  steps: 15
  surface: CONFIG_SEARCH
  target_metric: "eval_loss"
  target_direction: "minimize"
  output_dir: "output/aide_test"
  workspace_dir: "workspaces/aide_test"

  code_llm:
    model: "claude-sonnet-4-20250514"
    temperature: 0.7

  feedback_llm:
    model: "gpt-4.1-mini"
    temperature: 0.3

  search:
    num_drafts: 3
    debug_prob: 0.6
    max_debug_depth: 2

  execution:
    timeout: 1800

mutable_config_paths:
  - "training.learning_rate"
  - "training.optimizer"
  - "peft.lora_r"
"""
    yaml_path = tmp_path / "aide_config.yaml"
    yaml_path.write_text(yaml_content)

    config = AideConfig.from_yaml_and_arg_list(str(yaml_path), [])

    assert config.model.model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert config.goal == "Optimize eval_loss for SmolLM"
    assert config.aide.steps == 15
    assert config.aide.surface == AideOptimizationSurface.CONFIG_SEARCH
    assert config.aide.code_llm.model == "claude-sonnet-4-20250514"
    assert config.aide.code_llm.temperature == 0.7
    assert config.aide.feedback_llm.model == "gpt-4.1-mini"
    assert config.aide.search.num_drafts == 3
    assert config.aide.search.debug_prob == 0.6
    assert config.aide.search.max_debug_depth == 2
    assert config.aide.execution.timeout == 1800
    assert config.mutable_config_paths == [
        "training.learning_rate",
        "training.optimizer",
        "peft.lora_r",
    ]


def test_aide_config_reward_function_surface(tmp_path: Path):
    yaml_content = """\
model:
  model_name: "test-model"

goal: "Design a reward function for math reasoning"

aide:
  steps: 10
  surface: REWARD_FUNCTION
  target_metric: "reward_accuracy"
  target_direction: "maximize"
"""
    yaml_path = tmp_path / "reward_aide.yaml"
    yaml_path.write_text(yaml_content)

    config = AideConfig.from_yaml_and_arg_list(str(yaml_path), [])
    assert config.aide.surface == AideOptimizationSurface.REWARD_FUNCTION
    assert config.aide.target_direction == "maximize"


def test_aide_config_can_import_from_core_configs():
    """Verify AIDE configs are accessible from the centralized configs module."""
    from oumi.core.configs import (
        AideConfig as AC,
    )
    from oumi.core.configs import (
        AideExecParams as AEP,
    )
    from oumi.core.configs import (
        AideLLMParams as ALP,
    )
    from oumi.core.configs import (
        AideOptimizationSurface as AOS,
    )
    from oumi.core.configs import (
        AideParams as AP,
    )
    from oumi.core.configs import (
        AideSearchParams as ASP,
    )

    assert AC is AideConfig
    assert AP is AideParams
    assert AOS is AideOptimizationSurface
    assert ALP is AideLLMParams
    assert ASP is AideSearchParams
    assert AEP is AideExecParams
