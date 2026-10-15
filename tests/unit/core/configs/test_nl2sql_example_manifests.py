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

"""Coverage for the two NL2SQL manifests test_parse_configs.py has to exclude.

Neither is a `TrainingConfig`: one is an `EnvironmentConfig`, the other is
verl-format. This pins their shape and the wiring between them instead.
"""

from pathlib import Path

import pytest
import yaml

from oumi.core.configs.environment_config import EnvironmentConfig

_EXAMPLE_DIR = Path(__file__).parents[4] / "configs" / "examples" / "grpo_verl_nl2sql"


@pytest.fixture
def environment_config() -> EnvironmentConfig:
    return EnvironmentConfig.from_yaml(_EXAMPLE_DIR / "environment_config.yaml")


@pytest.fixture
def verl_tool_config() -> dict:
    return yaml.safe_load((_EXAMPLE_DIR / "verl_tool_config.yaml").read_text())


def test_environment_config_parses_and_declares_one_read_only_sql_tool(
    environment_config,
):
    (environment,) = environment_config.environments
    (tool,) = environment.tools
    assert environment.env_type == "database"
    assert tool.read_only is True
    # Asserted as a string, not imported: the executor ships in the env/reward PR.
    assert tool.executor == "oumi.environments.examples.nl2sql.run_sql"


def test_verl_tool_config_dispatches_to_the_environment_config(verl_tool_config):
    (tool,) = verl_tool_config["tools"]
    assert tool["class_name"].endswith("oumi_verl_tool.OumiVerlTool")
    assert tool["config"]["type"] == "native"
    referenced = Path(tool["config"]["oumi_env_config"])
    assert referenced.name == "environment_config.yaml"
    assert (_EXAMPLE_DIR.parents[2] / referenced).is_file()


def test_the_two_run_sql_declarations_do_not_drift(
    environment_config, verl_tool_config
):
    (oumi_tool,) = environment_config.environments[0].tools
    verl_function = verl_tool_config["tools"][0]["tool_schema"]["function"]
    # OumiVerlTool refuses to construct when these names disagree.
    assert verl_function["name"] == oumi_tool.id == oumi_tool.name
    assert verl_function["description"] == oumi_tool.description
    assert verl_function["parameters"] == oumi_tool.parameters


def test_train_config_points_at_the_verl_tool_config():
    train = yaml.safe_load((_EXAMPLE_DIR / "train.yaml").read_text())
    multi_turn = train["training"]["verl_config_overrides"]["actor_rollout_ref"][
        "rollout"
    ]["multi_turn"]
    assert Path(multi_turn["tool_config_path"]).name == "verl_tool_config.yaml"
    assert train["training"]["reward_functions"] == ["sql_execution_match"]
