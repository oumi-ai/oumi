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

from typing import Any

import pytest

import oumi.environments  # noqa: F401  populates env registry
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)


def _make_tool() -> DeterministicTool:
    return DeterministicTool(
        id="t",
        name="t",
        description="t",
        deterministic_outputs=[DeterministicToolOutput(input={}, output={})],
    )


def test_constructs_with_required_fields():
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
    )
    assert p.id == "e1"
    assert p.env_type == "deterministic"
    assert len(p.tools) == 1


def test_finalize_and_validate_rejects_unknown_env_type():
    p = EnvironmentParams(id="e1", name="E1", description="d", env_type="banana")
    with pytest.raises(ValueError, match="Unknown env_type 'banana'"):
        p.finalize_and_validate()


@pytest.mark.parametrize(
    "field,value",
    [("id", ""), ("name", ""), ("description", ""), ("env_type", "")],
)
def test_finalize_and_validate_rejects_empty_required_field(field, value):
    base: dict[str, Any] = dict(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
    )
    base[field] = value
    p = EnvironmentParams(**base)
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        p.finalize_and_validate()


def test_finalize_and_validate_rejects_duplicate_tool_ids():
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[
            DeterministicTool(
                id="dup",
                name="A",
                description="A",
                deterministic_outputs=[DeterministicToolOutput(input={}, output={})],
            ),
            DeterministicTool(
                id="dup",
                name="B",
                description="B",
                deterministic_outputs=[DeterministicToolOutput(input={}, output={})],
            ),
        ],
    )
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        p.finalize_and_validate()


def test_environment_config_finalize_and_validate_descends_into_list():
    bad = EnvironmentParams(id="e1", name="E1", description="d", env_type="banana")
    cfg = EnvironmentConfig(environments=[bad])
    with pytest.raises(ValueError, match="Unknown env_type 'banana'"):
        cfg.finalize_and_validate()


def test_environment_config_duplicate_env_ids_raises():
    a = EnvironmentParams(id="dup", name="A", description="d", env_type="deterministic")
    b = EnvironmentParams(id="dup", name="B", description="d", env_type="deterministic")
    with pytest.raises(ValueError, match="duplicate environment id 'dup'"):
        EnvironmentConfig(environments=[a, b])


def test_post_init_coerces_dict_tools_to_tool_params():
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[{"id": "t", "name": "t", "description": "t"}],  # type: ignore[list-item]
    )
    assert isinstance(p.tools[0], ToolParams)
    assert p.tools[0].id == "t"


def test_environment_config_duplicate_tool_ids_across_envs_raises():
    a = EnvironmentParams(
        id="env1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
    )
    b = EnvironmentParams(
        id="env2",
        name="E2",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
    )
    with pytest.raises(ValueError, match="duplicate tool id 't'"):
        EnvironmentConfig(environments=[a, b])
