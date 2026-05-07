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

import logging
from typing import Any

import pytest

import oumi.environments  # noqa: F401  populates env registry
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import (
    GroundingConfig,
    ToolGroundingConfig,
)
from oumi.core.configs.params.tool_params import ToolParams


def _make_tool(tool_id: str = "t") -> ToolParams:
    return ToolParams(id=tool_id, name=tool_id, description="t")


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
        tools=[_make_tool("dup"), _make_tool("dup")],
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
    assert type(p.tools[0]) is ToolParams
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


# --- grounding ---


def test_grounding_post_init_coerces_dict_to_grounding_config():
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
        grounding={"sample_size": 5, "tools": {"t": {"fields": ["a"]}}},  # type: ignore[arg-type]
    )
    assert isinstance(p.grounding, GroundingConfig)
    assert p.grounding.sample_size == 5
    assert isinstance(p.grounding.tools["t"], ToolGroundingConfig)


def test_grounding_validate_rejects_empty_tools_dict():
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
        grounding=GroundingConfig(sample_size=3, tools={}),
    )
    with pytest.raises(ValueError, match="grounding.tools is empty"):
        p.finalize_and_validate()


def test_grounding_validate_passes_with_at_least_one_tool():
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
        grounding=GroundingConfig(
            sample_size=3,
            tools={"t": ToolGroundingConfig(fields=["a"])},
        ),
    )
    p.finalize_and_validate()
    assert p.grounding is not None


def test_grounding_warns_on_stale_tool_ids(caplog):
    """Warning, not error, when grounding.tools names a tool not in env.tools."""
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool("real_tool")],
        grounding=GroundingConfig(
            sample_size=3,
            tools={
                "real_tool": ToolGroundingConfig(fields=["a"]),
                "ghost_tool": ToolGroundingConfig(fields=["b"]),
            },
        ),
    )
    with caplog.at_level(logging.WARNING, logger="oumi"):
        p.finalize_and_validate()
    assert any(
        "ghost_tool" in rec.getMessage() and "unknown tool" in rec.getMessage()
        for rec in caplog.records
    )


def test_no_grounding_does_not_validate_tools():
    """env.grounding is None → no grounding-related validation runs."""
    p = EnvironmentParams(
        id="e1",
        name="E1",
        description="d",
        env_type="deterministic",
        tools=[_make_tool()],
        grounding=None,
    )
    p.finalize_and_validate()
    assert p.grounding is None
