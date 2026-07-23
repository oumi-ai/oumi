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

"""Registry-name and dotted-path executor resolution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import REGISTRY, Registry, RegistryType, register_tool_executor
from oumi.core.types.tool_call import ToolResult
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.utils import resolve_executor


@pytest.fixture(autouse=True)
def restore_registry():
    snapshot = Registry()
    for reg_type in RegistryType:
        for key, value in REGISTRY.get_all(reg_type).items():
            snapshot.register(key, reg_type, value)
    yield
    REGISTRY.clear()
    REGISTRY._initialized = False
    for reg_type in RegistryType:
        for key, value in snapshot.get_all(reg_type).items():
            REGISTRY.register(key, reg_type, value)


def _registered_executor(arguments, context):
    return ToolResult(output={"via": "registry", "args": arguments})


# Must stay module-level: the fallback test resolves it by dotted path.
def _dotted_executor(arguments, context):
    return ToolResult(output={"via": "dotted"})


def test_register_tool_executor_registers_under_tool_executor_type():
    register_tool_executor("res.registered")(_registered_executor)
    assert REGISTRY.get("res.registered", RegistryType.TOOL_EXECUTOR) is (
        _registered_executor
    )


def test_resolve_executor_prefers_registry_name():
    register_tool_executor("res.registered")(_registered_executor)
    assert resolve_executor("res.registered", "t") is _registered_executor


def test_resolve_executor_falls_back_to_dotted_path():
    resolved = resolve_executor(f"{__name__}._dotted_executor", "t")
    assert resolved is _dotted_executor


def test_resolve_executor_unknown_name_raises():
    with pytest.raises(ValueError, match="executor"):
        resolve_executor("res.never_registered", "t")


class _EchoExecEnv(ExecutableEnvironment):
    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        yield None


def test_executable_environment_dispatches_registry_named_executor():
    register_tool_executor("res.registered")(_registered_executor)
    env = _EchoExecEnv(
        EnvironmentParams(
            id="e",
            name="e",
            description="d",
            env_type="executable",
            tools=[
                ExecutableTool(
                    id="t", name="t", description="d", executor="res.registered"
                )
            ],
        )
    )
    [result] = env.step([("t", {"a": 1})])
    assert result.output == {"via": "registry", "args": {"a": 1}}
