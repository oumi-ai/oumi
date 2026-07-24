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

"""Per-rollout Oumi environment resolution for the verl tool adapter."""

from __future__ import annotations

import copy
import weakref
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.synthesis.tool_router import ToolRouter

# Routers stay module-local because their environment connections are not picklable.
_ROUTERS: dict[str, ToolRouter] = {}


class _ToolKwargs(TypedDict, total=False):
    create_kwargs: Mapping[str, Any]


class _RolloutData(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def tools_kwargs(self) -> Mapping[str, _ToolKwargs | None] | None: ...


def _rollout_create_kwargs(
    agent_data: _RolloutData,
) -> dict[str, dict[str, Any]]:
    create_kwargs_by_tool = {}
    for tool_id, entry in (agent_data.tools_kwargs or {}).items():
        if entry and (create_kwargs := entry.get("create_kwargs")):
            create_kwargs_by_tool[tool_id] = dict(create_kwargs)
    return create_kwargs_by_tool


def _build_router(
    base_env_config: EnvironmentConfig,
    create_kwargs_by_tool: dict[str, dict[str, Any]],
) -> ToolRouter:
    cfg = copy.deepcopy(base_env_config)
    tool_env = cfg.tool_environment_map
    unknown_tool_ids = create_kwargs_by_tool.keys() - tool_env.keys()
    if unknown_tool_ids:
        raise ValueError(
            "Rollout create_kwargs contain unknown tool IDs: "
            f"{sorted(unknown_tool_ids)}"
        )
    env_kwargs_by_env: dict[str, dict[str, Any]] = {}
    for tool_id, create_kwargs in create_kwargs_by_tool.items():
        env_id = tool_env[tool_id]
        if env_id in env_kwargs_by_env and env_kwargs_by_env[env_id] != create_kwargs:
            raise ValueError(
                f"Tools provide conflicting create_kwargs for environment {env_id!r}."
            )
        env_kwargs_by_env[env_id] = create_kwargs
    for env in cfg.environments:
        if env.id in env_kwargs_by_env:
            env.env_kwargs = env_kwargs_by_env[env.id]
    return ToolRouter.from_environment_config(cfg).for_sample()


def _teardown(request_id: str) -> None:
    router = _ROUTERS.pop(request_id, None)
    if router is not None:
        router.close()


def get_or_build_router(
    agent_data: _RolloutData, base_env_config: EnvironmentConfig
) -> ToolRouter:
    """Return the rollout-scoped router."""
    request_id = agent_data.request_id
    router = _ROUTERS.get(request_id)
    if router is not None:
        return router
    router = _build_router(base_env_config, _rollout_create_kwargs(agent_data))
    _ROUTERS[request_id] = router
    weakref.finalize(agent_data, _teardown, request_id)
    return router
