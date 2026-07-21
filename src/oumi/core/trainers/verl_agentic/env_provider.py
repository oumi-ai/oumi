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
from typing import Any

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.synthesis.tool_router import ToolRouter

# Module-level so it's never pickled onto agent_data (its DB connection is unpicklable).
_ROUTERS: dict[str, ToolRouter] = {}


def _rollout_create_kwargs(agent_data: Any) -> dict[str, dict[str, Any]]:
    """This rollout's create_kwargs, keyed by the tool id each belongs to."""
    return {
        tool_id: dict(entry["create_kwargs"])
        for tool_id, entry in (agent_data.tools_kwargs or {}).items()
        if entry and entry.get("create_kwargs")
    }


def _build_router(
    base_env_config: EnvironmentConfig,
    create_kwargs_by_tool: dict[str, dict[str, Any]],
) -> ToolRouter:
    """Build this rollout's router with its own isolated env instance(s).

    Each tool's create_kwargs replaces (not merges) the env_kwargs of the env
    that backs it, so a second env in the config never receives another's spec.
    """
    cfg = copy.deepcopy(base_env_config)
    tool_env = cfg.tool_environment_map
    env_kwargs_by_env = {
        tool_env[tool_id]: ck
        for tool_id, ck in create_kwargs_by_tool.items()
        if tool_id in tool_env
    }
    for env in cfg.environments:
        if env.id in env_kwargs_by_env:
            env.env_kwargs = env_kwargs_by_env[env.id]
    return ToolRouter.from_environment_config(cfg).for_sample()


def _teardown(request_id: str) -> None:
    """Pop and close the rollout's router (fired when its agent_data is GC'd)."""
    router = _ROUTERS.pop(request_id, None)
    if router is not None:
        router.close()


def get_or_build_router(
    agent_data: Any, base_env_config: EnvironmentConfig
) -> ToolRouter:
    """Return this rollout's router, building it on the first tool call."""
    request_id = agent_data.request_id
    router = _ROUTERS.get(request_id)
    if router is not None:
        return router
    router = _build_router(base_env_config, _rollout_create_kwargs(agent_data))
    _ROUTERS[request_id] = router
    weakref.finalize(agent_data, _teardown, request_id)
    return router
