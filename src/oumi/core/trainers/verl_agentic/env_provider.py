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


def _rollout_create_kwargs(agent_data: Any) -> dict[str, Any]:
    """The per-rollout env spec: first tools_kwargs entry carrying create_kwargs."""
    for entry in (agent_data.tools_kwargs or {}).values():
        create_kwargs = (entry or {}).get("create_kwargs")
        if create_kwargs:
            return dict(create_kwargs)
    return {}


def _build_router(
    base_env_config: EnvironmentConfig, create_kwargs: dict[str, Any]
) -> ToolRouter:
    """Build this rollout's router with its own isolated env instance(s)."""
    cfg = copy.deepcopy(base_env_config)
    for env in cfg.environments:
        # Replace not merge, so a stale placeholder can't collide with an incoming key.
        if create_kwargs:
            env.env_kwargs = dict(create_kwargs)
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
