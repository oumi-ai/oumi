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

import weakref
from collections.abc import Mapping
from functools import partial
from typing import Any, Protocol, TypedDict

from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.trainers.verl_agentic.router_worker import RouterWorker, next_lane
from oumi.environments.simulated_environment import SimulatedEnvironment

# Routers stay module-local because their environment connections are not picklable.
# Value is (id of the parent router, rollout clone) — see `get_or_build_router`.
_ROUTERS: dict[str, tuple[int, ToolRouter]] = {}
_WORKERS: dict[str, tuple[int, RouterWorker]] = {}


class _ToolKwargs(TypedDict, total=False):
    create_kwargs: Mapping[str, Any]


class _RolloutData(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def tools_kwargs(self) -> Mapping[str, _ToolKwargs | None] | None: ...


def _rollout_env_kwargs(
    agent_data: _RolloutData, tool_env_map: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Regroup this rollout's per-tool `create_kwargs` by owning environment."""
    create_kwargs_by_tool = {}
    for tool_id, entry in (agent_data.tools_kwargs or {}).items():
        if entry and (create_kwargs := entry.get("create_kwargs")):
            create_kwargs_by_tool[tool_id] = dict(create_kwargs)

    unknown_tool_ids = create_kwargs_by_tool.keys() - tool_env_map.keys()
    if unknown_tool_ids:
        raise ValueError(
            "Rollout create_kwargs contain unknown tool IDs: "
            f"{sorted(unknown_tool_ids)}"
        )
    env_kwargs_by_env_id: dict[str, dict[str, Any]] = {}
    for tool_id, create_kwargs in create_kwargs_by_tool.items():
        env_id = tool_env_map[tool_id]
        if (
            env_id in env_kwargs_by_env_id
            and env_kwargs_by_env_id[env_id] != create_kwargs
        ):
            raise ValueError(
                f"Tools provide conflicting create_kwargs for environment {env_id!r}."
            )
        env_kwargs_by_env_id[env_id] = create_kwargs
    return env_kwargs_by_env_id


def _teardown(request_id: str) -> None:
    entry = _ROUTERS.pop(request_id, None)
    if entry is not None:
        entry[1].close()


def _teardown_worker(request_id: str) -> None:
    """Queue thread-affine teardown without blocking a GC callback.

    Runs from a `weakref.finalize` callback, possibly from inside an allocation
    on a lane thread, so it must not lock or submit on that lane. `RouterWorker`
    keeps the lock-free entry point for exactly this path.
    """
    entry = _WORKERS.pop(request_id, None)
    if entry is not None:
        entry[1].close_from_finalizer()


def _reject_unsupported_envs(parent_router: ToolRouter) -> None:
    """Fail early for environments that cannot run on a worker thread.

    Thread mode has no event loop, so anything that would drive an inference
    engine would block on, or be corrupted by, a foreign loop. Rejecting here
    turns that into a config error at build time instead of a hang mid-rollout.
    """
    unsupported: list[str] = []
    env_by_id = getattr(parent_router, "env_by_id", None)
    if not isinstance(env_by_id, Mapping):
        # Test doubles and third-party routers need not expose env internals.
        return
    for env_id, env in env_by_id.items():
        if isinstance(env, SimulatedEnvironment):
            # Even stateless simulated envs fall back to the simulator engine for
            # any tool without a local executor, and that engine owns a loop.
            unsupported.append(f"{env_id!r} (SimulatedEnvironment)")
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValueError(
            "execution_mode: 'thread' cannot run these environments because they "
            f"need an asyncio-bound inference engine: {joined}. "
            "Use execution_mode: 'sync' (the default) for this config."
        )


def get_or_build_worker(
    agent_data: _RolloutData, parent_router: ToolRouter
) -> RouterWorker:
    """Return one thread-affine worker per rollout, shared by all its tools."""
    request_id = agent_data.request_id
    if request_id in _ROUTERS:
        raise ValueError("Tools in one rollout must use the same execution_mode.")
    entry = _WORKERS.get(request_id)
    if entry is not None:
        if entry[0] != id(parent_router):
            raise ValueError(
                "Oumi tools in one rollout must share a single environment config."
            )
        return entry[1]
    _reject_unsupported_envs(parent_router)
    overrides = _rollout_env_kwargs(agent_data, parent_router.tool_env_map)
    worker = RouterWorker(partial(parent_router.for_sample, overrides), next_lane())
    _WORKERS[request_id] = (id(parent_router), worker)
    # Never capture agent_data in the worker: that would keep its owner alive.
    finalizer = weakref.finalize(agent_data, _teardown_worker, request_id)
    # Lanes drain and close live routers themselves during process shutdown.
    finalizer.atexit = False
    return worker


def get_or_build_router(
    agent_data: _RolloutData, parent_router: ToolRouter
) -> ToolRouter:
    """Return the rollout-scoped clone of the process-wide `parent_router`.

    `request_id` must be unique per live rollout: teardown is tied to this
    `agent_data`, so a reused id would close the router still held by the other owner.

    Every Oumi tool in one rollout shares that rollout's router, so they must all
    name the same environment config. One config already covers many environments
    (`EnvironmentConfig.environments` is a list), so a second one is a mistake.
    """
    request_id = agent_data.request_id
    if request_id in _WORKERS:
        raise ValueError("Tools in one rollout must use the same execution_mode.")
    entry = _ROUTERS.get(request_id)
    if entry is not None:
        # `parent_router` is cached per config path, so identity == same config.
        if entry[0] != id(parent_router):
            raise ValueError(
                "Oumi tools in one rollout must share a single environment config; "
                "found tools built from two different `oumi_env_config` files. "
                "List every environment in one config instead."
            )
        return entry[1]
    router = parent_router.for_sample(
        _rollout_env_kwargs(agent_data, parent_router.tool_env_map)
    )
    _ROUTERS[request_id] = (id(parent_router), router)
    weakref.finalize(agent_data, _teardown, request_id)
    return router
