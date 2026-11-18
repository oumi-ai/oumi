# Copyright 2026 - Oumi
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

"""Per-rollout oumi tool routers that the verl bridge serves tool calls from.

Each verl rollout gets its own clone of a ``ToolRouter``, keyed by verl's
``request_id``. A clone's environments can hold thread-bound resources, such as a
SQLite connection, while the bridge answers every request on a new thread. So each
rollout owns one worker thread, and its router is built, called and closed only
there. This also serializes a rollout's parallel tool calls.
"""

import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.tool_call import ToolResult
from oumi.utils.logging import logger


def _rollout_env_kwargs(
    tools_kwargs: Mapping[str, Mapping[str, object] | None],
    tool_env_map: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Regroups a rollout's per-tool ``create_kwargs`` by owning environment."""
    create_kwargs_by_tool: dict[str, dict[str, object]] = {}
    for tool_id, entry in tools_kwargs.items():
        create_kwargs = entry.get("create_kwargs") if entry else None
        if not create_kwargs:
            continue
        if not isinstance(create_kwargs, Mapping):
            raise ValueError(f"create_kwargs of tool {tool_id!r} must be a mapping.")
        create_kwargs_by_tool[tool_id] = dict(create_kwargs)

    unknown_tool_ids = create_kwargs_by_tool.keys() - tool_env_map.keys()
    if unknown_tool_ids:
        raise ValueError(
            "Rollout create_kwargs contain unknown tool IDs: "
            f"{sorted(unknown_tool_ids)}"
        )
    env_kwargs_by_env_id: dict[str, dict[str, object]] = {}
    for tool_id, create_kwargs in create_kwargs_by_tool.items():
        env_id = tool_env_map[tool_id]
        if env_kwargs_by_env_id.get(env_id, create_kwargs) != create_kwargs:
            raise ValueError(
                f"Tools provide conflicting create_kwargs for environment {env_id!r}."
            )
        env_kwargs_by_env_id[env_id] = create_kwargs
    return env_kwargs_by_env_id


class RolloutToolSession:
    """One rollout's router, built, called and closed on the rollout's own thread."""

    def __init__(
        self,
        parent: ToolRouter,
        tools_kwargs: Mapping[str, Mapping[str, object] | None],
    ) -> None:
        """Starts the rollout's worker thread and builds the router on it."""
        self.parent = parent
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="oumi-rollout-tools"
        )
        self._router: Future[ToolRouter] = self._worker.submit(
            lambda: parent.for_sample(
                _rollout_env_kwargs(tools_kwargs, parent.tool_env_map)
            )
        )

    def route(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Runs ``calls`` on the rollout's thread and waits for their results."""
        router = self._router.result()
        return self._worker.submit(router.route_batch, calls).result()

    def _wait_until_built(self) -> None:
        self._router.result()

    def _close(self) -> None:
        try:
            self._worker.submit(self._close_router).result()
        finally:
            self._worker.shutdown()

    def _close_router(self) -> None:
        # The build was this thread's first task, so it has already finished.
        if self._router.exception() is None:
            self._router.result().close()


class RolloutRouters:
    """Thread-safe map from a verl ``request_id`` to that rollout's tool session.

    Parent routers are built once per environment config path; each rollout's
    session clones its config's parent.
    """

    def __init__(self) -> None:
        """Creates an empty map."""
        self._lock = threading.Lock()
        self._sessions: dict[str, RolloutToolSession] = {}
        self._parent_lock = threading.Lock()
        self._configs: dict[str, EnvironmentConfig] = {}
        self._parents: dict[str, ToolRouter] = {}

    def check_tool(self, env_config: str, tool_name: str) -> None:
        """Checks that ``env_config`` loads and defines ``tool_name``.

        Raises:
            ValueError: If the tool is not in the config, listing the known tools.
        """
        known_tools = self._config(env_config).tool_environment_map
        if tool_name not in known_tools:
            raise ValueError(
                f"Tool '{tool_name}' is not defined in {env_config}. "
                f"Known tools: {sorted(known_tools)}"
            )

    def get_or_build(
        self,
        request_id: str,
        *,
        env_config: str,
        tools_kwargs: Mapping[str, Mapping[str, object] | None],
    ) -> RolloutToolSession:
        """Returns the rollout's session, building its router on first use.

        ``request_id`` must be unique per live rollout: a reused id would share
        the other rollout's environments.

        Raises:
            ValueError: If ``tools_kwargs`` names an unknown tool or gives one
                environment conflicting ``create_kwargs``, or if the rollout
                already uses a different environment config.
        """
        parent = self._parent(env_config)
        with self._lock:
            session = self._sessions.get(request_id)
            if session is None:
                session = RolloutToolSession(parent, tools_kwargs)
                self._sessions[request_id] = session
        # Parents are cached per config path, so identity means the same config.
        if session.parent is not parent:
            raise ValueError(
                "Oumi tools in one rollout must share a single environment config; "
                "found tools built from two different `oumi_env_config` files. "
                "List every environment in one config instead."
            )
        try:
            session._wait_until_built()
        except BaseException:
            self._evict(request_id, session)
            raise
        return session

    def close(self, request_id: str) -> bool:
        """Closes and forgets the rollout's router; False if it had none."""
        with self._lock:
            session = self._sessions.pop(request_id, None)
        if session is None:
            return False
        session._close()
        return True

    def close_all(self) -> int:
        """Closes every live router and returns how many there were."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session._close()
            except Exception:
                logger.exception("Failed to close a rollout's tool router.")
        return len(sessions)

    @property
    def live_count(self) -> int:
        """The number of rollouts with a live router."""
        with self._lock:
            return len(self._sessions)

    def _evict(self, request_id: str, session: RolloutToolSession) -> None:
        # Whoever removes a session from the map closes it, so it closes once.
        with self._lock:
            if self._sessions.get(request_id) is not session:
                return
            del self._sessions[request_id]
        session._close()

    def _config(self, env_config: str) -> EnvironmentConfig:
        with self._parent_lock:
            config = self._configs.get(env_config)
            if config is None:
                config = EnvironmentConfig.from_yaml(env_config)
                config.finalize_and_validate()
                self._configs[env_config] = config
            return config

    def _parent(self, env_config: str) -> ToolRouter:
        config = self._config(env_config)
        with self._parent_lock:
            parent = self._parents.get(env_config)
            if parent is None:
                parent = ToolRouter.from_environment_config(config)
                self._parents[env_config] = parent
            return parent
