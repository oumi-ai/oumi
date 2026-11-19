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

"""verl tool that runs oumi tool environments through the oumi bridge.

verl's tool agent loop builds this tool from its tool config in an environment
without oumi, so each tool call is forwarded to the bridge. The bridge keeps one
oumi router per rollout, keyed by verl's ``request_id``, and a background thread
asks it to close that router once verl drops the rollout's ``agent_data``. Instance
ids, step rewards and releases stay local: no oumi state belongs to an instance.

Fails closed: a bridge that is down, answers with an error or returns a malformed
body raises ``BridgeError``.
"""

import asyncio
import functools
import logging
import queue
import threading
import weakref
from typing import Any, Final, TypeVar
from uuid import uuid4

from verl.tools.base_tool import BaseTool  # pyright: ignore[reportMissingImports]
from verl.tools.schemas import (  # pyright: ignore[reportMissingImports]
    OpenAIFunctionToolSchema,
    ToolResponse,
)

from oumi_verl_ext.bridge_client import (
    TOOL_PATH_PREFIX,
    BridgeError,
    post_json,
    to_json_value,
)

TOOL_TIMEOUT_S: Final = 600.0

logger = logging.getLogger("oumi_verl_ext")
_T = TypeVar("_T")
# Rollouts whose router closes once their agent_data is collected. Shared by every
# tool instance, since one rollout can call several oumi tools.
_tracked_request_ids: set[str] = set()
_tracked_lock = threading.Lock()
_close_queue: queue.SimpleQueue[str] = queue.SimpleQueue()


def _queue_close(request_id: str) -> None:
    # A weakref callback on whichever thread frees agent_data, usually verl's event
    # loop and possibly inside _track_rollout, so it takes no lock and does no I/O.
    _tracked_request_ids.discard(request_id)
    _close_queue.put(request_id)


def _close_queued_rollouts() -> None:
    while True:
        request_id = _close_queue.get()
        try:
            post_json(
                TOOL_PATH_PREFIX + "close", {"request_id": request_id}, timeout_s=30.0
            )
        except BridgeError as error:
            logger.warning(
                f"Could not close the oumi tools of rollout {request_id}: {error}"
            )
        except Exception:
            # Any other error is a bug; the thread keeps closing later rollouts.
            logger.exception(f"Could not close the oumi tools of rollout {request_id}.")


@functools.cache
def _start_closer() -> None:
    # Closes still queued at interpreter exit are dropped; the bridge closes every
    # router when it stops.
    threading.Thread(
        target=_close_queued_rollouts, name="oumi-bridge-tool-closer", daemon=True
    ).start()


def _track_rollout(agent_data: object, request_id: str) -> None:
    with _tracked_lock:
        if request_id in _tracked_request_ids:
            return
        _tracked_request_ids.add(request_id)
        _start_closer()
    weakref.finalize(agent_data, _queue_close, request_id).atexit = False


async def _post(op: str, payload: dict[str, object]) -> dict[str, object]:
    return await asyncio.to_thread(
        post_json, TOOL_PATH_PREFIX + op, payload, timeout_s=TOOL_TIMEOUT_S
    )


def _expect(response: dict[str, object], key: str, kind: type[_T]) -> _T:
    value = response.get(key)
    if not isinstance(value, kind):
        raise BridgeError(f"Bridge response has no {kind.__name__} {key!r}: {response}")
    return value


class OumiBridgeTool(BaseTool):
    """A verl tool backed by the rollout's oumi router behind the bridge."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema) -> None:
        """Checks with the bridge that ``config["oumi_env_config"]`` has this tool.

        Raises:
            ValueError: If ``config`` has no ``oumi_env_config``.
            BridgeError: If the bridge is down or refuses the tool or its config.
        """
        super().__init__(config, tool_schema)
        env_config = config.get("oumi_env_config")
        if not env_config:
            raise ValueError(
                f"Tool '{self.name}' needs 'oumi_env_config' in its verl tool config."
            )
        self._env_config = env_config
        # The tool name is spelled out in both the verl tool config and the env
        # config; a mismatch would otherwise fail every call of every rollout.
        post_json(
            TOOL_PATH_PREFIX + "check",
            {"env_config": self._env_config, "tool_name": self.name},
            timeout_s=TOOL_TIMEOUT_S,
        )

    async def create(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> tuple[str, ToolResponse]:
        """Mints an instance id; ``create_kwargs`` reach the bridge with each call."""
        return instance_id or uuid4().hex, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        """Routes one call through the rollout's oumi router on the bridge."""
        request_id = agent_data.request_id
        _track_rollout(agent_data, request_id)
        response = await _post(
            "execute",
            {
                "env_config": self._env_config,
                "tool_name": self.name,
                "instance_id": instance_id,
                "request_id": request_id,
                "tools_kwargs": to_json_value(agent_data.tools_kwargs or {}),
                "parameters": to_json_value(parameters),
            },
        )
        return (
            ToolResponse(text=_expect(response, "text", str)),
            _expect(response, "reward", float),
            _expect(response, "metrics", dict),
        )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        """Returns 0.0: oumi tools give no step reward."""
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """Does nothing; the rollout's router outlives each instance."""
        return None
