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

"""Local HTTP bridge that serves oumi rewards and tool environments to verl.

verl runs in an environment without oumi, so it cannot call oumi's registered
reward functions or tool environments directly. The bridge runs in the oumi
environment on a localhost port; ``oumi_verl_ext.bridge_reward`` posts each
rollout to it, and ``oumi_verl_ext.bridge_tool`` posts each tool call.
"""

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from oumi.core.trainers.verl_grpo_rewards import (
    combined_reward,
    resolve_reward_functions,
)
from oumi.exceptions import OumiConfigError
from oumi.utils.logging import logger
from oumi.verl_grpo.tool_routers import RolloutRouters
from oumi_verl_ext.bridge_client import (
    BRIDGE_REWARD_PATH,
    HEALTH_PATH,
    REWARD_PATH,
    TOOL_PATH_PREFIX,
)

_LOOPBACK_HOST: Final = "127.0.0.1"


class _PayloadError(ValueError):
    """A request body that does not match the wire contract."""


def _json_object(body: bytes) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except ValueError as error:
        raise _PayloadError(f"request body is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise _PayloadError("request body must be a JSON object")
    return payload


def _str_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _PayloadError(f"'{key}' must be a string")
    return value


def _object_field(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _PayloadError(f"'{key}' must be null or an object")
    return value


@dataclass(frozen=True)
class _RewardRequest:
    data_source: object
    solution_str: str
    ground_truth: object
    extra_info: object
    reward_function_names: list[str]
    reward_weights: list[float] | None
    reward_function_kwargs: dict[str, dict[str, object]] | None

    @classmethod
    def from_body(cls, body: bytes) -> "_RewardRequest":
        payload = _json_object(body)
        solution_str = payload.get("solution_str")
        if not isinstance(solution_str, str):
            raise _PayloadError("'solution_str' must be a string")
        names = payload.get("reward_functions")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
        ):
            raise _PayloadError("'reward_functions' must be a non-empty list of names")
        weights = payload.get("reward_weights")
        if weights is not None:
            if not isinstance(weights, list) or not all(
                isinstance(weight, (int, float)) and not isinstance(weight, bool)
                for weight in weights
            ):
                raise _PayloadError(
                    "'reward_weights' must be null or a list of numbers"
                )
            if len(weights) != len(names):
                raise _PayloadError(
                    f"'reward_weights' has {len(weights)} entries for "
                    f"{len(names)} reward functions"
                )
        kwargs = payload.get("reward_function_kwargs")
        if kwargs is not None and (
            not isinstance(kwargs, dict)
            or not all(isinstance(value, dict) for value in kwargs.values())
        ):
            raise _PayloadError(
                "'reward_function_kwargs' must be null or an object of objects"
            )
        return cls(
            data_source=payload.get("data_source"),
            solution_str=solution_str,
            ground_truth=payload.get("ground_truth"),
            extra_info=payload.get("extra_info"),
            reward_function_names=names,
            reward_weights=weights,
            reward_function_kwargs=kwargs,
        )


@dataclass(frozen=True)
class _ToolCheckRequest:
    env_config: str
    tool_name: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "_ToolCheckRequest":
        return cls(
            env_config=_str_field(payload, "env_config"),
            tool_name=_str_field(payload, "tool_name"),
        )


@dataclass(frozen=True)
class _ToolExecuteRequest:
    env_config: str
    tool_name: str
    request_id: str
    tools_kwargs: dict[str, dict[str, object] | None]
    parameters: dict[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "_ToolExecuteRequest":
        tools_kwargs = _object_field(payload, "tools_kwargs")
        if not all(
            value is None or isinstance(value, dict) for value in tools_kwargs.values()
        ):
            raise _PayloadError("'tools_kwargs' values must be null or objects")
        return cls(
            env_config=_str_field(payload, "env_config"),
            tool_name=_str_field(payload, "tool_name"),
            request_id=_str_field(payload, "request_id"),
            tools_kwargs=cast(dict[str, dict[str, object] | None], tools_kwargs),
            parameters=_object_field(payload, "parameters"),
        )


def _check_tool(
    routers: RolloutRouters, payload: dict[str, object]
) -> dict[str, object]:
    request = _ToolCheckRequest.from_payload(payload)
    routers.check_tool(request.env_config, request.tool_name)
    return {}


def _create_tool(
    routers: RolloutRouters, payload: dict[str, object]
) -> dict[str, object]:
    # Only the id is minted here: a rollout's create_kwargs reach the bridge with
    # each execute's tools_kwargs, so its router is built on first use.
    instance_id = payload.get("instance_id")
    if instance_id is not None and not isinstance(instance_id, str):
        raise _PayloadError("'instance_id' must be null or a string")
    return {"instance_id": instance_id or uuid4().hex}


def _execute_tool(
    routers: RolloutRouters, payload: dict[str, object]
) -> dict[str, object]:
    request = _ToolExecuteRequest.from_payload(payload)
    session = routers.get_or_build(
        request.request_id,
        env_config=request.env_config,
        tools_kwargs=request.tools_kwargs,
    )
    # verl sends None for unset optional args; the executor should see them absent.
    arguments = {
        name: value for name, value in request.parameters.items() if value is not None
    }
    try:
        output = session.route([(request.tool_name, arguments)])[0].output
    except Exception as error:
        # A bad tool call is an observation the policy can recover from, not a
        # reason to fail the rollout.
        logger.warning(f"Tool '{request.tool_name}' call failed: {error}")
        return {"text": f"Tool error: {error}", "reward": 0.0, "metrics": {}}
    text = output if isinstance(output, str) else json.dumps(output)
    return {"text": text, "reward": 0.0, "metrics": {}}


def _calc_tool_reward(
    routers: RolloutRouters, payload: dict[str, object]
) -> dict[str, object]:
    _str_field(payload, "instance_id")
    return {"reward": 0.0}


def _release_tool(
    routers: RolloutRouters, payload: dict[str, object]
) -> dict[str, object]:
    _str_field(payload, "instance_id")
    return {}


def _close_rollout(
    routers: RolloutRouters, payload: dict[str, object]
) -> dict[str, object]:
    return {"closed": routers.close(_str_field(payload, "request_id"))}


_TOOL_OPS: Final = {
    "check": _check_tool,
    "create": _create_tool,
    "execute": _execute_tool,
    "calc_reward": _calc_tool_reward,
    "release": _release_tool,
    "close": _close_rollout,
}


class _BridgeHTTPServer(ThreadingHTTPServer):
    # Must be a class attribute: the listen call runs in the constructor. A reward
    # burst larger than the default backlog of 5 would otherwise drop connections.
    request_queue_size = 128
    daemon_threads = True

    def __init__(self, routers: RolloutRouters) -> None:
        super().__init__((_LOOPBACK_HOST, 0), _BridgeHandler)
        self.routers = routers


class _BridgeHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: Mapping[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @property
    def _routers(self) -> RolloutRouters:
        return cast(_BridgeHTTPServer, self.server).routers

    def do_GET(self) -> None:  # noqa: N802
        if self.path == HEALTH_PATH:
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "live_routers": self._routers.live_count},
            )
        else:
            self._send_json(
                HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path}"}
            )

    def do_POST(self) -> None:  # noqa: N802
        tool_op = (
            self.path.removeprefix(TOOL_PATH_PREFIX)
            if self.path.startswith(TOOL_PATH_PREFIX)
            else None
        )
        if self.path != REWARD_PATH and tool_op not in _TOOL_OPS:
            self._send_json(
                HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path}"}
            )
            return
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Content-Length must be a non-negative integer"},
            )
            return
        body = self.rfile.read(length)
        if tool_op is None:
            self._serve_reward(body)
        else:
            self._serve_tool(tool_op, body)

    def _serve_tool(self, op: str, body: bytes) -> None:
        try:
            response = _TOOL_OPS[op](self._routers, _json_object(body))
        except (ValueError, OumiConfigError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:
            logger.exception(f"Bridge tool op '{op}' failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(error).__name__}: {error}"},
            )
            return
        self._send_json(HTTPStatus.OK, response)

    def _serve_reward(self, body: bytes) -> None:
        try:
            request = _RewardRequest.from_body(body)
        except _PayloadError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        try:
            scores = combined_reward(
                request.data_source,
                request.solution_str,
                request.ground_truth,
                request.extra_info,
                reward_function_names=request.reward_function_names,
                reward_weights=request.reward_weights,
                reward_function_kwargs=request.reward_function_kwargs,
            )
            body = json.dumps({"scores": scores}).encode()
        except Exception as error:
            logger.exception(
                f"Bridge reward scoring failed for {request.reward_function_names}"
            )
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(error).__name__}: {error}"},
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("bridge: " + format, *args)


class BridgeServer:
    """Serves oumi rewards and tool environments on an ephemeral 127.0.0.1 port."""

    def __init__(self) -> None:
        """Creates a server that is not yet listening."""
        self._server: _BridgeHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._routers = RolloutRouters()
        self.url = ""

    def __enter__(self) -> "BridgeServer":
        """Binds the port and starts serving in a daemon thread."""
        server = _BridgeHTTPServer(self._routers)
        self._server = server
        self.url = f"http://{_LOOPBACK_HOST}:{server.server_address[1]}"
        self._thread = threading.Thread(
            target=server.serve_forever,
            # A short poll keeps shutdown() from waiting out the 0.5 s default.
            kwargs={"poll_interval": 0.05},
            name="oumi-verl-bridge",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stops serving, releases the port and closes every live tool router."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None
        logger.info(f"closed {self._routers.close_all()} live tool routers")


def uses_bridge(verl_config: DictConfig) -> bool:
    """Whether verl's reward or one of its tools goes through the bridge."""
    return _uses_bridge_reward(verl_config) or _uses_bridge_tool(verl_config)


def _uses_bridge_reward(verl_config: DictConfig) -> bool:
    path = OmegaConf.select(verl_config, "reward.custom_reward_function.path")
    return path == BRIDGE_REWARD_PATH


def _uses_bridge_tool(verl_config: DictConfig) -> bool:
    path = OmegaConf.select(
        verl_config, "actor_rollout_ref.rollout.multi_turn.tool_config_path"
    )
    # verl also resolves paths inside its own package, whose bundled tool
    # configs never name an oumi tool.
    if not path or not Path(path).is_file():
        return False
    tool_config = OmegaConf.load(path)
    tools = tool_config.get("tools") if isinstance(tool_config, DictConfig) else None
    return any(
        isinstance(tool, DictConfig)
        and str(tool.get("class_name", "")).startswith("oumi_verl_ext.")
        for tool in tools or []
    )


def check_bridge_config(verl_config: DictConfig) -> None:
    """Validates a bridge-backed reward or tool config before the job starts.

    Raises:
        OumiConfigError: If a reward function is unregistered, the weights do
            not match the functions, a bridge tool runs with full determinism,
            or the run spans several nodes.
    """
    uses_reward = _uses_bridge_reward(verl_config)
    uses_tool = _uses_bridge_tool(verl_config)
    if not (uses_reward or uses_tool):
        return
    if (OmegaConf.select(verl_config, "trainer.nnodes") or 1) > 1:
        raise OumiConfigError(
            "The bridge serves oumi on localhost, so it supports a single node "
            "only (trainer.nnodes must be 1)."
        )
    if uses_tool and OmegaConf.select(
        verl_config, "actor_rollout_ref.rollout.full_determinism"
    ):
        raise OumiConfigError(
            "The bridge tool keeps one router per verl request id, but "
            "actor_rollout_ref.rollout.full_determinism makes verl reuse "
            "'det-<priority>' request ids across concurrent rollouts. Disable it."
        )
    if not uses_reward:
        return
    raw_kwargs = OmegaConf.select(
        verl_config, "reward.custom_reward_function.reward_kwargs"
    )
    reward_kwargs = (
        OmegaConf.to_container(raw_kwargs, resolve=True)
        if isinstance(raw_kwargs, DictConfig)
        else raw_kwargs
    )
    if reward_kwargs is None:
        reward_kwargs = {}
    if not isinstance(reward_kwargs, dict):
        raise OumiConfigError(
            "reward.custom_reward_function.reward_kwargs must be a mapping."
        )
    names = reward_kwargs.get("reward_functions")
    if not isinstance(names, list) or not names:
        raise OumiConfigError(
            "The bridge reward needs a non-empty 'reward_functions' list in "
            "reward.custom_reward_function.reward_kwargs."
        )
    try:
        resolve_reward_functions([str(name) for name in names])
    except ValueError as error:
        raise OumiConfigError(str(error)) from error
    weights = reward_kwargs.get("reward_weights")
    if weights is not None and (
        not isinstance(weights, list)
        or not all(
            isinstance(weight, (int, float)) and not isinstance(weight, bool)
            for weight in weights
        )
    ):
        raise OumiConfigError("reward_weights must be a list of numbers.")
    if weights is not None and len(weights) != len(names):
        raise OumiConfigError(
            f"Number of reward_weights ({len(weights)}) must match the number "
            f"of reward functions ({len(names)})."
        )
