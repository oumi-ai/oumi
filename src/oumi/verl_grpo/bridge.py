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

"""Local HTTP bridge that scores verl rollouts with oumi's registry rewards.

verl runs in an environment without oumi, so it cannot call oumi's registered
reward functions directly. The bridge runs in the oumi environment on a
localhost port; ``oumi_verl_ext.bridge_reward`` posts each rollout to it.
"""

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from omegaconf import DictConfig, OmegaConf

from oumi.core.trainers.verl_grpo_rewards import (
    combined_reward,
    resolve_reward_functions,
)
from oumi.exceptions import OumiConfigError
from oumi.utils.logging import logger
from oumi_verl_ext.bridge_client import (
    BRIDGE_REWARD_PATH,
    HEALTH_PATH,
    REWARD_PATH,
)

_LOOPBACK_HOST: Final = "127.0.0.1"


class _PayloadError(ValueError):
    """A request body that does not match the wire contract."""


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
        try:
            payload = json.loads(body)
        except ValueError as error:
            raise _PayloadError(f"request body is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise _PayloadError("request body must be a JSON object")
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


class _BridgeHTTPServer(ThreadingHTTPServer):
    # Must be a class attribute: the listen call runs in the constructor. A reward
    # burst larger than the default backlog of 5 would otherwise drop connections.
    request_queue_size = 128
    daemon_threads = True


class _BridgeHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: Mapping[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == HEALTH_PATH:
            self._send_json(HTTPStatus.OK, {"status": "ok", "live_routers": 0})
        else:
            self._send_json(
                HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path}"}
            )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != REWARD_PATH:
            self._send_json(
                HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path}"}
            )
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = _RewardRequest.from_body(self.rfile.read(length))
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
    """Serves oumi registry rewards on an ephemeral 127.0.0.1 port."""

    def __init__(self) -> None:
        """Creates a server that is not yet listening."""
        self._server: _BridgeHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def __enter__(self) -> "BridgeServer":
        """Binds the port and starts serving in a daemon thread."""
        server = _BridgeHTTPServer((_LOOPBACK_HOST, 0), _BridgeHandler)
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
        """Stops serving and releases the port."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None


def uses_bridge(verl_config: DictConfig) -> bool:
    """Whether verl's reward is the bridge shim."""
    path = OmegaConf.select(verl_config, "reward.custom_reward_function.path")
    return path == BRIDGE_REWARD_PATH


def check_bridge_config(verl_config: DictConfig) -> None:
    """Validates a bridge-backed reward config before the job starts.

    Raises:
        OumiConfigError: If a reward function is unregistered, the weights do
            not match the functions, or the run spans several nodes.
    """
    if not uses_bridge(verl_config):
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
    if (OmegaConf.select(verl_config, "trainer.nnodes") or 1) > 1:
        raise OumiConfigError(
            "The bridge reward serves rewards on localhost, so it supports a "
            "single node only (trainer.nnodes must be 1)."
        )
