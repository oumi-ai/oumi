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

"""Constants and HTTP client shared by the verl-side shims of the oumi bridge."""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

BRIDGE_URL_ENV: Final = "OUMI_VERL_BRIDGE_URL"
HEALTH_PATH: Final = "/health"
REWARD_PATH: Final = "/reward"
TOOL_PATH_PREFIX: Final = "/tool/"
BRIDGE_REWARD_PATH: Final = "pkg://oumi_verl_ext.bridge_reward"
BRIDGE_REWARD_NAME: Final = "oumi_registry_reward"

HTTP_OK: Final = 200


class BridgeError(RuntimeError):
    """The bridge could not score a request."""


def to_json_value(value: object) -> object:
    """Converts mappings, sequences and numpy values to JSON-encodable Python."""
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [to_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def post_json(
    path: str, payload: Mapping[str, object], *, timeout_s: float
) -> dict[str, object]:
    """Posts JSON to the bridge once and returns the decoded JSON object.

    Never retries, since a retry could bill a judge call twice. Proxies are
    bypassed so a cluster proxy cannot reroute localhost.

    Raises:
        BridgeError: If the URL is unset, the transport fails or times out, the
            status is not 200, or the body is not a JSON object.
    """
    base_url = os.environ.get(BRIDGE_URL_ENV)
    if not base_url:
        raise BridgeError(f"{BRIDGE_URL_ENV} is not set; the bridge is not running.")
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        raise BridgeError(
            f"Bridge returned HTTP {error.code} for {path}: {_error_message(error)}"
        ) from error
    except (urllib.error.URLError, OSError) as error:
        raise BridgeError(f"Bridge request to {path} failed: {error}") from error
    if status != HTTP_OK:
        raise BridgeError(f"Bridge returned HTTP {status} for {path}.")
    try:
        decoded = json.loads(body)
    except ValueError as error:
        raise BridgeError(f"Bridge response for {path} is not JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise BridgeError(f"Bridge response for {path} is not a JSON object.")
    return decoded


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read())
    except ValueError:
        return error.reason
    if isinstance(body, dict) and "error" in body:
        return str(body["error"])
    return error.reason
