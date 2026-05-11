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

"""Shared helper backing the ``--remote`` flag on ``train``/``evaluate``/etc.

The helper takes a fully-validated oumi config and submits it to the
platform's ``POST /v1/projects/{}/jobs:submit`` endpoint, then optionally
waits for completion. It is intentionally independent of the
:class:`OumiPlatformCloud` launcher; this is the simpler one-shot path for
users who don't want to think about ``JobConfig`` wrappers.
"""

import dataclasses
import logging
import sys
from typing import Any

from oumi.platform.client import Client
from oumi.platform.exceptions import PlatformAPIError, PlatformError
from oumi.utils.logging import logger

_VALID_KINDS = frozenset({"train", "evaluate", "synth", "judge", "infer"})


def submit_remote_run(
    config: Any,
    *,
    kind: str,
    project_id: str | None = None,
    name: str | None = None,
    wait: bool = True,
    client: Client | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Submit ``config`` to the platform and optionally wait for completion.

    Args:
        config: A oumi config dataclass (e.g. ``TrainingConfig``,
            ``EvaluationConfig``). Serialized to a dict via
            :func:`dataclasses.asdict`.
        kind: One of ``train``, ``evaluate``, ``synth``, ``judge``, ``infer``.
            Tells the platform which workflow to dispatch.
        project_id: Override the platform default project.
        name: Optional human-readable submission name (shown in the web UI).
        wait: When ``True`` (default), block until the platform operation
            reaches a terminal status, raising on failure. When ``False``,
            return immediately with the initial submission payload.
        client: Override the platform :class:`Client`. Defaults to
            :func:`oumi.platform.get_default_client`.
        log: Override the logger.

    Returns:
        The final operation payload (when ``wait=True``) or the initial
        submission payload (when ``wait=False``).
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Unknown job kind {kind!r}; expected one of "
            f"{sorted(_VALID_KINDS)}."
        )
    log = log or logger

    if client is None:
        from oumi.platform.client import get_default_client

        resolved_client = get_default_client()
    else:
        resolved_client = client

    payload: dict[str, Any] = {
        "kind": kind,
        "config": _serialize_config(config),
    }
    if name:
        payload["displayName"] = name

    project = resolved_client._resolve_project_id(project_id)
    try:
        op = resolved_client.request(
            "POST",
            f"/v1/projects/{project}/jobs:submit",
            json_body=payload,
        )
    except PlatformAPIError as exc:
        if exc.status_code == 404:
            raise PlatformError(
                "--remote requires the Oumi Enterprise platform endpoint "
                f"/v1/projects/{project}/jobs:submit, which the server at "
                f"{resolved_client.credentials.api_url} does not expose. "
                "Upgrade the platform or run the job locally without --remote."
            ) from exc
        raise

    op_id = _operation_id(op)
    log.info(
        f"Submitted {kind} job to {resolved_client.credentials.api_url} "
        f"(operation id: {op_id})."
    )

    if not wait:
        return op

    log.info("Waiting for operation to complete… (use --detach to skip)")
    return resolved_client.operations.wait(op_id, project_id=project)


def _serialize_config(config: Any) -> dict[str, Any]:
    """Convert a config dataclass to a JSON-safe ``dict``."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if isinstance(config, dict):
        return config
    raise TypeError(
        f"Cannot serialize {type(config).__name__}; expected a dataclass or "
        "dict."
    )


def _operation_id(op: Any) -> str:
    if isinstance(op, dict):
        for key in ("id", "operationId", "name"):
            value = op.get(key)
            if value is not None:
                return str(value)
    raise PlatformError(
        f"Platform :submit response missing an operation id: {op!r}"
    )


def print_operation_summary(op: dict[str, Any]) -> None:
    """Write a brief operation summary to stderr.

    Used by ``--remote --detach`` so the user sees the operation id without
    having to parse JSON.
    """
    op_id = _operation_id(op) if isinstance(op, dict) else "?"
    status = op.get("status", "unknown") if isinstance(op, dict) else "unknown"
    print(f"Operation: {op_id}  status: {status}", file=sys.stderr)
