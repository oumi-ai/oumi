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

"""Expose Oumi Enterprise platform resources through the oumi-mcp server.

When the user has ``OUMI_API_KEY`` set, an MCP client connected to
``oumi-mcp`` can list and inspect platform datasets, models, judges, and
operations — making it easy for tooling like Claude Code or Cursor to
ground its reasoning in the user's enterprise resources.

Credentials are resolved lazily on each tool call. A missing or invalid
key returns a structured error inside the tool response rather than
crashing the MCP server, so the rest of oumi-mcp keeps working.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _client_or_error() -> tuple[Any, str | None]:
    """Resolve a platform :class:`Client`, or return an error message instead.

    Returns:
        ``(client, None)`` on success, ``(None, error_string)`` if the user
        is not logged in or the credentials file is malformed.
    """
    try:
        from oumi.platform import Client, PlatformError
    except ImportError as exc:  # pragma: no cover - import guard
        return None, f"oumi.platform is not available: {exc}"
    try:
        return Client(), None
    except PlatformError as exc:
        return None, str(exc)


def _error_result(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _ok_result(payload: Any) -> dict[str, Any]:
    return {"ok": True, "result": payload}


# ---------------------------- datasets ----------------------------


def list_platform_datasets(project: str | None = None) -> dict[str, Any]:
    """List datasets in the configured Oumi Enterprise project.

    Args:
        project: Override the default platform project id.

    Returns:
        A dict with ``ok``, plus either ``result`` (the list payload) or
        ``error`` (a human-readable message).
    """
    client, err = _client_or_error()
    if err:
        return _error_result(err)
    try:
        return _ok_result(client.datasets.list(project_id=project))
    except Exception as exc:
        return _error_result(f"datasets.list failed: {exc}")


def get_platform_dataset(
    dataset_id: str, project: str | None = None
) -> dict[str, Any]:
    """Get a single platform dataset by id.

    Args:
        dataset_id: The platform's dataset id.
        project: Override the default platform project id.
    """
    client, err = _client_or_error()
    if err:
        return _error_result(err)
    try:
        return _ok_result(client.datasets.get(dataset_id, project_id=project))
    except Exception as exc:
        return _error_result(f"datasets.get({dataset_id}) failed: {exc}")


# ---------------------------- models ----------------------------


def list_platform_models(project: str | None = None) -> dict[str, Any]:
    """List models in the configured Oumi Enterprise project.

    Args:
        project: Override the default platform project id.
    """
    client, err = _client_or_error()
    if err:
        return _error_result(err)
    try:
        return _ok_result(client.models.list(project_id=project))
    except Exception as exc:
        return _error_result(f"models.list failed: {exc}")


def get_platform_model(
    model_id: str, project: str | None = None
) -> dict[str, Any]:
    """Get a single platform model by id.

    Args:
        model_id: The platform's model id.
        project: Override the default platform project id.
    """
    client, err = _client_or_error()
    if err:
        return _error_result(err)
    try:
        return _ok_result(client.models.get(model_id, project_id=project))
    except Exception as exc:
        return _error_result(f"models.get({model_id}) failed: {exc}")


# ---------------------------- judges ----------------------------


def list_platform_judges(project: str | None = None) -> dict[str, Any]:
    """List judges (evaluators of type ``judge``) in your project.

    Args:
        project: Override the default platform project id.
    """
    client, err = _client_or_error()
    if err:
        return _error_result(err)
    try:
        return _ok_result(
            client.evaluators.list(
                evaluator_type="judge", project_id=project
            )
        )
    except Exception as exc:
        return _error_result(f"evaluators.list failed: {exc}")


# ---------------------------- operations ----------------------------


def get_platform_operation(
    operation_id: str, project: str | None = None
) -> dict[str, Any]:
    """Look up the status of a long-running platform operation.

    Args:
        operation_id: The id returned when the job was submitted.
        project: Override the default platform project id.
    """
    client, err = _client_or_error()
    if err:
        return _error_result(err)
    try:
        return _ok_result(client.operations.get(operation_id, project_id=project))
    except Exception as exc:
        return _error_result(
            f"operations.get({operation_id}) failed: {exc}"
        )


def register_platform_tools(mcp: Any) -> None:
    """Wire the platform tools into the given FastMCP instance.

    Called from :mod:`oumi.mcp.server` during server construction. Kept as
    a separate function so the module can be imported without immediately
    decorating anything (e.g., during unit tests).
    """
    mcp.tool()(list_platform_datasets)
    mcp.tool()(get_platform_dataset)
    mcp.tool()(list_platform_models)
    mcp.tool()(get_platform_model)
    mcp.tool()(list_platform_judges)
    mcp.tool()(get_platform_operation)
