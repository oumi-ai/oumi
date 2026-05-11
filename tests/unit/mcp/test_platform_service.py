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

from unittest.mock import patch

import httpx
import pytest

from oumi.mcp.platform_service import (
    get_platform_dataset,
    get_platform_operation,
    list_platform_datasets,
    list_platform_judges,
    list_platform_models,
    register_platform_tools,
)
from oumi.platform import Client, Credentials

_TEST_API_URL = "https://platform.test"
_TEST_CREDS = Credentials(
    api_url=_TEST_API_URL, api_key="k", project_id="proj-1"
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "OUMI_API_URL",
        "OUMI_API_KEY",
        "OUMI_PROJECT_ID",
        "OUMI_CREDENTIALS_FILE",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(k, raising=False)


def _client(handler) -> Client:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        base_url=_TEST_API_URL,
        transport=transport,
        headers={"Authorization": "Bearer k", "X-API-Key": "k"},
    )
    return Client(credentials=_TEST_CREDS, http_client=http)


def test_list_platform_datasets_returns_payload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"datasets": [{"id": 1, "displayName": "a"}]}
        )

    client = _client(handler)
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(client, None),
    ):
        result = list_platform_datasets()

    assert result["ok"] is True
    assert result["result"]["datasets"][0]["displayName"] == "a"


def test_get_platform_dataset_routes_correctly():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"id": 5, "displayName": "x"})

    client = _client(handler)
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(client, None),
    ):
        result = get_platform_dataset("5")

    assert result["ok"] is True
    assert seen[0] == "/v1/projects/proj-1/datasets/5"


def test_list_platform_models_returns_payload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"models": [{"id": 7, "displayName": "m"}]}
        )

    client = _client(handler)
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(client, None),
    ):
        result = list_platform_models()

    assert result["ok"] is True
    assert result["result"]["models"][0]["id"] == 7


def test_list_platform_judges_filters_to_judge():
    seen_params: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params).get("evaluatorType", ""))
        return httpx.Response(
            200, json={"evaluators": [{"id": 1, "evaluatorType": "judge"}]}
        )

    client = _client(handler)
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(client, None),
    ):
        result = list_platform_judges()

    assert result["ok"] is True
    assert seen_params == ["judge"]


def test_get_platform_operation_returns_status():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": 42, "status": "running", "done": False}
        )

    client = _client(handler)
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(client, None),
    ):
        result = get_platform_operation("42")

    assert result["ok"] is True
    assert result["result"]["status"] == "running"


def test_not_logged_in_returns_structured_error():
    """An invalid platform client must surface as an error response, not a crash."""
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(None, "Not logged in"),
    ):
        result = list_platform_datasets()

    assert result == {"ok": False, "error": "Not logged in"}


def test_underlying_failure_is_wrapped():
    """Network/HTTP errors come back as ok: False, error: ... not exceptions."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    client = _client(handler)
    with patch(
        "oumi.mcp.platform_service._client_or_error",
        return_value=(client, None),
    ):
        result = list_platform_datasets()

    assert result["ok"] is False
    assert "datasets.list failed" in result["error"]


def test_register_platform_tools_decorates_each_function():
    """Wiring helper should call mcp.tool() for each exported function."""
    captured: list = []

    class _FakeMCP:
        def tool(self):
            def decorator(fn):
                captured.append(fn.__name__)
                return fn

            return decorator

    register_platform_tools(_FakeMCP())

    assert captured == [
        "list_platform_datasets",
        "get_platform_dataset",
        "list_platform_models",
        "get_platform_model",
        "list_platform_judges",
        "get_platform_operation",
    ]
