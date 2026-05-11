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

import json
from unittest.mock import patch

import httpx
import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.platform import (
    list_datasets,
    list_judges,
    list_models_,
    login,
    logout,
    operation_status,
    operation_stop,
    pull_dataset,
    pull_model,
    whoami,
)
from oumi.platform import Client, Credentials

_TEST_API_URL = "https://platform.test"
_TEST_CREDS = Credentials(
    api_url=_TEST_API_URL, api_key="testkey1234", project_id="proj-1"
)

runner = CliRunner()


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
        headers={"Authorization": "Bearer testkey1234", "X-API-Key": "testkey1234"},
    )
    return Client(credentials=_TEST_CREDS, http_client=http)


def _app_with(command, name: str | None = None) -> typer.Typer:
    app = typer.Typer()
    if name:
        app.command(name=name)(command)
    else:
        app.command()(command)
    return app


# ---------------------------------------------------------------- auth ---


def test_login_writes_credentials_file(monkeypatch, tmp_path):
    creds_path = tmp_path / "credentials.json"
    monkeypatch.setenv("OUMI_CREDENTIALS_FILE", str(creds_path))

    app = _app_with(login)
    result = runner.invoke(
        app,
        ["--api-url", "https://api.example", "--api-key", "abc", "--project", "p1"],
    )

    assert result.exit_code == 0
    saved = json.loads(creds_path.read_text())
    assert saved == {
        "api_url": "https://api.example",
        "api_key": "abc",
        "project_id": "p1",
    }


def test_logout_removes_credentials_file(tmp_path):
    creds_path = tmp_path / "credentials.json"
    creds_path.write_text("{}")

    app = _app_with(logout)
    result = runner.invoke(app, ["--path", str(creds_path)])

    assert result.exit_code == 0
    assert not creds_path.exists()


def test_logout_is_idempotent_when_file_missing(tmp_path):
    creds_path = tmp_path / "credentials.json"
    app = _app_with(logout)

    result = runner.invoke(app, ["--path", str(creds_path)])

    assert result.exit_code == 0
    assert "No credentials file" in result.stdout


def test_whoami_prints_resolved_credentials(monkeypatch, tmp_path):
    creds_path = tmp_path / "credentials.json"
    creds_path.write_text(
        json.dumps(
            {
                "api_url": "https://api.example",
                "api_key": "secret9999",
                "project_id": "p-7",
            }
        )
    )
    monkeypatch.setenv("OUMI_CREDENTIALS_FILE", str(creds_path))

    result = runner.invoke(_app_with(whoami), [])

    assert result.exit_code == 0
    assert "https://api.example" in result.stdout
    assert "p-7" in result.stdout
    assert "9999" in result.stdout  # last 4 shown
    assert "secret" not in result.stdout  # full key hidden


# ----------------------------------------------------- datasets / models ---


def test_list_datasets_prints_table():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "datasets": [
                    {
                        "id": 1,
                        "displayName": "first",
                        "schemaType": "conversation",
                        "version": 2,
                        "versionName": "v2",
                    },
                ]
            },
        )

    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(_app_with(list_datasets), [])

    assert result.exit_code == 0
    assert "first" in result.stdout
    assert "conversation" in result.stdout


def test_pull_dataset_downloads(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=b'{"a": 1}\n')
        if request.url.path.endswith("/datasets/42:download"):
            return httpx.Response(200, json={"url": "https://storage.test/d.jsonl"})
        raise AssertionError(f"Unexpected URL: {request.url}")

    target = tmp_path / "out.jsonl"
    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(
            _app_with(pull_dataset), ["42", "--out", str(target)]
        )

    assert result.exit_code == 0, result.stdout
    assert target.read_text() == '{"a": 1}\n'


def test_list_models_prints_table():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "id": 7,
                        "displayName": "my-sft",
                        "version": 3,
                        "versionName": "v3",
                        "latest": True,
                    }
                ]
            },
        )

    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(_app_with(list_models_), [])

    assert result.exit_code == 0
    assert "my-sft" in result.stdout
    assert "v3" in result.stdout


def test_pull_model_downloads_directory(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=b"weights")
        if request.url.path.endswith("/models/m1:download"):
            return httpx.Response(
                200,
                json={
                    "files": [
                        {"name": "config.json", "url": "https://storage.test/c"}
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(
            _app_with(pull_model), ["m1", "--out", str(tmp_path / "m")]
        )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "m" / "config.json").read_bytes() == b"weights"


def test_list_judges_filters_to_judge_type():
    seen_params: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params).get("evaluatorType", ""))
        return httpx.Response(
            200,
            json={"evaluators": [{"id": 1, "displayName": "qual"}]},
        )

    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(_app_with(list_judges), [])

    assert result.exit_code == 0
    assert seen_params == ["judge"]


# ----------------------------------------------------------- operations ---


def test_operation_status_prints_details():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": 42,
                "status": "running",
                "done": False,
                "type": "model_training",
            },
        )

    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(_app_with(operation_status), ["42"])

    assert result.exit_code == 0
    assert "id=42" in result.stdout
    assert "status=running" in result.stdout
    assert "model_training" in result.stdout


def test_operation_stop_routes_to_post():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(
            200, json={"id": 42, "status": "cancelling", "done": False}
        )

    client = _client(handler)
    with patch("oumi.cli.platform._client", return_value=client):
        result = runner.invoke(_app_with(operation_stop), ["42"])

    assert result.exit_code == 0
    assert any("POST /v1/projects/proj-1/operations/42:stop" in s for s in seen)


def test_client_helper_exits_when_not_logged_in(monkeypatch):
    """`oumi platform datasets list` without creds should exit non-zero."""
    monkeypatch.setenv(
        "OUMI_CREDENTIALS_FILE", "/nonexistent/path/credentials.json"
    )

    result = runner.invoke(_app_with(list_datasets), [])

    assert result.exit_code != 0
    assert "Not logged in" in result.stdout
