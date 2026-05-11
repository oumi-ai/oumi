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
from dataclasses import dataclass

import httpx
import pytest

from oumi.cli._remote import submit_remote_run
from oumi.platform import Client, Credentials, PlatformError

_TEST_API_URL = "https://platform.test"
_TEST_CREDS = Credentials(
    api_url=_TEST_API_URL, api_key="k", project_id="proj-1"
)


@dataclass
class _FakeConfig:
    model_name: str = "abc"
    epochs: int = 3


def _make_client(handler) -> Client:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        base_url=_TEST_API_URL,
        transport=transport,
        headers={"Authorization": "Bearer k", "X-API-Key": "k"},
    )
    return Client(credentials=_TEST_CREDS, http_client=http)


def test_submit_remote_run_posts_config_and_returns_immediately_when_detached():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs:submit"):
            seen.append(json.loads(request.content.decode()))
            return httpx.Response(
                200, json={"id": 5, "status": "pending", "done": False}
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    op = submit_remote_run(
        _FakeConfig(),
        kind="train",
        wait=False,
        client=client,
    )

    assert op["id"] == 5
    assert seen[0]["kind"] == "train"
    assert seen[0]["config"] == {"model_name": "abc", "epochs": 3}


def test_submit_remote_run_waits_when_wait_is_true():
    statuses = iter(
        [
            {"id": 7, "status": "pending", "done": False},
            {"id": 7, "status": "running", "done": False},
            {"id": 7, "status": "completed", "done": True},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs:submit"):
            return httpx.Response(200, json=next(statuses))
        return httpx.Response(200, json=next(statuses))

    client = _make_client(handler)

    op = submit_remote_run(
        _FakeConfig(),
        kind="evaluate",
        wait=True,
        client=client,
    )
    # With sleep_fn defaulted to time.sleep but only two polls of <2s,
    # this should resolve quickly enough on CI; we don't override here on
    # purpose so the timing path is exercised once. If it ever becomes flaky,
    # patch oumi.platform.client.time.sleep.
    assert op["status"] == "completed"


def test_submit_includes_display_name_when_provided():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"id": 1, "status": "pending"})

    client = _make_client(handler)

    submit_remote_run(
        _FakeConfig(),
        kind="train",
        name="my-experiment",
        wait=False,
        client=client,
    )

    assert seen[0]["displayName"] == "my-experiment"


def test_submit_uses_explicit_project():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"id": 2, "status": "pending"})

    client = _make_client(handler)

    submit_remote_run(
        _FakeConfig(),
        kind="train",
        project_id="custom-proj",
        wait=False,
        client=client,
    )

    assert seen_paths[0] == "/v1/projects/custom-proj/jobs:submit"


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        submit_remote_run(_FakeConfig(), kind="bogus", client=None)


def test_404_gives_helpful_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "missing"})

    client = _make_client(handler)

    with pytest.raises(PlatformError, match="jobs:submit"):
        submit_remote_run(
            _FakeConfig(),
            kind="train",
            wait=False,
            client=client,
        )


def test_non_dataclass_non_dict_raises():
    with pytest.raises(TypeError, match="dataclass"):
        submit_remote_run("not a config", kind="train", client=None)


def test_dict_config_passes_through():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"id": 1, "status": "pending"})

    client = _make_client(handler)

    submit_remote_run(
        {"raw": "config"},
        kind="train",
        wait=False,
        client=client,
    )

    assert seen[0]["config"] == {"raw": "config"}
