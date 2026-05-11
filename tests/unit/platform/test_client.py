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
from collections.abc import Callable

import httpx
import pytest

from oumi.platform import (
    Client,
    Credentials,
    PlatformAPIError,
    PlatformAuthError,
    PlatformError,
    PlatformOperationError,
)

_TEST_API_URL = "https://platform.test"
_TEST_CREDS = Credentials(
    api_url=_TEST_API_URL,
    api_key="test-key",
    project_id="proj-1",
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    credentials: Credentials = _TEST_CREDS,
) -> Client:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        base_url=credentials.api_url,
        transport=transport,
        headers={
            "Authorization": f"Bearer {credentials.api_key}",
            "X-API-Key": credentials.api_key,
        },
    )
    return Client(credentials=credentials, http_client=http)


class _Recorder:
    """Capture every request to verify URL, method, params, and body."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    def respond(self, *, status: int = 200, body: object = None):
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            if body is None:
                return httpx.Response(status)
            return httpx.Response(status, json=body)

        return handler


def test_get_dataset_sends_auth_headers():
    recorder = _Recorder()
    client = _make_client(recorder.respond(body={"id": 1, "displayName": "ds"}))

    result = client.datasets.get("42")

    assert result == {"id": 1, "displayName": "ds"}
    assert len(recorder.calls) == 1
    request = recorder.calls[0]
    assert request.method == "GET"
    assert str(request.url) == f"{_TEST_API_URL}/v1/projects/proj-1/datasets/42"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-API-Key"] == "test-key"


def test_list_datasets_passes_pagination():
    recorder = _Recorder()
    client = _make_client(recorder.respond(body={"datasets": []}))

    client.datasets.list(page_size=25, page_token="cursor")

    request = recorder.calls[0]
    assert request.url.params["pageSize"] == "25"
    assert request.url.params["pageToken"] == "cursor"


def test_list_datasets_drops_null_params():
    recorder = _Recorder()
    client = _make_client(recorder.respond(body={"datasets": []}))

    client.datasets.list(page_size=None, page_token=None)

    request = recorder.calls[0]
    assert "pageSize" not in request.url.params
    assert "pageToken" not in request.url.params


def test_per_call_project_id_overrides_default():
    recorder = _Recorder()
    client = _make_client(recorder.respond(body={"datasets": []}))

    client.datasets.list(project_id="proj-other")

    assert "/v1/projects/proj-other/datasets" in str(recorder.calls[0].url)


def test_missing_project_id_raises():
    creds = Credentials(api_url=_TEST_API_URL, api_key="k")
    recorder = _Recorder()
    client = _make_client(recorder.respond(), credentials=creds)

    with pytest.raises(PlatformError, match="project"):
        client.datasets.list()


def test_4xx_response_maps_to_api_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "not found"})

    client = _make_client(handler)

    with pytest.raises(PlatformAPIError) as exc_info:
        client.datasets.get("missing")
    assert exc_info.value.status_code == 404
    assert exc_info.value.response_body == {"message": "not found"}


def test_401_maps_to_auth_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad key"})

    client = _make_client(handler)

    with pytest.raises(PlatformAuthError):
        client.datasets.get("anything")


def test_403_also_maps_to_auth_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "forbidden"})

    client = _make_client(handler)

    with pytest.raises(PlatformAuthError):
        client.datasets.list()


def test_network_error_wrapped():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=_request)

    client = _make_client(handler)

    with pytest.raises(PlatformError, match="Network error"):
        client.datasets.list()


def test_models_list_versions_routes_correctly():
    recorder = _Recorder()
    client = _make_client(recorder.respond(body={"versions": []}))

    client.models.list_versions("m-1")

    assert (
        str(recorder.calls[0].url)
        == f"{_TEST_API_URL}/v1/projects/proj-1/models/m-1/versions"
    )


def test_evaluators_supported_models_routes_correctly():
    recorder = _Recorder()
    client = _make_client(recorder.respond(body={"models": []}))

    client.evaluators.supported_models()

    assert (
        str(recorder.calls[0].url)
        == f"{_TEST_API_URL}/v1/projects/proj-1/evaluators:supported_models"
    )


def test_operations_get_routes_correctly():
    recorder = _Recorder()
    client = _make_client(
        recorder.respond(
            body={"id": 7, "status": "running", "done": False, "type": "model_training"}
        )
    )

    op = client.operations.get(7)

    assert op["status"] == "running"
    assert (
        str(recorder.calls[0].url)
        == f"{_TEST_API_URL}/v1/projects/proj-1/operations/7"
    )


def test_operations_stop_uses_post():
    recorder = _Recorder()
    client = _make_client(
        recorder.respond(body={"id": 7, "status": "cancelling"})
    )

    client.operations.stop(7)

    request = recorder.calls[0]
    assert request.method == "POST"
    assert str(request.url).endswith("/operations/7:stop")


def test_operations_wait_polls_until_completed():
    statuses = iter(
        [
            {"id": 1, "status": "pending", "done": False},
            {"id": 1, "status": "running", "done": False},
            {"id": 1, "status": "completed", "done": True, "result": {"ok": 1}},
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(statuses))

    client = _make_client(handler)
    sleeps: list[float] = []

    op = client.operations.wait(
        1,
        poll_interval=1.0,
        max_poll_interval=4.0,
        sleep_fn=sleeps.append,
        time_fn=lambda: 0.0,
    )

    assert op["status"] == "completed"
    assert op["result"] == {"ok": 1}
    assert sleeps == [1.0, 2.0]  # exponential backoff, no sleep after terminal


def test_operations_wait_raises_on_failure():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": 5,
                "status": "failed",
                "done": True,
                "error": {"message": "training crashed"},
            },
        )

    client = _make_client(handler)

    with pytest.raises(PlatformOperationError) as exc_info:
        client.operations.wait(5, sleep_fn=lambda _: None)
    assert exc_info.value.status == "failed"
    assert exc_info.value.operation_id == 5
    assert "training crashed" in str(exc_info.value)


def test_operations_wait_returns_failure_when_raise_disabled():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": 5, "status": "failed", "done": True}
        )

    client = _make_client(handler)

    op = client.operations.wait(
        5,
        raise_on_failure=False,
        sleep_fn=lambda _: None,
    )

    assert op["status"] == "failed"


def test_operations_wait_times_out():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": 5, "status": "running", "done": False}
        )

    client = _make_client(handler)
    times = iter([0.0, 0.0, 100.0])

    with pytest.raises(TimeoutError):
        client.operations.wait(
            5,
            timeout=5.0,
            sleep_fn=lambda _: None,
            time_fn=lambda: next(times),
        )


def test_dataset_download_uses_presigned_url(tmp_path):
    """When the platform returns a {url: ...} payload, stream from that URL."""
    presigned = "https://storage.test/signed/ds-42.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "platform.test":
            return httpx.Response(200, json={"url": presigned})
        if str(request.url) == presigned:
            return httpx.Response(200, content=b'{"a":1}\n{"b":2}\n')
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)
    target = tmp_path / "ds.jsonl"

    result = client.datasets.download("42", target)

    assert result == target
    assert json.loads(target.read_text().splitlines()[0]) == {"a": 1}


def test_model_download_handles_multiple_files(tmp_path):
    """A multi-file payload should download each file under the destination dir."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "platform.test":
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "name": "config.json",
                            "url": "https://storage.test/config",
                        },
                        {
                            "name": "weights/model.safetensors",
                            "url": "https://storage.test/weights",
                        },
                    ]
                },
            )
        if request.url.host == "storage.test":
            return httpx.Response(200, content=request.url.path.encode())
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)
    dest = tmp_path / "model"

    result = client.models.download("m1", dest, version_id="v1")

    assert result == dest
    assert (dest / "config.json").read_bytes() == b"/config"
    assert (dest / "weights" / "model.safetensors").read_bytes() == b"/weights"


def test_model_download_raises_on_empty_url_list(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"files": []})

    client = _make_client(handler)

    with pytest.raises(PlatformError, match="did not include any URLs"):
        client.models.download("m1", tmp_path / "model")


def test_close_only_closes_owned_http_client():
    """If the caller passes their own httpx.Client, close() should not close it."""
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={})
    )
    http = httpx.Client(base_url=_TEST_API_URL, transport=transport)
    client = Client(credentials=_TEST_CREDS, http_client=http)

    client.close()

    # Caller's http client is still usable.
    response = http.get("/v1/projects/proj-1/datasets")
    assert response.status_code == 200
    http.close()


def test_mutually_exclusive_credentials_args():
    with pytest.raises(ValueError):
        Client(credentials=_TEST_CREDS, api_url="https://other")
