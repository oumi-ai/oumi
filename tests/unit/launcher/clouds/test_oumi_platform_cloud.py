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

import httpx
import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher.base_cluster import JobState
from oumi.launcher.clouds.oumi_platform_cloud import (
    OumiPlatformCloud,
    _infer_job_kind,
)
from oumi.platform import Client, Credentials, PlatformError

_TEST_API_URL = "https://platform.test"
_TEST_CREDS = Credentials(
    api_url=_TEST_API_URL,
    api_key="k",
    project_id="proj-1",
)


def _make_client(handler) -> Client:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        base_url=_TEST_API_URL,
        transport=transport,
        headers={"Authorization": "Bearer k", "X-API-Key": "k"},
    )
    return Client(credentials=_TEST_CREDS, http_client=http)


def _make_job(run: str = "oumi train --config foo.yaml") -> JobConfig:
    return JobConfig(
        name="my-job",
        resources=JobResources(cloud="oumi-platform"),
        run=run,
    )


def test_infer_job_kind_from_run_command():
    assert _infer_job_kind(_make_job("oumi train --config x.yaml")) == "train"
    assert (
        _infer_job_kind(_make_job("oumi evaluate --config x.yaml")) == "evaluate"
    )
    assert _infer_job_kind(_make_job("oumi judge --config x.yaml")) == "judge"
    assert _infer_job_kind(_make_job("oumi synth --config x.yaml")) == "synth"
    assert _infer_job_kind(_make_job("oumi infer --config x.yaml")) == "infer"


def test_infer_job_kind_defaults_to_train():
    assert _infer_job_kind(_make_job("python something_else.py")) == "train"


def test_infer_job_kind_respects_envs_override():
    job = _make_job("python whatever.py")
    job.envs = {"OUMI_JOB_KIND": "evaluate"}
    assert _infer_job_kind(job) == "evaluate"


def test_up_cluster_submits_and_returns_status():
    submissions: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/jobs:submit")
        submissions.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "id": 42,
                "status": "pending",
                "done": False,
                "type": "model_training",
                "metadata": {"foo": "bar"},
            },
        )

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)

    status = cloud.up_cluster(_make_job(), name="display")

    assert status.id == "42"
    assert status.state == JobState.PENDING
    assert status.cluster == "42"
    assert not status.done
    assert len(submissions) == 1
    body = submissions[0]
    assert body["kind"] == "train"
    assert body["displayName"] == "display"
    assert body["config"]["run"].startswith("oumi train")


def test_get_cluster_returns_submitted_cluster():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": 7, "status": "running", "done": False}
        )

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)
    cloud.up_cluster(_make_job(), name=None)

    cluster = cloud.get_cluster("7")

    assert cluster is not None
    assert cluster.name() == "7"


def test_list_clusters_includes_submitted():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 1, "status": "pending"})

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)
    cloud.up_cluster(_make_job(), name=None)
    cloud.up_cluster(_make_job(), name=None)

    assert len(cloud.list_clusters()) == 1  # same id reused, so dedup


def test_cluster_get_job_polls_status():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": 9, "status": "pending"})
        return httpx.Response(
            200,
            json={"id": 9, "status": "running", "done": False},
        )

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)
    cloud.up_cluster(_make_job(), name=None)

    cluster = cloud.get_cluster("9")
    assert cluster is not None
    status = cluster.get_job("9")
    assert status.state == JobState.RUNNING


def test_cluster_cancel_routes_to_stop():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.method == "POST" and request.url.path.endswith("/jobs:submit"):
            return httpx.Response(200, json={"id": 11, "status": "pending"})
        return httpx.Response(
            200, json={"id": 11, "status": "cancelling", "done": False}
        )

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)
    cloud.up_cluster(_make_job(), name=None)

    cluster = cloud.get_cluster("11")
    assert cluster is not None
    cluster.cancel_job("11")

    assert any(":stop" in entry for entry in seen)


def test_up_cluster_when_submit_endpoint_missing_gives_actionable_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "no such route"})

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)

    with pytest.raises(PlatformError, match="jobs:submit"):
        cloud.up_cluster(_make_job(), name=None)


def test_terminal_status_marks_job_done():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": 5, "status": "completed", "done": True},
        )

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)

    status = cloud.up_cluster(_make_job(), name=None)
    assert status.state == JobState.SUCCEEDED
    assert status.done


def test_failed_status_maps_correctly():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": 6, "status": "failed", "done": True},
        )

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)

    status = cloud.up_cluster(_make_job(), name=None)
    assert status.state == JobState.FAILED
    assert status.done


def test_logs_stream_not_implemented():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 1, "status": "pending"})

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)
    cloud.up_cluster(_make_job(), name=None)
    cluster = cloud.get_cluster("1")

    assert cluster is not None
    with pytest.raises(NotImplementedError):
        cluster.get_logs_stream("anything")


def test_run_job_on_cluster_is_not_supported():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 1, "status": "pending"})

    client = _make_client(handler)
    cloud = OumiPlatformCloud(client=client)
    cloud.up_cluster(_make_job(), name=None)
    cluster = cloud.get_cluster("1")

    assert cluster is not None
    with pytest.raises(NotImplementedError, match="single platform operation"):
        cluster.run_job(_make_job())
