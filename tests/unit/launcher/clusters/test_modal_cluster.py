# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :mod:`oumi.launcher.clusters.modal_cluster`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import ClusterNotFoundError, JobState, JobStatus
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clusters.modal_cluster import ModalCluster


def _status(sandbox_id: str, state: JobState = JobState.RUNNING) -> JobStatus:
    return JobStatus(
        name=sandbox_id,
        id=sandbox_id,
        cluster=sandbox_id,
        status=state.value,
        metadata="",
        done=state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED),
        state=state,
    )


def test_cluster_name_and_equality():
    client = MagicMock(spec=ModalClient)
    a = ModalCluster("cluster-foo", client)
    b = ModalCluster("cluster-foo", client)
    c = ModalCluster("cluster-bar", client)
    assert a.name() == "cluster-foo"
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_get_job_looks_up_sandbox_by_id_ignoring_cluster_name():
    """Cluster name is logical; job_id is the sandbox object_id."""
    client = MagicMock(spec=ModalClient)
    client.get_status.return_value = _status("sb-deadbeef", JobState.RUNNING)
    cluster = ModalCluster("cluster-foo", client)
    job = cluster.get_job("sb-deadbeef")
    assert job is not None
    assert job.state == JobState.RUNNING
    client.get_status.assert_called_once_with("sb-deadbeef")


def test_get_job_returns_none_when_sandbox_not_found():
    client = MagicMock(spec=ModalClient)
    client.get_status.side_effect = ClusterNotFoundError("missing")
    cluster = ModalCluster("cluster-foo", client)
    assert cluster.get_job("sb-missing") is None


def test_get_jobs_returns_status_for_each_tracked_sandbox():
    client = MagicMock(spec=ModalClient)
    client.find_sandboxes_for_cluster.return_value = ["sb-1", "sb-2"]
    client.get_status.side_effect = [
        _status("sb-1", JobState.SUCCEEDED),
        _status("sb-2", JobState.RUNNING),
    ]
    cluster = ModalCluster("cluster-foo", client)
    jobs = cluster.get_jobs()
    assert [j.id for j in jobs] == ["sb-1", "sb-2"]
    client.find_sandboxes_for_cluster.assert_called_once_with("cluster-foo")


def test_get_jobs_skips_sandboxes_that_404():
    client = MagicMock(spec=ModalClient)
    client.find_sandboxes_for_cluster.return_value = ["sb-1", "sb-gone"]
    client.get_status.side_effect = [
        _status("sb-1", JobState.RUNNING),
        ClusterNotFoundError("gone"),
    ]
    cluster = ModalCluster("cluster-foo", client)
    jobs = cluster.get_jobs()
    assert [j.id for j in jobs] == ["sb-1"]


def test_cancel_job_cancels_by_sandbox_id():
    client = MagicMock(spec=ModalClient)
    client.get_status.return_value = _status("sb-1", JobState.CANCELLED)
    cluster = ModalCluster("cluster-foo", client)
    status = cluster.cancel_job("sb-1")
    client.cancel.assert_called_once_with("sb-1")
    assert status.state == JobState.CANCELLED


def test_run_job_unsupported():
    client = MagicMock(spec=ModalClient)
    cluster = ModalCluster("cluster-foo", client)
    job = JobConfig(
        name="x",
        user="u",
        working_dir="./",
        num_nodes=1,
        resources=JobResources(cloud="modal"),
        envs={},
        file_mounts={},
        storage_mounts={},
        run="echo hi",
    )
    with pytest.raises(NotImplementedError):
        cluster.run_job(job)


def test_stop_and_down_cancel_every_tracked_sandbox():
    client = MagicMock(spec=ModalClient)
    client.find_sandboxes_for_cluster.return_value = ["sb-1", "sb-2"]
    cluster = ModalCluster("cluster-foo", client)
    cluster.down()
    cancelled = [c.args[0] for c in client.cancel.call_args_list]
    assert cancelled == ["sb-1", "sb-2"]


def test_get_logs_stream_uses_job_id_when_provided():
    client = MagicMock(spec=ModalClient)
    cluster = ModalCluster("cluster-foo", client)
    cluster.get_logs_stream("cluster-foo", job_id="sb-1")
    client.get_logs_stream.assert_called_once_with("sb-1")


def test_get_logs_stream_falls_back_to_most_recent_sandbox():
    client = MagicMock(spec=ModalClient)
    client.find_sandboxes_for_cluster.return_value = ["sb-old", "sb-new"]
    cluster = ModalCluster("cluster-foo", client)
    cluster.get_logs_stream("cluster-foo", job_id=None)
    client.get_logs_stream.assert_called_once_with("sb-new")


def test_get_logs_stream_raises_when_no_tracked_sandbox_and_no_job_id():
    client = MagicMock(spec=ModalClient)
    client.find_sandboxes_for_cluster.return_value = []
    cluster = ModalCluster("cluster-foo", client)
    with pytest.raises(ClusterNotFoundError):
        cluster.get_logs_stream("cluster-foo", job_id=None)
