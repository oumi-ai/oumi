# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :mod:`oumi.launcher.clusters.modal_cluster`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import JobState, JobStatus
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clusters.modal_cluster import ModalCluster


def _status(state: JobState = JobState.RUNNING) -> JobStatus:
    return JobStatus(
        name="fc-1",
        id="fc-1",
        cluster="fc-1",
        status=state.value,
        metadata="",
        done=state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED),
        state=state,
    )


def test_cluster_name_and_equality():
    client = MagicMock(spec=ModalClient)
    a = ModalCluster("fc-1", client)
    b = ModalCluster("fc-1", client)
    c = ModalCluster("fc-2", client)
    assert a.name() == "fc-1"
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_get_job_returns_none_for_mismatched_id():
    client = MagicMock(spec=ModalClient)
    cluster = ModalCluster("fc-1", client)
    assert cluster.get_job("not-this-job") is None
    client.get_status.assert_not_called()


def test_get_job_delegates_to_client_when_id_matches():
    client = MagicMock(spec=ModalClient)
    client.get_status.return_value = _status(JobState.RUNNING)
    cluster = ModalCluster("fc-1", client)
    job = cluster.get_job("fc-1")
    assert job is not None
    assert job.state == JobState.RUNNING
    client.get_status.assert_called_once_with("fc-1")


def test_get_jobs_returns_single_status():
    client = MagicMock(spec=ModalClient)
    client.get_status.return_value = _status(JobState.SUCCEEDED)
    cluster = ModalCluster("fc-1", client)
    jobs = cluster.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].state == JobState.SUCCEEDED


def test_cancel_job_calls_client_and_returns_status():
    client = MagicMock(spec=ModalClient)
    client.get_status.return_value = _status(JobState.CANCELLED)
    cluster = ModalCluster("fc-1", client)
    status = cluster.cancel_job("fc-1")
    client.cancel.assert_called_once_with("fc-1")
    assert status.state == JobState.CANCELLED


def test_cancel_job_raises_for_mismatched_id():
    client = MagicMock(spec=ModalClient)
    cluster = ModalCluster("fc-1", client)
    with pytest.raises(RuntimeError):
        cluster.cancel_job("other")


def test_run_job_unsupported():
    client = MagicMock(spec=ModalClient)
    cluster = ModalCluster("fc-1", client)
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


def test_stop_and_down_cancel():
    client = MagicMock(spec=ModalClient)
    cluster = ModalCluster("fc-1", client)
    cluster.stop()
    cluster.down()
    assert client.cancel.call_count == 2
