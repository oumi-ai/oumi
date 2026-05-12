# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :mod:`oumi.launcher.clouds.modal_cloud`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import JobState, JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clouds.modal_cloud import ModalCloud
from oumi.launcher.clusters.modal_cluster import ModalCluster


def _job() -> JobConfig:
    return JobConfig(
        name="myjob",
        user="user",
        working_dir="./",
        num_nodes=1,
        resources=JobResources(cloud="modal", accelerators="H100:8"),
        envs={},
        file_mounts={},
        storage_mounts={},
        setup="pip install -e .",
        run="./run.sh",
    )


def _status() -> JobStatus:
    return JobStatus(
        name="fc-1",
        id="fc-1",
        cluster="fc-1",
        status="pending",
        metadata="",
        done=False,
        state=JobState.PENDING,
    )


@pytest.fixture
def mock_modal_client():
    with patch("oumi.launcher.clouds.modal_cloud.ModalClient") as cls:
        instance = MagicMock(spec=ModalClient)
        instance.launch.return_value = _status()
        cls.return_value = instance
        yield instance


def test_up_cluster_registers_a_modal_cluster(mock_modal_client):
    cloud = ModalCloud()
    status = cloud.up_cluster(_job(), name=None)
    assert status.cluster == "fc-1"
    assert isinstance(cloud.get_cluster("fc-1"), ModalCluster)
    assert cloud.list_clusters() and cloud.list_clusters()[0].name() == "fc-1"


def test_get_cluster_creates_lazy_cluster_for_unknown_name(mock_modal_client):
    cloud = ModalCloud()
    cluster = cloud.get_cluster("fc-unseen")
    assert isinstance(cluster, ModalCluster)
    assert cluster.name() == "fc-unseen"


def test_modal_builder_is_registered():
    builder = REGISTRY.get("modal", RegistryType.CLOUD)
    assert builder is not None
    assert isinstance(builder(), ModalCloud)
