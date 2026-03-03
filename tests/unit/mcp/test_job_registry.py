# pyright: reportOptionalMemberAccess=false, reportCallIssue=false
"""Tests for oumi.mcp.job_service — JobRecord persistence, update, prune."""

import json
from pathlib import Path

from oumi.mcp.job_service import JobRecord, JobRegistry


def _make_record(**overrides) -> JobRecord:
    defaults = dict(
        job_id="j1",
        command="train",
        config_path="/tmp/t.yaml",
        cloud="local",
        cluster_name="",
        model_name="test",
        submit_time="2099-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return JobRecord(**defaults)


def test_persists_to_disk(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.add(_make_record(cloud="gcp", cluster_name="c1"))

    reg2 = JobRegistry(path)
    loaded = reg2.get("j1")
    assert loaded is not None
    assert loaded.cloud == "gcp"
    assert loaded.job_id == "j1"


def test_update(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.add(_make_record())
    reg.update("j1", cluster_name="updated-cluster")
    assert reg.get("j1").cluster_name == "updated-cluster"


def test_update_persists(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.add(_make_record(cloud="gcp"))
    reg.update("j1", cluster_name="cl-1")

    reg2 = JobRegistry(path)
    loaded = reg2.get("j1")
    assert loaded is not None
    assert loaded.cluster_name == "cl-1"


def test_update_missing_noop(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.update("nonexistent", cluster_name="cl-99")
    assert reg.get("nonexistent") is None


def test_remove(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.add(_make_record())
    reg.remove("j1")
    assert reg.get("j1") is None

    reg2 = JobRegistry(path)
    assert reg2.get("j1") is None


def test_find_by_cloud(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.add(_make_record(job_id="sky-99", cloud="gcp", cluster_name="c1"))
    assert reg.find_by_cloud("gcp", "sky-99").job_id == "sky-99"
    assert reg.find_by_cloud("aws", "sky-99") is None


def test_all(tmp_path: Path):
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    for i in range(3):
        reg.add(_make_record(job_id=f"j{i}"))
    assert len(reg.all()) == 3


def test_load_corrupt_file_starts_fresh(tmp_path: Path):
    path = tmp_path / "jobs.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    reg = JobRegistry(path)
    assert len(reg.all()) == 0


def test_prune_old(tmp_path: Path):
    """Records older than 7 days are pruned on load."""
    path = tmp_path / "jobs.json"
    reg = JobRegistry(path)
    reg.add(
        _make_record(job_id="old", cloud="gcp", submit_time="2020-01-01T00:00:00+00:00")
    )
    reg.add(
        _make_record(job_id="new", cloud="gcp", submit_time="2099-01-01T00:00:00+00:00")
    )

    reg2 = JobRegistry(path)
    assert reg2.get("old") is None
    assert reg2.get("new") is not None


def test_legacy_records_with_status(tmp_path: Path):
    """Legacy JSON records containing a 'status' field load without error."""
    path = tmp_path / "jobs.json"
    legacy = [
        {
            "job_id": "legacy-1",
            "command": "train",
            "config_path": "/tmp/t.yaml",
            "cloud": "gcp",
            "cluster_name": "cl",
            "model_name": "m",
            "submit_time": "2099-01-01T00:00:00+00:00",
            "status": "RUNNING",
        }
    ]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    reg = JobRegistry(path)
    loaded = reg.get("legacy-1")
    assert loaded is not None
    assert loaded.job_id == "legacy-1"
    assert not hasattr(loaded, "status")
