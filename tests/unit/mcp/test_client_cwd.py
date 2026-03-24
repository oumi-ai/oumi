# pyright: reportArgumentType=false, reportCallIssue=false
"""Tests for client_cwd path resolution."""

import pytest

pytest.importorskip("fastmcp", reason="fastmcp is required for MCP server tests")

from pathlib import Path
from unittest.mock import MagicMock, patch

from oumi.mcp.config_service import resolve_config_path, resolve_path
from oumi.mcp.job_launcher import start_local_job
from oumi.mcp.job_registry import JobRecord
from oumi.mcp.job_runtime import JobRuntime
from oumi.mcp.server import (
    pre_flight_check,
    run_oumi_job,
    validate_config,
)


def test_resolve_path_absolute_unchanged():
    result = resolve_path("/abs/path/config.yaml", Path("/some/cwd"))
    assert result == Path("/abs/path/config.yaml")


def test_resolve_path_relative(tmp_path: Path):
    subdir = tmp_path / "configs"
    subdir.mkdir()
    config = subdir / "train.yaml"
    config.write_text("")
    result = resolve_path("configs/train.yaml", Path(tmp_path))
    assert result == config.resolve()


def test_resolve_path_tilde():
    result = resolve_path("~/configs/train.yaml", Path("/some/cwd"))
    assert result.is_absolute()
    assert "configs/train.yaml" in str(result)


def test_resolve_config_path_relative(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text("model:\n  name: test\n")
    resolved, err = resolve_config_path("train.yaml", str(tmp_path))
    assert err is None
    assert resolved == config.resolve()


def test_resolve_config_path_absolute(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text("model:\n  name: test\n")
    resolved, err = resolve_config_path(str(config), str(tmp_path))
    assert err is None
    assert resolved == config.resolve()


def test_resolve_config_path_relative_cwd_rejected():
    _, err = resolve_config_path("train.yaml", "relative/cwd")
    assert err is not None
    assert "absolute" in err


def test_resolve_config_path_nonexistent(tmp_path: Path):
    _, err = resolve_config_path("nonexistent.yaml", str(tmp_path))
    assert err is not None
    assert "not found" in err.lower()


def test_validate_config_resolves_relative_path(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text("training:\n  data:\n    train:\n      dataset_name: test\n")
    mock_cfg = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_yaml.return_value = mock_cfg
    with patch("oumi.mcp.server.TASK_MAPPING", {"training": mock_cls}):
        result = validate_config("train.yaml", "training", client_cwd=str(tmp_path))
        assert result["valid"]


def test_pre_flight_resolves_relative_config(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text("model:\n  model_name: gpt2\n")
    result = pre_flight_check("train.yaml", client_cwd=str(tmp_path))
    blocking_errors = result.get("errors", [])
    path_errors = [e for e in blocking_errors if "absolute" in e.lower()]
    assert path_errors == []


@pytest.mark.asyncio
async def test_dry_run_resolves_relative_config(tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text("model:\n  model_name: gpt2\ntraining:\n  output_dir: ./output\n")
    result = await run_oumi_job(
        config_path="train.yaml",
        command="train",
        client_cwd=str(tmp_path),
        dry_run=True,
    )
    assert result.get("success", False), result.get("error", "")
    assert result["dry_run"] is True


def test_start_local_job_sets_cwd(tmp_path: Path):
    log_dir = tmp_path / "logs"
    record = JobRecord(
        job_id="test-job",
        command="train",
        config_path="/tmp/train.yaml",
        cloud="local",
        cluster_name="",
        model_name="gpt2",
        submit_time="2026-01-01T00:00:00Z",
    )
    rt = JobRuntime()
    rt.log_dir = log_dir

    mock_proc = MagicMock()
    mock_proc.pid = 12345

    with (
        patch(
            "oumi.mcp.job_service.subprocess.Popen", return_value=mock_proc
        ) as mock_popen,
        patch("oumi.mcp.job_service.get_registry") as mock_reg,
    ):
        mock_reg.return_value.update = MagicMock()
        start_local_job(record, rt, client_cwd="/home/alice/project")

    assert mock_popen.call_args.kwargs.get("cwd") == "/home/alice/project"
