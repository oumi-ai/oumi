# pyright: reportTypedDictNotRequiredAccess=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalSubscript=false
"""Tests for job recovery, cancellation, cluster lifecycle, and cloud launch."""

import pytest

pytest.importorskip("fastmcp", reason="fastmcp is required for MCP server tests")

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from oumi.mcp import job_launcher, job_logs, job_service, server
from oumi.mcp.job_launcher import cancel
from oumi.mcp.job_registry import JobRecord, make_job_id
from oumi.mcp.job_runtime import JobRuntime


def _make_record(**overrides) -> JobRecord:
    defaults = dict(
        job_id="test-job",
        command="train",
        config_path="/tmp/train.yaml",
        cloud="gcp",
        cluster_name="",
        model_name="",
        submit_time="2026-02-12T17:09:25+00:00",
    )
    defaults.update(overrides)
    return JobRecord(**defaults)


@pytest.mark.asyncio
async def test_cancel_job_direct_cloud_identity():
    with (
        patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
        patch("oumi.mcp.job_service.launcher.cancel", return_value=None) as mock_cancel,
    ):
        response = await server.cancel_job(
            job_id="sky-job-123", cloud="gcp", cluster_name="cluster-a"
        )

    assert response["success"]
    mock_cancel.assert_called_once_with("sky-job-123", "gcp", "cluster-a")


@pytest.mark.asyncio
async def test_cancel_job_launcher_failure():
    with (
        patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
        patch(
            "oumi.mcp.job_service.launcher.cancel",
            side_effect=RuntimeError("cancel failed"),
        ),
    ):
        response = await server.cancel_job(
            job_id="sky-job-123", cloud="gcp", cluster_name="cluster-a"
        )

    assert not response["success"]
    assert "Failed to cancel cloud job" in response["error"]


@pytest.mark.asyncio
async def test_cancel_job_timeout():
    import asyncio as _asyncio

    async def _fake_wait_for(coro, timeout):  # noqa: ANN001, ANN202
        try:
            coro.close()
        except Exception:
            pass
        raise _asyncio.TimeoutError

    with (
        patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
        patch("oumi.mcp.job_service.asyncio.wait_for", side_effect=_fake_wait_for),
    ):
        response = await server.cancel_job(
            job_id="sky-job-789", cloud="gcp", cluster_name="cluster-c"
        )

    assert not response["success"]
    assert "timed out" in response.get("error", "")


@pytest.mark.asyncio
async def test_cancel_pending_cloud_launch_sets_flag_without_cancelling_task():
    record = _make_record()
    rt = JobRuntime()
    mock_task = asyncio.Future()
    rt.runner_task = mock_task  # type: ignore[assignment]
    response = await cancel(record, rt)
    assert response["success"]
    assert rt.cancel_requested
    assert not mock_task.cancelled()


@pytest.mark.asyncio
async def test_cancel_launched_cloud_job_calls_launcher_cancel():
    record = _make_record(job_id="sky-99")
    rt = JobRuntime()
    with patch("oumi.mcp.job_service.launcher") as mock_launcher:
        mock_launcher.cancel.return_value = None
        response = await cancel(record, rt)
    assert response["success"]
    mock_launcher.cancel.assert_called_once_with(
        "sky-99", record.cloud, record.cluster_name
    )


@pytest.mark.asyncio
async def test_get_job_status_not_found_graceful():
    with (
        patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
        patch("oumi.mcp.job_service._fetch_cloud_status_direct", return_value=None),
    ):
        response = await server.get_job_status(
            job_id="sky-job-123", cloud="gcp", cluster_name="cluster-a"
        )

    assert not response["success"]
    assert response["status"] == "not_found"


@pytest.mark.asyncio
async def test_get_job_logs_direct_identity():
    mock_logs = ("line1\nline2\nline3", 3)
    with (
        patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
        patch(
            "oumi.mcp.job_service._get_cloud_logs",
            new_callable=AsyncMock,
            return_value=mock_logs,
        ),
    ):
        response = await server.get_job_logs(
            job_id="sky-job-123", cloud="gcp", cluster_name="cluster-a", lines=50
        )

    assert response["success"]
    assert response["lines_returned"] == 3
    assert "line1" in response["logs"]


@pytest.mark.asyncio
async def test_get_job_logs_requires_cluster_name():
    with patch("oumi.mcp.job_service._resolve_job_record", return_value=None):
        response = await server.get_job_logs(
            job_id="sky-job-123", cloud="gcp", cluster_name="", lines=50
        )

    assert not response["success"]

    if response["error"]:
        assert "cluster_name is required" in response["error"]


@pytest.mark.asyncio
async def test_run_oumi_job_blocks_malformed_yaml():
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_cfg = Path(tmp_dir) / "bad.yaml"
        bad_cfg.write_text("model: [", encoding="utf-8")
        response = await server.run_oumi_job(
            config_path=str(bad_cfg),
            command="train",
            client_cwd=tmp_dir,
            dry_run=False,
        )
    assert not response["success"]
    assert "Invalid YAML config" in response["error"]


@pytest.mark.asyncio
async def test_dry_run_cloud_rejects_training_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = Path(tmp_dir) / "train.yaml"
        cfg.write_text("model: {model_name: test/model}\n", encoding="utf-8")
        response = await server.run_oumi_job(
            config_path=str(cfg),
            client_cwd=tmp_dir,
            command="train",
            cloud="gcp",
            dry_run=True,
        )
    assert not response["success"]
    assert "job config" in response["error"].lower()
    assert "guidance://cloud-launch" in response["error"]


@pytest.mark.asyncio
async def test_dry_run_cloud_shows_jobconfig_yaml_preview():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = Path(tmp_dir) / "job.yaml"
        cfg.write_text(
            "name: test-job\n"
            "resources:\n  cloud: gcp\n  accelerators: 'A100:4'\n"
            "working_dir: .\n"
            "setup: |\n  pip install oumi[gpu]\n"
            "run: |\n  oumi train -c ./config.yaml\n",
            encoding="utf-8",
        )
        response = await server.run_oumi_job(
            config_path=str(cfg),
            client_cwd=tmp_dir,
            command="train",
            cloud="gcp",
            dry_run=True,
        )
    assert response["success"]
    assert "oumi launch up" in response["message"]
    assert "Generated JobConfig" in response["message"]


@pytest.mark.asyncio
async def test_cloud_launch_rejects_training_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "train.yaml"
        cfg_path.write_text("model: {model_name: test/model}\n", encoding="utf-8")
        record = _make_record(
            job_id="train_20260220_000001_abc123", config_path=str(cfg_path)
        )
        rt = JobRuntime()
        rt.run_dir = Path(tmp_dir) / "run"
        with patch("oumi.mcp.job_service.get_registry") as mock_reg:
            mock_reg.return_value.update = lambda *a, **kw: None
            await job_launcher._launch_cloud(record, rt, client_cwd=tmp_dir)
        assert rt.error_message is not None
        assert "job config" in rt.error_message.lower()


@pytest.mark.asyncio
async def test_cloud_launch_reconciles_pending_cancel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "job.yaml"
        cfg_path.write_text(
            "name: test-job\n"
            "resources:\n  cloud: gcp\n  accelerators: 'A100:1'\n"
            "working_dir: .\n"
            "setup: |\n  echo setup\n"
            "run: |\n  echo run\n",
            encoding="utf-8",
        )
        record = _make_record(
            job_id="train_20260220_000002_def456", config_path=str(cfg_path)
        )
        rt = JobRuntime()
        rt.cancel_requested = True
        rt.run_dir = Path(tmp_dir) / "run"

        status = SimpleNamespace(
            id="cloud-456",
            cluster="cluster-b",
            done=False,
            status="RUNNING",
            state=SimpleNamespace(name="RUNNING"),
            metadata={},
        )
        cancelled = SimpleNamespace(
            id="cloud-456",
            cluster="cluster-b",
            done=True,
            status="CANCELLED",
            state=SimpleNamespace(name="CANCELLED"),
            metadata={},
        )

        def _mock_update(job_id, **fields):  # noqa: ANN001, ANN003
            for k, v in fields.items():
                setattr(record, k, v)

        mock_reg = SimpleNamespace(
            update=_mock_update,
            get=lambda jid: record,
            remove=lambda jid: None,
            add=lambda rec: None,
        )
        with (
            patch(
                "oumi.mcp.job_service.launcher.up",
                return_value=(SimpleNamespace(), status),
            ),
            patch(
                "oumi.mcp.job_service.launcher.cancel", return_value=cancelled
            ) as mock_cancel,
            patch("oumi.mcp.job_service.get_registry", return_value=mock_reg),
        ):
            await job_launcher._launch_cloud(record, rt, client_cwd=tmp_dir)  # type: ignore[attr-defined]

        mock_cancel.assert_called_once_with("cloud-456", "gcp", "cluster-b")
        assert rt.cancel_requested


@pytest.mark.asyncio
async def test_launch_cloud_client_cwd_sets_working_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        client_dir = Path(tmp_dir) / "project"
        client_dir.mkdir()
        cfg_path = client_dir / "job.yaml"
        cfg_path.write_text(
            "name: test-job\n"
            "resources:\n  cloud: gcp\n  accelerators: 'A100:1'\n"
            "working_dir: .\n"
            "setup: |\n  echo setup\n"
            "run: |\n  echo run\n",
            encoding="utf-8",
        )
        record = _make_record(
            job_id="train_20260220_000003_gh789", config_path=str(cfg_path)
        )
        rt = JobRuntime()
        rt.run_dir = Path(tmp_dir) / "run"

        captured_wd = {}

        def _fake_up(job_cfg, cluster_name):  # noqa: ANN001
            captured_wd["working_dir"] = job_cfg.working_dir
            status = SimpleNamespace(
                id="cloud-789",
                cluster="cluster-x",
                done=False,
                status="RUNNING",
                state=SimpleNamespace(name="RUNNING"),
                metadata={},
            )
            return (SimpleNamespace(), status)

        with patch("oumi.mcp.job_service.launcher.up", side_effect=_fake_up):
            with patch("oumi.mcp.job_service.get_registry") as mock_registry:
                mock_registry.return_value.update = lambda *a, **kw: None
                mock_registry.return_value.get = lambda jid: record
                await job_launcher._launch_cloud(  # type: ignore[attr-defined]
                    record, rt, client_cwd=str(client_dir)
                )

        expected = os.path.realpath(str(client_dir))  # noqa: ASYNC240
        assert captured_wd["working_dir"] == expected


def test_read_log_tail_large_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "big.log"
        lines = [f"line-{idx}" for idx in range(1, 20001)]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tail, count = job_logs.read_log_tail(log_path, 5)
    assert count == 5
    assert tail.splitlines() == lines[-5:]


def test_make_job_id_sanitizes_path_traversal():
    assert "/" not in make_job_id("train", job_name="../../etc/evil")
    assert "\\" not in make_job_id("train", job_name="..\\..\\evil")
    assert ".." not in make_job_id("train", job_name="../up")


def test_make_job_id_rejects_empty_after_sanitization():
    with pytest.raises(ValueError):
        make_job_id("train", job_name="../../..")


@pytest.mark.asyncio
async def test_stop_cluster_calls_launcher():
    with patch("oumi.mcp.job_service.launcher.stop") as mock_stop:
        response = await server.stop_cluster(cloud="gcp", cluster_name="sky-xxxx")
    assert response["success"]
    mock_stop.assert_called_once_with("gcp", "sky-xxxx")


@pytest.mark.asyncio
async def test_stop_cluster_error():
    with patch(
        "oumi.mcp.job_service.launcher.stop",
        side_effect=RuntimeError("network error"),
    ):
        response = await server.stop_cluster(cloud="gcp", cluster_name="sky-xxxx")
    assert not response["success"]
    assert "Failed to stop cluster" in response.get("error", "")


@pytest.mark.asyncio
async def test_stop_cluster_rejects_empty_args():
    response = await server.stop_cluster(cloud="", cluster_name="sky-xxxx")
    assert not response["success"]
    assert "required" in response.get("error", "")


@pytest.mark.asyncio
async def test_down_cluster_without_confirm():
    with patch("oumi.mcp.job_service.launcher.down") as mock_down:
        response = await server.down_cluster(cloud="gcp", cluster_name="sky-xxxx")
    mock_down.assert_not_called()
    assert response["success"]
    assert "IRREVERSIBLE" in response.get("message", "")


@pytest.mark.asyncio
async def test_down_cluster_with_confirm():
    with patch("oumi.mcp.job_service.launcher.down") as mock_down:
        response = await server.down_cluster(
            cloud="gcp",
            cluster_name="sky-xxxx",
            confirm=True,
            user_confirmation="DOWN",
        )
    assert response["success"]
    mock_down.assert_called_once_with("gcp", "sky-xxxx")


@pytest.mark.asyncio
async def test_down_cluster_wrong_phrase():
    with patch("oumi.mcp.job_service.launcher.down") as mock_down:
        response = await server.down_cluster(
            cloud="gcp",
            cluster_name="sky-xxxx",
            confirm=True,
            user_confirmation="EXECUTE",
        )
    mock_down.assert_not_called()
    assert not response["success"]


@pytest.mark.asyncio
async def test_down_cluster_error():
    with patch(
        "oumi.mcp.job_service.launcher.down",
        side_effect=RuntimeError("cloud error"),
    ):
        response = await server.down_cluster(
            cloud="gcp",
            cluster_name="sky-xxxx",
            confirm=True,
            user_confirmation="DOWN",
        )
    assert not response["success"]
    assert "Failed to delete cluster" in response.get("error", "")


def test_get_started_mentions_tools():
    result = server.get_started()
    assert "stop_cluster" in result
    assert "down_cluster" in result
    assert "Cloud Job Workflow" in result
    assert "Cluster Lifecycle" in result
    assert "suggested_configs" in result


@pytest.mark.asyncio
async def test_list_jobs_calls_launcher_status():
    job_status = SimpleNamespace(
        id="sky-job-001",
        cluster="cluster-y",
        done=False,
        status="RUNNING",
        state=SimpleNamespace(name="RUNNING"),
        metadata={},
    )

    with (
        patch(
            "oumi.mcp.job_service.launcher.status", return_value={"gcp": [job_status]}
        ),
        patch("oumi.mcp.job_service.get_registry") as mock_reg,
    ):
        mock_reg.return_value.find_by_cloud.return_value = None
        mock_reg.return_value.all.return_value = []
        summaries = await job_service._list_job_summaries()

    assert len(summaries) == 1
    assert summaries[0]["cloud"] == "gcp"
    assert summaries[0]["status"] == "RUNNING"


@pytest.mark.asyncio
async def test_list_jobs_enriches_with_mcp_job_id():
    job_status = SimpleNamespace(
        id="sky-job-002",
        cluster="cluster-z",
        done=True,
        status="SUCCEEDED",
        state=SimpleNamespace(name="SUCCEEDED"),
        metadata={},
    )
    mcp_record = _make_record(
        job_id="sky-job-002",
        cloud="aws",
        cluster_name="cluster-z",
        model_name="meta-llama/Llama-3.1-8B",
    )

    with (
        patch(
            "oumi.mcp.job_service.launcher.status", return_value={"aws": [job_status]}
        ),
        patch("oumi.mcp.job_service.get_registry") as mock_reg,
    ):
        mock_reg.return_value.find_by_cloud.return_value = mcp_record
        mock_reg.return_value.all.return_value = []
        summaries = await job_service._list_job_summaries()

    assert len(summaries) == 1
    assert summaries[0]["job_id"] == "sky-job-002"
    assert summaries[0]["model_name"] == "meta-llama/Llama-3.1-8B"


def _write_job_yaml(directory: str) -> Path:
    cfg = Path(directory) / "job.yaml"
    cfg.write_text(
        "name: test-job\n"
        "resources:\n  cloud: gcp\n  accelerators: 'A100:1'\n"
        "working_dir: .\nsetup: |\n  echo setup\nrun: |\n  echo run\n",
        encoding="utf-8",
    )
    return cfg


def _patch_cloud_submission():
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        rt = SimpleNamespace(
            log_dir=None,
            run_dir=None,
            runner_task=None,
            error_message=None,
            cancel_requested=False,
        )
        with (
            patch("oumi.mcp.server.launch_job", new_callable=AsyncMock),
            patch("oumi.mcp.server.get_registry") as mock_reg,
            patch("oumi.mcp.server.get_runtime", return_value=rt),
        ):
            mock_reg.return_value.add = lambda rec: None
            yield

    return _ctx()


@pytest.mark.asyncio
async def test_skip_preflight_bypasses_checks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _write_job_yaml(tmp_dir)
        with (
            patch("oumi.mcp.server._pre_flight_check") as mock_pf,
            _patch_cloud_submission(),
        ):
            resp = await server.run_oumi_job(
                str(cfg),
                "train",
                tmp_dir,
                cloud="gcp",
                dry_run=False,
                skip_preflight=True,
            )
        mock_pf.assert_not_called()
        assert resp["success"]
        assert "preflight" not in resp


@pytest.mark.asyncio
async def test_skip_preflight_false_runs_checks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _write_job_yaml(tmp_dir)
        pf_result = {
            "blocking": False,
            "summary": "Ready",
            "errors": [],
            "warnings": [],
            "skypilot_compat_issue": False,
        }
        with (
            patch(
                "oumi.mcp.server._pre_flight_check", return_value=pf_result
            ) as mock_pf,
            _patch_cloud_submission(),
        ):
            resp = await server.run_oumi_job(
                str(cfg),
                "train",
                tmp_dir,
                cloud="gcp",
                dry_run=False,
                skip_preflight=False,
            )
        mock_pf.assert_called_once()
        assert "preflight" in resp


@pytest.mark.asyncio
async def test_skypilot_compat_issue_blocks_launch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _write_job_yaml(tmp_dir)
        pf_result = {
            "blocking": False,
            "summary": "warnings",
            "errors": [],
            "warnings": [],
            "skypilot_compat_issue": True,
        }
        with patch("oumi.mcp.server._pre_flight_check", return_value=pf_result):
            resp = await server.run_oumi_job(
                str(cfg),
                "train",
                tmp_dir,
                cloud="gcp",
                dry_run=False,
            )
        assert not resp["success"]
        assert resp["status"] == "blocked"
        assert resp["preflight"]["blocking"]
