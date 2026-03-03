import asyncio
import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from oumi.mcp import job_service, server
from oumi.mcp.job_service import (
    JobRecord,
    JobRegistry,
    JobRuntime,
    cancel,
    make_job_id,
)


class JobRecoveryAndControlTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_persists_and_rehydrates(self) -> None:
        from datetime import datetime, timezone

        recent = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "jobs.json"
            reg1 = JobRegistry(path=state_file)
            record = JobRecord(
                job_id="sky-job-123",
                command="train",
                config_path="/tmp/train.yaml",
                cloud="gcp",
                cluster_name="cluster-a",
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                submit_time=recent,
            )
            reg1.add(record)

            reg2 = JobRegistry(path=state_file)
            loaded = reg2.get(record.job_id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.cloud, "gcp")
        self.assertEqual(loaded.cluster_name, "cluster-a")
        self.assertEqual(loaded.job_id, "sky-job-123")

    def test_registry_prunes_old_terminal_jobs(self) -> None:
        """Terminal jobs older than 7 days are evicted on load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "jobs.json"
            reg = JobRegistry(path=state_file)
            reg.add(
                JobRecord(
                    job_id="old_job",
                    command="train",
                    config_path="/tmp/t.yaml",
                    cloud="gcp",
                    cluster_name="c",
                    model_name="m",
                    submit_time="2020-01-01T00:00:00+00:00",
                )
            )
            reg.add(
                JobRecord(
                    job_id="recent_job",
                    command="train",
                    config_path="/tmp/t.yaml",
                    cloud="gcp",
                    cluster_name="c",
                    model_name="m",
                    submit_time=datetime.now(timezone.utc).isoformat(),
                )
            )

            # Reload — should prune old_job but keep recent_job
            reg2 = JobRegistry(path=state_file)
            self.assertIsNone(reg2.get("old_job"))
            self.assertIsNotNone(reg2.get("recent_job"))

    async def test_cancel_job_supports_direct_cloud_identity(self) -> None:
        with (
            patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
            patch(
                "oumi.mcp.job_service.launcher.cancel", return_value=None
            ) as mock_cancel,
        ):
            response = await server.cancel_job(
                job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertTrue(response["success"])
        mock_cancel.assert_called_once_with("sky-job-123", "gcp", "cluster-a")

    async def test_cancel_job_returns_structured_error_on_launcher_failure(
        self,
    ) -> None:
        with (
            patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
            patch(
                "oumi.mcp.job_service.launcher.cancel",
                side_effect=RuntimeError("cancel failed"),
            ),
        ):
            response = await server.cancel_job(
                job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertFalse(response["success"])
        self.assertIn("Failed to cancel cloud job", response["error"])

    async def test_get_job_status_by_direct_identity_not_found_is_graceful(
        self,
    ) -> None:
        with (
            patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
            patch("oumi.mcp.job_service._fetch_cloud_status_direct", return_value=None),
        ):
            response = await server.get_job_status(
                job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertFalse(response["success"])
        self.assertEqual(response["status"], "not_found")

    async def test_get_job_logs_by_direct_identity_tries_cloud_retrieval(
        self,
    ) -> None:
        """When job is not in registry, get_job_logs should attempt direct cloud
        log retrieval via an ephemeral JobRecord instead of giving up."""
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
                job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
                lines=50,
            )

        self.assertTrue(response["success"])
        self.assertEqual(response["lines_returned"], 3)
        self.assertIn("line1", response["logs"])

    async def test_get_job_logs_by_direct_identity_without_cluster_name(
        self,
    ) -> None:
        """Direct cloud log retrieval requires cluster_name."""
        with patch("oumi.mcp.job_service._resolve_job_record", return_value=None):
            response = await server.get_job_logs(
                job_id="sky-job-123",
                cloud="gcp",
                cluster_name="",
                lines=50,
            )

        self.assertFalse(response["success"])
        self.assertIn("cluster_name is required", response["error"])

    def test_logging_configuration_downgrades_noisy_mcp_loggers(self) -> None:
        server._configure_logging()
        self.assertEqual(
            logging.getLogger("mcp.server.lowlevel.server").level,
            logging.WARNING,
        )

    async def test_run_oumi_job_blocks_malformed_yaml_at_execution_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_cfg = Path(tmp_dir) / "bad.yaml"
            bad_cfg.write_text("model: [", encoding="utf-8")
            response = await server.run_oumi_job(
                config_path=str(bad_cfg),
                command="train",
                client_cwd=tmp_dir,
                dry_run=False,
                confirm=True,
                user_confirmation="EXECUTE",
            )
        self.assertFalse(response["success"])
        self.assertIn("Invalid YAML config", response["error"])

    async def test_dry_run_cloud_rejects_training_config(self) -> None:
        """Cloud dry-run should reject training configs with helpful error."""
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
        self.assertFalse(response["success"])
        self.assertIn("job config", response["error"].lower())
        self.assertIn("guidance://cloud-launch", response["error"])

    async def test_cancel_pending_cloud_launch_marks_intent(self) -> None:
        record = JobRecord(
            job_id="job-1",
            command="train",
            config_path="/tmp/train.yaml",
            cloud="gcp",
            cluster_name="",
            model_name="",
            submit_time="2026-02-12T17:09:25+00:00",
        )
        rt = JobRuntime()
        response = await job_service.cancel(record, rt)
        self.assertTrue(response["success"])
        self.assertTrue(rt.cancel_requested)

    async def test_cloud_launch_rejects_training_config_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "train.yaml"
            cfg_path.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            record = JobRecord(
                job_id="train_20260220_000001_abc123",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
                cluster_name="",
                model_name="",
                submit_time="2026-02-20T00:00:01+00:00",
            )
            rt = JobRuntime()
            rt.run_dir = Path(tmp_dir) / "run"
            with patch("oumi.mcp.job_service.get_registry") as mock_reg:
                mock_reg.return_value.update = lambda *a, **kw: None
                await job_service._launch_cloud(record, rt, client_cwd=tmp_dir)
            self.assertIsNotNone(rt.error_message)
            self.assertIn("job config", rt.error_message.lower())

    async def test_cloud_launch_reconciles_pending_cancel_after_id_available(
        self,
    ) -> None:
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
            record = JobRecord(
                job_id="train_20260220_000002_def456",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
                cluster_name="",
                model_name="",
                submit_time="2026-02-20T00:00:02+00:00",
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

            # Set up a mock registry that applies updates to the real record
            # so _launch_cloud can refresh it after updating.
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
                    "oumi.mcp.job_service.launcher.cancel",
                    return_value=cancelled,
                ) as mock_cancel,
                patch(
                    "oumi.mcp.job_service.get_registry",
                    return_value=mock_reg,
                ),
            ):
                await job_service._launch_cloud(record, rt, client_cwd=tmp_dir)  # type: ignore[attr-defined]

            mock_cancel.assert_called_once_with("cloud-456", "gcp", "cluster-b")
            self.assertTrue(rt.cancel_requested)

    def test_read_log_tail_returns_last_n_lines_for_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "big.log"
            lines = [f"line-{idx}" for idx in range(1, 20001)]
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tail, count = job_service._read_log_tail(log_path, 5)
        self.assertEqual(count, 5)
        self.assertEqual(tail.splitlines(), lines[-5:])

    def test_cluster_lifecycle_response_is_importable(self) -> None:
        from oumi.mcp.models import ClusterLifecycleResponse

        r: ClusterLifecycleResponse = {"success": True, "message": "ok"}
        self.assertTrue(r["success"])

    async def test_dry_run_cloud_shows_jobconfig_yaml_preview(self) -> None:
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
        self.assertTrue(response["success"])
        msg = response["message"]
        self.assertIn("oumi launch up", msg)
        self.assertIn("Generated JobConfig", msg)

    async def test_stop_cluster_calls_launcher_stop(self) -> None:
        with patch("oumi.mcp.job_service.launcher.stop") as mock_stop:
            response = await server.stop_cluster(cloud="gcp", cluster_name="sky-xxxx")
        self.assertTrue(response["success"])
        mock_stop.assert_called_once_with("gcp", "sky-xxxx")

    async def test_stop_cluster_returns_error_on_failure(self) -> None:
        with patch(
            "oumi.mcp.job_service.launcher.stop",
            side_effect=RuntimeError("network error"),
        ):
            response = await server.stop_cluster(cloud="gcp", cluster_name="sky-xxxx")
        self.assertFalse(response["success"])
        self.assertIn("Failed to stop cluster", response.get("error", ""))

    async def test_stop_cluster_rejects_empty_args(self) -> None:
        response = await server.stop_cluster(cloud="", cluster_name="sky-xxxx")
        self.assertFalse(response["success"])
        self.assertIn("required", response.get("error", ""))

    async def test_down_cluster_without_confirm_returns_dryrun_message(self) -> None:
        with patch("oumi.mcp.job_service.launcher.down") as mock_down:
            response = await server.down_cluster(cloud="gcp", cluster_name="sky-xxxx")
        mock_down.assert_not_called()
        self.assertTrue(response["success"])
        self.assertIn("IRREVERSIBLE", response.get("message", ""))

    async def test_down_cluster_with_confirm_calls_launcher_down(self) -> None:
        with patch("oumi.mcp.job_service.launcher.down") as mock_down:
            response = await server.down_cluster(
                cloud="gcp",
                cluster_name="sky-xxxx",
                confirm=True,
                user_confirmation="DOWN",
            )
        self.assertTrue(response["success"])
        mock_down.assert_called_once_with("gcp", "sky-xxxx")

    async def test_down_cluster_wrong_confirmation_phrase_is_blocked(self) -> None:
        with patch("oumi.mcp.job_service.launcher.down") as mock_down:
            response = await server.down_cluster(
                cloud="gcp",
                cluster_name="sky-xxxx",
                confirm=True,
                user_confirmation="EXECUTE",
            )
        mock_down.assert_not_called()
        self.assertFalse(response["success"])

    async def test_down_cluster_returns_error_on_failure(self) -> None:
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
        self.assertFalse(response["success"])
        self.assertIn("Failed to delete cluster", response.get("error", ""))

    def test_get_started_mentions_all_new_tools(self) -> None:
        result = server.get_started()
        self.assertIn("stop_cluster", result)
        self.assertIn("down_cluster", result)
        self.assertIn("Cloud Job Workflow", result)
        self.assertIn("Cluster Lifecycle", result)
        self.assertIn("suggested_configs", result)

    async def test_cancel_pending_cloud_job_cancels_runner_task(self) -> None:
        """cancel() should call runner_task.cancel() for pre-launch cloud jobs."""
        record = JobRecord(
            job_id="test-cancel-task",
            command="train",
            config_path="/tmp/fake.yaml",
            cloud="gcp",
            cluster_name="",
            model_name="",
            submit_time="2026-02-12T17:09:25+00:00",
        )
        rt = JobRuntime()
        mock_task = asyncio.Future()
        rt.runner_task = mock_task  # type: ignore[assignment]
        result = await cancel(record, rt)
        self.assertTrue(result["success"])
        self.assertTrue(mock_task.cancelled())

    def test_make_job_id_sanitizes_path_traversal(self) -> None:
        """make_job_id should strip path traversal characters from job_name."""
        self.assertNotIn("/", make_job_id("train", job_name="../../etc/evil"))
        self.assertNotIn("\\", make_job_id("train", job_name="..\\..\\evil"))
        self.assertNotIn("..", make_job_id("train", job_name="../up"))

    def test_make_job_id_rejects_empty_after_sanitization(self) -> None:
        """make_job_id should raise ValueError if job_name is only unsafe chars."""
        with self.assertRaises(ValueError):
            make_job_id("train", job_name="../../..")

    # -- Task 16: cancel_job end-to-end + cancel timeout --

    async def test_cancel_job_delegates_to_launcher_cancel(self) -> None:
        """cancel_job calls launcher.cancel with the correct arguments."""
        with (
            patch("oumi.mcp.job_service._resolve_job_record", return_value=None),
            patch(
                "oumi.mcp.job_service.launcher.cancel", return_value=None
            ) as mock_cancel,
        ):
            response = await server.cancel_job(
                job_id="sky-job-456",
                cloud="aws",
                cluster_name="cluster-b",
            )

        self.assertTrue(response["success"])
        mock_cancel.assert_called_once_with("sky-job-456", "aws", "cluster-b")

    async def test_cancel_job_timeout_returns_structured_error(self) -> None:
        """cancel_job returns a structured timeout error when launcher.cancel hangs."""
        import asyncio as _asyncio

        async def _fake_wait_for(coro, timeout):  # noqa: ANN001, ANN202
            # Consume the coroutine (close it cleanly) then raise TimeoutError
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
                job_id="sky-job-789",
                cloud="gcp",
                cluster_name="cluster-c",
            )

        self.assertFalse(response["success"])
        self.assertIn("timed out", response.get("error", ""))

    # -- Task 17: _launch_cloud with client_cwd --

    async def test_launch_cloud_client_cwd_sets_working_dir(self) -> None:
        """For job-config passthrough, client_cwd resolves relative working_dir."""
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
            record = JobRecord(
                job_id="train_20260220_000003_gh789",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
                cluster_name="",
                model_name="",
                submit_time="2026-02-20T00:00:03+00:00",
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
                    await job_service._launch_cloud(  # type: ignore[attr-defined]
                        record, rt, client_cwd=str(client_dir)
                    )

            # Job-config passthrough: relative working_dir resolved against client_cwd
            self.assertEqual(
                Path(captured_wd["working_dir"]).resolve(),  # noqa: ASYNC240
                client_dir.resolve(),
            )

    # -- Task 18: list_jobs via launcher.status() --

    async def test_list_jobs_calls_launcher_status(self) -> None:
        """list_jobs queries launcher.status() and enriches results with MCP IDs."""
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
                "oumi.mcp.job_service.launcher.status",
                return_value={"gcp": [job_status]},
            ),
            patch("oumi.mcp.job_service.get_registry") as mock_reg,
        ):
            mock_reg.return_value.find_by_cloud.return_value = None
            mock_reg.return_value.all.return_value = []
            summaries = await job_service._list_job_summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["cloud"], "gcp")
        self.assertEqual(summaries[0]["status"], "RUNNING")

    async def test_list_jobs_enriches_with_mcp_job_id(self) -> None:
        """list_jobs maps cloud identity to MCP job_id via registry lookup."""
        job_status = SimpleNamespace(
            id="sky-job-002",
            cluster="cluster-z",
            done=True,
            status="SUCCEEDED",
            state=SimpleNamespace(name="SUCCEEDED"),
            metadata={},
        )
        mcp_record = JobRecord(
            job_id="sky-job-002",
            command="train",
            config_path="/tmp/t.yaml",
            cloud="aws",
            cluster_name="cluster-z",
            model_name="meta-llama/Llama-3.1-8B",
            submit_time="2026-02-20T00:00:04+00:00",
        )

        with (
            patch(
                "oumi.mcp.job_service.launcher.status",
                return_value={"aws": [job_status]},
            ),
            patch("oumi.mcp.job_service.get_registry") as mock_reg,
        ):
            mock_reg.return_value.find_by_cloud.return_value = mcp_record
            mock_reg.return_value.all.return_value = []
            summaries = await job_service._list_job_summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["job_id"], "sky-job-002")
        self.assertEqual(summaries[0]["model_name"], "meta-llama/Llama-3.1-8B")


if __name__ == "__main__":
    unittest.main()
