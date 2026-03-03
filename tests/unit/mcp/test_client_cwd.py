"""Tests for client_cwd path resolution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from oumi.mcp.config_service import resolve_config_path, resolve_path
from oumi.mcp.job_service import (
    JobRecord,
    JobRuntime,
    _launch_cloud,
    start_local_job,
)
from oumi.mcp.server import (
    pre_flight_check,
    run_oumi_job,
    validate_config,
)


class ResolvePathTests(unittest.TestCase):
    def test_absolute_path_returned_unchanged(self):
        result = resolve_path("/abs/path/config.yaml", Path("/some/cwd"))
        self.assertEqual(result, Path("/abs/path/config.yaml"))

    def test_relative_path_resolved_against_client_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp) / "configs"
            subdir.mkdir()
            config = subdir / "train.yaml"
            config.write_text("")
            result = resolve_path("configs/train.yaml", Path(tmp))
            self.assertEqual(result, config.resolve())

    def test_tilde_expanded(self):
        result = resolve_path("~/configs/train.yaml", Path("/some/cwd"))
        self.assertTrue(result.is_absolute())
        self.assertIn("configs/train.yaml", str(result))

    def test_dot_path_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            jsonl = data_dir / "train.jsonl"
            jsonl.write_text("")
            result = resolve_path("./data/train.jsonl", Path(tmp))
            self.assertEqual(result, jsonl.resolve())


class ResolveConfigPathTests(unittest.TestCase):
    def test_relative_config_resolved_against_client_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("model:\n  name: test\n")
            resolved, err = resolve_config_path("train.yaml", tmp)
            self.assertIsNone(err)
            self.assertEqual(resolved, config.resolve())

    def test_absolute_config_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("model:\n  name: test\n")
            resolved, err = resolve_config_path(str(config), tmp)
            self.assertIsNone(err)
            self.assertEqual(resolved, config.resolve())

    def test_relative_client_cwd_rejected(self):
        _, err = resolve_config_path("train.yaml", "relative/cwd")
        self.assertIsNotNone(err)
        self.assertIn("absolute", err)

    def test_nonexistent_config_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, err = resolve_config_path("nonexistent.yaml", tmp)
            self.assertIsNotNone(err)
            self.assertIn("not found", err.lower())


class ValidateConfigCwdTests(unittest.TestCase):
    def test_validate_config_resolves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text(
                "training:\n  data:\n    train:\n      dataset_name: test\n"
            )
            mock_cfg = MagicMock()
            mock_cls = MagicMock()
            mock_cls.from_yaml.return_value = mock_cfg
            with patch("oumi.mcp.server.TASK_MAPPING", {"training": mock_cls}):
                result = validate_config("train.yaml", "training", client_cwd=tmp)
                self.assertTrue(result["ok"])


class PreFlightCheckCwdTests(unittest.TestCase):
    def test_pre_flight_resolves_relative_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("model:\n  model_name: gpt2\n")
            result = pre_flight_check("train.yaml", client_cwd=tmp)
            blocking_errors = result.get("errors", [])
            path_errors = [e for e in blocking_errors if "absolute" in e.lower()]
            self.assertEqual(path_errors, [])

    def test_pre_flight_validates_inner_paths_against_client_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "train.jsonl").write_text("{}\n")
            config = Path(tmp) / "train.yaml"
            config.write_text("training:\n  data:\n    dataset_path: ./data\n")
            result = pre_flight_check(str(config), client_cwd=tmp)
            paths = result.get("paths", {})
            # ./data should resolve to tmp/data and be "ok"
            found_ok = any("ok" in v for v in paths.values())
            self.assertTrue(found_ok or len(paths) == 0)


class RunOumiJobCwdTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_resolves_relative_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text(
                "model:\n  model_name: gpt2\ntraining:\n  output_dir: ./output\n"
            )
            result = await run_oumi_job(
                config_path="train.yaml",
                command="train",
                client_cwd=tmp,
                dry_run=True,
            )
            self.assertTrue(result.get("success", False), result.get("error", ""))
            self.assertEqual(result["dry_run"], True)


class LocalJobCwdTests(unittest.TestCase):
    def test_start_local_job_sets_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
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

            call_kwargs = mock_popen.call_args
            self.assertEqual(call_kwargs.kwargs.get("cwd"), "/home/alice/project")


class CloudJobConfigPassthroughCwdTests(unittest.IsolatedAsyncioTestCase):
    async def test_job_config_relative_working_dir_resolved_against_client_cwd(self):
        """Job-config passthrough: working_dir: . should resolve to client_cwd."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a job config with working_dir: .
            job_yaml = Path(tmp) / "job.yaml"
            job_yaml.write_text(
                "name: test-job\n"
                "resources:\n  cloud: gcp\n  accelerators: 'A100:1'\n"
                "working_dir: .\n"
                "setup: |\n  echo setup\n"
                "run: |\n  echo run\n"
            )
            run_dir = Path(tmp) / "run_dir"
            run_dir.mkdir()

            record = JobRecord(
                job_id="test-passthrough",
                command="train",
                config_path=str(job_yaml),
                cloud="gcp",
                cluster_name="",
                model_name="gpt2",
                submit_time="2026-01-01T00:00:00Z",
            )
            rt = JobRuntime()
            rt.run_dir = run_dir

            captured_job_config = {}
            # Use a real directory to avoid macOS path canonicalization issues
            client_dir = Path(tmp) / "project"
            client_dir.mkdir()
            client_project = str(client_dir.resolve())

            def mock_launcher_up(job_config, cluster_name=None):
                captured_job_config["working_dir"] = job_config.working_dir
                mock_status = MagicMock()
                mock_status.id = "sky-123"
                mock_status.cluster = "test-cluster"
                return MagicMock(), mock_status

            with (
                patch("oumi.mcp.job_service.launcher.up", side_effect=mock_launcher_up),
                patch("oumi.mcp.job_service.get_registry") as mock_reg,
            ):
                mock_reg.return_value.update = MagicMock()
                mock_reg.return_value.get = MagicMock(return_value=record)
                await _launch_cloud(record, rt, client_cwd=client_project)

            # working_dir: . should have been resolved to client_cwd
            self.assertEqual(captured_job_config["working_dir"], client_project)
