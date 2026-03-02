"""Tests for cloud file resolution pre-flight checks."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from oumi.mcp.preflight_service import _check_cloud_files, validate_datasets


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Helper to write a YAML string to a temp file and return its Path."""
    def _write(content: str, name: str = "config.yaml") -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(content))
        return p
    return _write


# -- Job-config passthrough mode --

class TestJobConfigPassthrough:
    """Tests for job-config passthrough (config has resources/setup/run keys)."""

    def test_file_mounts_source_exists(self, tmp_config, tmp_path: Path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text("data")
        cfg = {
            "resources": {"cloud": "aws"},
            "file_mounts": {"/data/train.jsonl": str(data_file)},
            "run": "oumi train -c config.yaml",
        }
        result = _check_cloud_files(cfg, tmp_path / "job.yaml", "aws")
        assert result[str(data_file)] == "ok"

    def test_file_mounts_source_missing(self, tmp_config, tmp_path: Path):
        cfg = {
            "resources": {"cloud": "aws"},
            "file_mounts": {"/data/train.jsonl": "/nonexistent/train.jsonl"},
            "run": "oumi train -c config.yaml",
        }
        result = _check_cloud_files(cfg, tmp_path / "job.yaml", "aws")
        assert result["/nonexistent/train.jsonl"] == "missing_local_source"

    def test_working_dir_dot_is_ok(self, tmp_config, tmp_path: Path):
        """working_dir: . is the correct portable default (resolved via client_cwd)."""
        cfg = {
            "resources": {"cloud": "aws"},
            "working_dir": ".",
            "run": "oumi train -c config.yaml",
        }
        config_path = tmp_path / "job.yaml"
        config_path.write_text("")
        result = _check_cloud_files(cfg, config_path, "aws")
        assert "working_dir_suspicious" not in result.values()

    def test_working_dir_absolute_exists(self, tmp_config, tmp_path: Path):
        cfg = {
            "resources": {"cloud": "aws"},
            "working_dir": str(tmp_path),
            "run": "oumi train -c config.yaml",
        }
        result = _check_cloud_files(cfg, tmp_path / "job.yaml", "aws")
        assert "working_dir_suspicious" not in result.values()

    def test_working_dir_missing_warns(self, tmp_config, tmp_path: Path):
        cfg = {
            "resources": {"cloud": "aws"},
            "working_dir": "/nonexistent/dir",
            "run": "oumi train -c config.yaml",
        }
        result = _check_cloud_files(cfg, tmp_path / "job.yaml", "aws")
        assert result.get("working_dir: /nonexistent/dir") == "working_dir_suspicious"

    def test_file_mounts_tilde_expansion(self, tmp_config, tmp_path: Path):
        cfg = {
            "resources": {"cloud": "aws"},
            "file_mounts": {"/remote/.netrc": "~/.netrc"},
            "run": "oumi train",
        }
        result = _check_cloud_files(cfg, tmp_path / "job.yaml", "aws")
        expanded = str(Path("~/.netrc").expanduser())
        if Path(expanded).exists():
            assert result["~/.netrc"] == "ok"
        else:
            assert result["~/.netrc"] == "missing_local_source"


# -- Training-config wrapping mode --

class TestTrainingConfigWrapping:
    """Tests for training-config wrapping (no resources/setup/run keys)."""

    def test_relative_path_blocking(self, tmp_config, tmp_path: Path):
        cfg = {
            "model": {"model_name": "meta-llama/Llama-3.1-8B"},
            "data": {
                "train": {
                    "datasets": [{"dataset_name": "x", "dataset_path": "data/train.jsonl"}]
                }
            },
        }
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "aws")
        assert result["data/train.jsonl"] == "not_reachable_on_vm"

    def test_relative_path_exists_locally_still_unreachable(self, tmp_config, tmp_path: Path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "train.jsonl").write_text("data")
        cfg = {
            "model": {"model_name": "meta-llama/Llama-3.1-8B"},
            "training": {"output_dir": "data/train.jsonl"},
        }
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "aws")
        assert result["data/train.jsonl"] == "not_reachable_on_vm"

    def test_remote_absolute_path_warning(self, tmp_config, tmp_path: Path):
        cfg = {
            "model": {"model_name": "meta-llama/Llama-3.1-8B"},
            "training": {"output_dir": "/home/ubuntu/output"},
        }
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "aws")
        assert result["/home/ubuntu/output"] == "unverifiable_remote"

    def test_hf_repo_skipped(self, tmp_config, tmp_path: Path):
        cfg = {
            "model": {"model_name": "meta-llama/Llama-3.1-8B"},
        }
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "aws")
        assert "meta-llama/Llama-3.1-8B" not in result

    def test_local_machine_path_blocking(self, tmp_config, tmp_path: Path):
        cfg = {
            "model": {"model_name": "meta-llama/Llama-3.1-8B"},
            "training": {"output_dir": str(Path.home() / "output")},
        }
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "aws")
        home_path = str(Path.home() / "output")
        assert result[home_path] == "not_reachable_on_vm"

    def test_empty_config_no_checks(self, tmp_config, tmp_path: Path):
        cfg = {"model": {"model_name": "meta-llama/Llama-3.1-8B"}}
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "aws")
        assert "meta-llama/Llama-3.1-8B" not in result


# -- Edge cases --

class TestEdgeCases:
    def test_no_cloud_returns_empty(self, tmp_path: Path):
        cfg = {"model": {"model_name": "x"}}
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "")
        assert result == {}

    def test_local_cloud_returns_empty(self, tmp_path: Path):
        cfg = {"model": {"model_name": "x"}}
        result = _check_cloud_files(cfg, tmp_path / "config.yaml", "local")
        assert result == {}


# -- Integration with _pre_flight_check --

_MOCK_CLOUD_READINESS = ([], [], {
    "sky_installed": True,
    "enabled_clouds": ["AWS"],
    "target_cloud_ready": True,
    "target_cloud": "aws",
})

_MOCK_HARDWARE = ([], [], {
    "accelerator_type": "none",
    "accelerator_count": 0,
    "gpu_name": None,
    "gpu_memory_gb": None,
    "compute_capability": None,
    "cuda_version": None,
    "packages": {},
})


class TestPreFlightIntegration:
    """Test that _pre_flight_check surfaces cloud_file_checks correctly."""

    @patch("oumi.mcp.preflight_service.validate_datasets", return_value={})
    @patch("oumi.mcp.preflight_service.check_hardware", return_value=_MOCK_HARDWARE)
    @patch("oumi.mcp.preflight_service.check_cloud_readiness", return_value=_MOCK_CLOUD_READINESS)
    @patch("oumi.mcp.preflight_service.whoami", side_effect=Exception("no token"))
    def test_missing_file_mount_is_blocking(
        self, _hf, _cloud, _hw, _ds, tmp_path: Path
    ):
        from oumi.mcp.preflight_service import _pre_flight_check

        job_yaml = tmp_path / "job.yaml"
        job_yaml.write_text(textwrap.dedent("""\
            resources:
              cloud: aws
              accelerators: A100:1
            file_mounts:
              /data/train.jsonl: /nonexistent/local/train.jsonl
            run: oumi train -c config.yaml
        """))

        result = _pre_flight_check(str(job_yaml), client_cwd=str(tmp_path), cloud="aws")
        assert result["blocking"] is True
        assert any("file_mounts source" in e for e in result["errors"])
        checks = result.get("cloud_file_checks", {})
        assert checks.get("/nonexistent/local/train.jsonl") == "missing_local_source"

    @patch("oumi.mcp.preflight_service.validate_datasets", return_value={})
    @patch("oumi.mcp.preflight_service.check_hardware", return_value=_MOCK_HARDWARE)
    @patch("oumi.mcp.preflight_service.check_cloud_readiness", return_value=_MOCK_CLOUD_READINESS)
    @patch("oumi.mcp.preflight_service.whoami", side_effect=Exception("no token"))
    def test_training_config_relative_path_is_blocking(
        self, _hf, _cloud, _hw, _ds, tmp_path: Path
    ):
        from oumi.mcp.preflight_service import _pre_flight_check

        config_yaml = tmp_path / "train_config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            model:
              model_name: meta-llama/Llama-3.1-8B
            data:
              train:
                datasets:
                  - dataset_name: custom
                    dataset_path: data/train.jsonl
        """))

        result = _pre_flight_check(str(config_yaml), client_cwd=str(tmp_path), cloud="aws")
        assert result["blocking"] is True
        assert any("no delivery mechanism" in e for e in result["errors"])


# -- validate_datasets client_cwd --

class TestValidateDatasetsClientCwd:
    """Verify that validate_datasets resolves relative ds_path against client_cwd."""

    def test_relative_ds_path_resolved_against_client_cwd(self, tmp_path: Path):
        """A relative dataset_path should resolve under client_cwd and report ok_local."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.jsonl").write_text("{}\n")

        cfg = {
            "data": {
                "train": {
                    "datasets": [
                        {"dataset_name": "", "dataset_path": "data/train.jsonl"}
                    ]
                }
            }
        }
        result = validate_datasets(cfg, client_cwd=str(tmp_path))
        assert result.get("data/train.jsonl") == "ok_local"

    def test_relative_ds_path_not_found_without_client_cwd(self, tmp_path: Path):
        """Without client_cwd, a relative path that only exists under tmp_path is not found."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.jsonl").write_text("{}\n")

        cfg = {
            "data": {
                "train": {
                    "datasets": [
                        {"dataset_name": "", "dataset_path": "data/train.jsonl"}
                    ]
                }
            }
        }
        # No client_cwd — resolves against cwd, not tmp_path, so not found
        result = validate_datasets(cfg, client_cwd="")
        assert result.get("data/train.jsonl") != "ok_local"
