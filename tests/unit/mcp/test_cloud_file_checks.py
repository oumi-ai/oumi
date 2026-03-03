"""Tests for cloud file resolution pre-flight checks."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from oumi.mcp.preflight_service import validate_datasets, validate_paths_cloud


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Helper to write a YAML string to a temp file and return its Path."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(content))
        return p

    return _write


def test_file_mounts_source_exists(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("data")
    cfg = {
        "resources": {"cloud": "aws"},
        "file_mounts": {"/data/train.jsonl": str(data_file)},
        "run": "oumi train -c config.yaml",
    }
    result = validate_paths_cloud(cfg, tmp_path / "job.yaml", str(tmp_path), "aws")
    assert result[str(data_file)] == "valid"


def test_file_mounts_source_missing(tmp_path: Path):
    cfg = {
        "resources": {"cloud": "aws"},
        "file_mounts": {"/data/train.jsonl": "/nonexistent/train.jsonl"},
        "run": "oumi train -c config.yaml",
    }
    result = validate_paths_cloud(cfg, tmp_path / "job.yaml", str(tmp_path), "aws")
    assert result["/nonexistent/train.jsonl"] == "missing_local_source"


def test_working_dir_dot_is_ok(tmp_path: Path):
    cfg = {
        "resources": {"cloud": "aws"},
        "working_dir": ".",
        "run": "oumi train -c config.yaml",
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text("")
    result = validate_paths_cloud(cfg, config_path, str(tmp_path), "aws")
    assert "working_dir_suspicious" not in result.values()


def test_working_dir_absolute_exists(tmp_path: Path):
    cfg = {
        "resources": {"cloud": "aws"},
        "working_dir": str(tmp_path),
        "run": "oumi train -c config.yaml",
    }
    result = validate_paths_cloud(cfg, tmp_path / "job.yaml", str(tmp_path), "aws")
    assert "working_dir_suspicious" not in result.values()


def test_working_dir_missing_warns(tmp_path: Path):
    cfg = {
        "resources": {"cloud": "aws"},
        "working_dir": "/nonexistent/dir",
        "run": "oumi train -c config.yaml",
    }
    result = validate_paths_cloud(cfg, tmp_path / "job.yaml", str(tmp_path), "aws")
    assert result.get("working_dir: /nonexistent/dir") == "working_dir_suspicious"


def test_file_mounts_tilde_expansion(tmp_path: Path):
    cfg = {
        "resources": {"cloud": "aws"},
        "file_mounts": {"/remote/.netrc": "~/.netrc"},
        "run": "oumi train",
    }
    result = validate_paths_cloud(cfg, tmp_path / "job.yaml", str(tmp_path), "aws")
    expanded = str(Path("~/.netrc").expanduser())
    if Path(expanded).exists():
        assert result["~/.netrc"] == "valid"
    else:
        assert result["~/.netrc"] == "missing_local_source"


def test_relative_path_warning(tmp_path: Path):
    cfg = {
        "model": {"model_name": "meta-llama/Llama-3.1-8B"},
        "data": {
            "train": {
                "datasets": [{"dataset_name": "x", "dataset_path": "data/train.jsonl"}]
            }
        },
    }
    result = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "aws")
    assert result["data/train.jsonl"] == "not_found_warning"


def test_relative_path_exists_locally(tmp_path: Path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text("data")
    cfg = {
        "model": {"model_name": "meta-llama/Llama-3.1-8B"},
        "training": {"output_dir": "data/train.jsonl"},
    }
    result = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "aws")
    assert result["data/train.jsonl"] == "valid"


def test_remote_absolute_path_warning(tmp_path: Path):
    cfg = {
        "model": {"model_name": "meta-llama/Llama-3.1-8B"},
        "training": {"output_dir": "/home/ubuntu/output"},
    }
    result = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "aws")
    assert result["/home/ubuntu/output"] == "unverifiable_remote"


def test_hf_repo_skipped(tmp_path: Path):
    cfg = {"model": {"model_name": "meta-llama/Llama-3.1-8B"}}
    result = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "aws")
    assert "meta-llama/Llama-3.1-8B" not in result


def test_local_machine_path_blocking(tmp_path: Path):
    cfg = {
        "model": {"model_name": "meta-llama/Llama-3.1-8B"},
        "training": {"output_dir": str(Path.home() / "output")},
    }
    result = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "aws")
    assert result[str(Path.home() / "output")] == "local_machine_path_error"


@pytest.mark.parametrize("cloud", ["", "local"])
def test_empty_or_local_cloud_returns_empty(tmp_path: Path, cloud: str):
    cfg = {"model": {"model_name": "x"}}
    result = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), cloud)
    assert result == {}


_MOCK_CLOUD_READINESS = (
    [],
    [],
    {
        "sky_installed": True,
        "enabled_clouds": ["AWS"],
        "target_cloud_ready": True,
        "target_cloud": "aws",
    },
)

_MOCK_HARDWARE = (
    [],
    [],
    {
        "accelerator_type": "none",
        "accelerator_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "compute_capability": None,
        "cuda_version": None,
        "packages": {},
    },
)


@patch("oumi.mcp.preflight_service.validate_datasets", return_value={})
@patch("oumi.mcp.preflight_service.check_hardware", return_value=_MOCK_HARDWARE)
@patch(
    "oumi.mcp.preflight_service.check_cloud_readiness",
    return_value=_MOCK_CLOUD_READINESS,
)
@patch("oumi.mcp.preflight_service.whoami", side_effect=Exception("no token"))
def test_missing_file_mount_is_blocking(_hf, _cloud, _hw, _ds, tmp_path: Path):
    from oumi.mcp.preflight_service import _pre_flight_check

    job_yaml = tmp_path / "job.yaml"
    job_yaml.write_text(
        textwrap.dedent("""\
        resources:
          cloud: aws
          accelerators: A100:1
        file_mounts:
          /data/train.jsonl: /nonexistent/local/train.jsonl
        run: oumi train -c config.yaml
    """)
    )

    result = _pre_flight_check(str(job_yaml), client_cwd=str(tmp_path), cloud="aws")
    assert result["blocking"] is True
    assert any("file_mounts source" in e for e in result["errors"])
    assert (
        result["paths"].get("/nonexistent/local/train.jsonl") == "missing_local_source"
    )


@patch("oumi.mcp.preflight_service.validate_datasets", return_value={})
@patch("oumi.mcp.preflight_service.check_hardware", return_value=_MOCK_HARDWARE)
@patch(
    "oumi.mcp.preflight_service.check_cloud_readiness",
    return_value=_MOCK_CLOUD_READINESS,
)
@patch("oumi.mcp.preflight_service.whoami", side_effect=Exception("no token"))
def test_training_config_relative_path_is_warning(
    _hf, _cloud, _hw, _ds, tmp_path: Path
):
    from oumi.mcp.preflight_service import _pre_flight_check

    config_yaml = tmp_path / "train_config.yaml"
    config_yaml.write_text(
        textwrap.dedent("""\
        model:
          model_name: meta-llama/Llama-3.1-8B
        data:
          train:
            datasets:
              - dataset_name: custom
                dataset_path: data/train.jsonl
    """)
    )

    result = _pre_flight_check(str(config_yaml), client_cwd=str(tmp_path), cloud="aws")
    assert any("not found locally" in w for w in result["warnings"])


def test_relative_ds_path_resolved_against_client_cwd(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n")

    cfg = {
        "data": {
            "train": {
                "datasets": [{"dataset_name": "", "dataset_path": "data/train.jsonl"}]
            }
        }
    }
    result = validate_datasets(cfg, client_cwd=str(tmp_path))
    assert result.get("data/train.jsonl") == "ok_local"
