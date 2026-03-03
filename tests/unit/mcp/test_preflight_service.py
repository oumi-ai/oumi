# pyright: reportTypedDictNotRequiredAccess=false
"""Tests for oumi.mcp.preflight_service — path/dataset/hardware validation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oumi.mcp.preflight_service import (
    _compat_error,
    _compat_warning,
    _empty_cloud_readiness,
    _is_local_machine_path,
    _looks_like_hf_repo,
    _pre_flight_check,
    get_gpu_info,
    get_repos,
    validate_datasets,
    validate_paths_cloud,
    validate_paths_local,
)


@pytest.mark.parametrize("val", ["meta-llama/Llama-3.1-8B", "org/model"])
def test_looks_like_hf_repo_valid(val: str):
    assert _looks_like_hf_repo(val) is True


@pytest.mark.parametrize(
    "val", ["", "/abs/path", "./rel", "~/home", "no-slash", "a/b/c"]
)
def test_looks_like_hf_repo_invalid(val: str):
    assert _looks_like_hf_repo(val) is False


def test_is_local_machine_path_home():
    assert _is_local_machine_path(str(Path.home() / "project")) is True


def test_is_local_machine_path_users():
    assert _is_local_machine_path("/Users/alice/work") is True


def test_is_local_machine_path_remote():
    assert _is_local_machine_path("/home/ubuntu/output") is False


def test_is_local_machine_path_relative():
    assert _is_local_machine_path("relative/path") is False


def test_validate_paths_local_exists(tmp_path: Path):
    d = tmp_path / "output"
    d.mkdir()
    cfg = {"training": {"output_dir": str(d)}}
    paths = validate_paths_local(cfg, tmp_path)
    assert paths[str(d)] == "valid"


def test_validate_paths_local_not_found(tmp_path: Path):
    cfg = {"training": {"output_dir": "/nonexistent/dir_xyz"}}
    paths = validate_paths_local(cfg, tmp_path)
    assert paths["/nonexistent/dir_xyz"] == "not_found"


def test_validate_paths_local_relative(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    cfg = {"training": {"output_dir": "data"}}
    paths = validate_paths_local(cfg, tmp_path)
    assert any("valid" in v for v in paths.values())


def test_validate_paths_local_hf_repo_skipped(tmp_path: Path):
    cfg = {"model": {"model_path": "meta-llama/Llama-3.1-8B"}}
    paths = validate_paths_local(cfg, tmp_path)
    assert len(paths) == 0


def test_validate_paths_local_empty(tmp_path: Path):
    assert validate_paths_local({}, tmp_path) == {}


def test_validate_paths_cloud_blocks_local_machine(tmp_path: Path):
    cfg = {"training": {"output_dir": str(Path.home() / "output")}}
    paths = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "gcp")
    assert any(v == "local_machine_path_error" for v in paths.values())


def test_validate_paths_cloud_remote_absolute(tmp_path: Path):
    cfg = {"training": {"output_dir": "/home/ubuntu/output"}}
    paths = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "gcp")
    assert any(v == "unverifiable_remote" for v in paths.values())


def test_validate_paths_cloud_relative_exists(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    cfg = {"training": {"output_dir": "data"}}
    paths = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "gcp")
    assert any(v == "valid" for v in paths.values())


def test_validate_paths_cloud_relative_not_found(tmp_path: Path):
    cfg = {"training": {"output_dir": "missing_dir"}}
    paths = validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "gcp")
    assert any(v == "not_found_warning" for v in paths.values())


def test_validate_datasets_registry_hit(tmp_path: Path):
    cfg = {"data": {"train": {"datasets": [{"dataset_name": "test_ds"}]}}}
    mock_reg = MagicMock()
    mock_reg.get_dataset.return_value = "something"
    with (
        patch("oumi.mcp.preflight_service.REGISTRY", mock_reg, create=True),
        patch.dict("sys.modules", {"oumi.core.registry": MagicMock(REGISTRY=mock_reg)}),
    ):
        results = validate_datasets(cfg, str(tmp_path))
    assert results.get("test_ds") == "ok_registry"


def test_validate_datasets_local_path(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    cfg = {
        "data": {
            "train": {"datasets": [{"dataset_name": "x", "dataset_path": str(data)}]}
        }
    }
    with patch(
        "oumi.mcp.preflight_service.REGISTRY", create=True, side_effect=Exception
    ):
        results = validate_datasets(cfg, str(tmp_path))
    assert any("ok_local" in v for v in results.values())


def test_validate_datasets_not_found(tmp_path: Path):
    cfg = {
        "data": {"train": {"datasets": [{"dataset_name": "fake_nonexistent_ds_xyz"}]}}
    }
    mock_reg = MagicMock()
    mock_reg.get_dataset.return_value = None
    mock_datasets_mod = MagicMock()
    mock_datasets_mod.load_dataset_builder.side_effect = Exception("not found")

    with patch.dict(
        "sys.modules",
        {
            "oumi.core.registry": MagicMock(REGISTRY=mock_reg),
            "datasets": mock_datasets_mod,
        },
    ):
        results = validate_datasets(cfg, str(tmp_path))
    assert any(v == "not_found" for v in results.values())


def test_validate_datasets_empty():
    assert validate_datasets({}) == {}


def test_get_gpu_info_no_torch():
    with patch.object(
        __import__("oumi.mcp.preflight_service", fromlist=["torch"]), "torch", None
    ):
        info = get_gpu_info()
    assert info["accelerator_type"] == "none"


def test_get_repos_model():
    cfg = {"model": {"model_name": "meta-llama/Llama-3.1-8B"}}
    repos = get_repos(cfg)
    assert "meta-llama/Llama-3.1-8B" in repos
    assert "model" in repos["meta-llama/Llama-3.1-8B"]


def test_get_repos_dataset():
    cfg = {"data": {"train": {"datasets": [{"dataset_name": "org/dataset"}]}}}
    repos = get_repos(cfg)
    assert "org/dataset" in repos


def _run_preflight(tmp_path: Path, cloud_errors=None, cloud_warnings=None):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("model: {model_name: test/model}\n")
    cloud_result = (cloud_errors or [], cloud_warnings or [], _empty_cloud_readiness())
    with (
        patch(
            "oumi.mcp.preflight_service.check_cloud_readiness",
            return_value=cloud_result,
        ),
        patch("oumi.mcp.preflight_service.whoami", side_effect=Exception),
    ):
        return _pre_flight_check(str(cfg), client_cwd=str(tmp_path))


def test_compat_error_sets_flag(tmp_path: Path):
    result = _run_preflight(tmp_path, cloud_errors=[_compat_error("fail")])
    assert result["skypilot_compat_issue"] is True


def test_compat_warning_sets_flag(tmp_path: Path):
    result = _run_preflight(tmp_path, cloud_warnings=[_compat_warning("warn")])
    assert result["skypilot_compat_issue"] is True


def test_no_compat_issue(tmp_path: Path):
    assert _run_preflight(tmp_path)["skypilot_compat_issue"] is False


def test_early_return_path_error_has_field():
    result = _pre_flight_check("/nonexistent/path.yaml", client_cwd="/tmp")
    assert result["skypilot_compat_issue"] is False
