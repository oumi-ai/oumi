"""Tests for oumi.mcp.preflight_service — path/dataset/hardware validation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oumi.mcp.preflight_service import (
    _is_local_machine_path,
    _looks_like_hf_repo,
    get_gpu_info,
    get_repos,
    validate_datasets,
    validate_paths,
)

# ------------------------------------------------------------------
# Helper predicates
# ------------------------------------------------------------------


class TestLooksLikeHfRepo:
    @pytest.mark.parametrize("val", ["meta-llama/Llama-3.1-8B", "org/model"])
    def test_valid(self, val: str):
        assert _looks_like_hf_repo(val) is True

    @pytest.mark.parametrize(
        "val", ["", "/abs/path", "./rel", "~/home", "no-slash", "a/b/c"]
    )
    def test_invalid(self, val: str):
        assert _looks_like_hf_repo(val) is False


class TestIsLocalMachinePath:
    def test_home_path(self):
        assert _is_local_machine_path(str(Path.home() / "project")) is True

    def test_users_path(self):
        assert _is_local_machine_path("/Users/alice/work") is True

    def test_remote_absolute(self):
        assert _is_local_machine_path("/home/ubuntu/output") is False

    def test_relative(self):
        assert _is_local_machine_path("relative/path") is False


# ------------------------------------------------------------------
# validate_paths
# ------------------------------------------------------------------


class TestValidatePaths:
    def test_local_absolute_exists(self, tmp_path: Path):
        d = tmp_path / "output"
        d.mkdir()
        cfg = {"training": {"output_dir": str(d)}}
        paths = validate_paths(cfg, tmp_path)
        assert paths[str(d)] == "ok"

    def test_local_absolute_not_found(self, tmp_path: Path):
        cfg = {"training": {"output_dir": "/nonexistent/dir_xyz"}}
        paths = validate_paths(cfg, tmp_path)
        assert paths["/nonexistent/dir_xyz"] == "not_found"

    def test_local_relative_resolved(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        cfg = {"training": {"output_dir": "data"}}
        paths = validate_paths(cfg, tmp_path)
        assert any("ok" in v for v in paths.values())

    def test_hf_repo_skipped(self, tmp_path: Path):
        cfg = {"model": {"model_path": "meta-llama/Llama-3.1-8B"}}
        paths = validate_paths(cfg, tmp_path)
        assert len(paths) == 0

    def test_cloud_blocks_local_machine_path(self, tmp_path: Path):
        cfg = {"training": {"output_dir": str(Path.home() / "output")}}
        paths = validate_paths(cfg, tmp_path, cloud="gcp")
        assert any(v == "local_machine_path_error" for v in paths.values())

    def test_cloud_remote_absolute_ok(self, tmp_path: Path):
        cfg = {"training": {"output_dir": "/home/ubuntu/output"}}
        paths = validate_paths(cfg, tmp_path, cloud="gcp")
        assert any(v == "ok_remote" for v in paths.values())

    def test_cloud_relative_exists(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        cfg = {"training": {"output_dir": "data"}}
        paths = validate_paths(cfg, tmp_path, cloud="gcp")
        assert any(v == "ok" for v in paths.values())

    def test_cloud_relative_not_found(self, tmp_path: Path):
        cfg = {"training": {"output_dir": "missing_dir"}}
        paths = validate_paths(cfg, tmp_path, cloud="gcp")
        assert any(v == "not_found_warning" for v in paths.values())

    def test_empty_config(self, tmp_path: Path):
        assert validate_paths({}, tmp_path) == {}


# ------------------------------------------------------------------
# validate_datasets
# ------------------------------------------------------------------


class TestValidateDatasets:
    def test_registry_hit(self, tmp_path: Path):
        cfg = {"data": {"train": {"datasets": [{"dataset_name": "test_ds"}]}}}
        mock_reg = MagicMock()
        mock_reg.get_dataset.return_value = "something"
        with (
            patch("oumi.mcp.preflight_service.REGISTRY", mock_reg, create=True),
            patch.dict(
                "sys.modules", {"oumi.core.registry": MagicMock(REGISTRY=mock_reg)}
            ),
        ):
            results = validate_datasets(cfg, str(tmp_path))
        assert results.get("test_ds") == "ok_registry"

    def test_local_path_hit(self, tmp_path: Path):
        data = tmp_path / "data.jsonl"
        data.write_text("{}\n")
        cfg = {
            "data": {
                "train": {
                    "datasets": [{"dataset_name": "x", "dataset_path": str(data)}]
                }
            }
        }
        with patch(
            "oumi.mcp.preflight_service.REGISTRY", create=True, side_effect=Exception
        ):
            # Registry lookup should fail, but local path should be found
            results = validate_datasets(cfg, str(tmp_path))
        assert any("ok_local" in v for v in results.values())

    def test_not_found(self, tmp_path: Path):
        cfg = {
            "data": {
                "train": {"datasets": [{"dataset_name": "fake_nonexistent_ds_xyz"}]}
            }
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

    def test_empty_config(self):
        assert validate_datasets({}) == {}


# ------------------------------------------------------------------
# get_gpu_info
# ------------------------------------------------------------------


class TestGetGpuInfo:
    def test_no_torch(self):
        with patch.object(
            __import__("oumi.mcp.preflight_service", fromlist=["torch"]), "torch", None
        ):
            info = get_gpu_info()
        assert info["accelerator_type"] == "none"

    def test_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = MagicMock()
        props.name = "A100"
        props.total_mem = 80_000_000_000
        props.major = 8
        props.minor = 0
        mock_torch.cuda.get_device_properties.return_value = props
        with patch.object(
            __import__("oumi.mcp.preflight_service", fromlist=["torch"]),
            "torch",
            mock_torch,
        ):
            info = get_gpu_info()
        assert info["accelerator_type"] == "cuda"
        assert info["gpu_name"] == "A100"


# ------------------------------------------------------------------
# get_repos
# ------------------------------------------------------------------


class TestGetRepos:
    def test_model_extraction(self):
        cfg = {"model": {"model_name": "meta-llama/Llama-3.1-8B"}}
        from oumi.mcp.preflight_service import get_repos

        repos = get_repos(cfg)
        assert "meta-llama/Llama-3.1-8B" in repos
        assert "model" in repos["meta-llama/Llama-3.1-8B"]

    def test_empty_config(self):
        assert get_repos({}) == {}

    def test_dataset_extraction(self):
        cfg = {"data": {"train": {"datasets": [{"dataset_name": "org/dataset"}]}}}
        repos = get_repos(cfg)
        assert "org/dataset" in repos
