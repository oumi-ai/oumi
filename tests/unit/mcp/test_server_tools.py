"""Tests for oumi.mcp.server tool functions — search, get, list, docs, preflight."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oumi.mcp.environment_service import _build_missing_env_warning
from oumi.mcp.server import (
    _build_version_warning,
    _extract_job_metadata_from_cfg,
    get_config,
    list_categories,
    search_configs,
)


# ------------------------------------------------------------------
# _extract_job_metadata_from_cfg
# ------------------------------------------------------------------


class TestExtractJobMetadata:
    def test_normal_config(self):
        cfg = {"model": {"model_name": "gpt2"}, "training": {"output_dir": "./out"}}
        model, output = _extract_job_metadata_from_cfg(cfg)
        assert model == "gpt2"
        assert output == "./out"

    def test_missing_model(self):
        model, output = _extract_job_metadata_from_cfg({})
        assert model == "unknown"
        assert output == "./output"

    def test_empty_model_name(self):
        cfg = {"model": {"model_name": ""}}
        model, _ = _extract_job_metadata_from_cfg(cfg)
        assert model == "unknown"

    def test_non_dict_model(self):
        cfg = {"model": "not_a_dict"}
        model, _ = _extract_job_metadata_from_cfg(cfg)
        assert model == "unknown"


# ------------------------------------------------------------------
# _build_version_warning
# ------------------------------------------------------------------


class TestBuildVersionWarning:
    def test_no_warning_for_unknown(self):
        with patch("oumi.mcp.server.get_oumi_version", return_value="unknown"), \
             patch("oumi.mcp.server.get_configs_source", return_value="bundled:0.7"):
            assert _build_version_warning() == ""

    def test_cache_main_with_release(self):
        with patch("oumi.mcp.server.get_oumi_version", return_value="0.7"), \
             patch("oumi.mcp.server.get_configs_source", return_value="cache:main"), \
             patch("oumi.mcp.server.is_oumi_dev_build", return_value=False):
            w = _build_version_warning()
            assert "main branch" in w

    def test_bundled_version_mismatch(self):
        with patch("oumi.mcp.server.get_oumi_version", return_value="0.8"), \
             patch("oumi.mcp.server.get_configs_source", return_value="bundled:0.7"), \
             patch("oumi.mcp.server.is_oumi_dev_build", return_value=False):
            w = _build_version_warning()
            assert "bundled" in w


# ------------------------------------------------------------------
# _build_missing_env_warning
# ------------------------------------------------------------------


class TestBuildMissingEnvWarning:
    def test_no_warning_when_no_env(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _build_missing_env_warning(None) == ""

    def test_warns_when_local_env_not_forwarded(self):
        with patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True):
            w = _build_missing_env_warning(None)
            assert "WANDB_API_KEY" in w

    def test_no_warning_when_forwarded(self):
        with patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True):
            w = _build_missing_env_warning({"WANDB_API_KEY": "secret"})
            assert w == ""


# ------------------------------------------------------------------
# search_configs tool
# ------------------------------------------------------------------


class TestSearchConfigsTool:
    def test_returns_list(self):
        mock_configs = [
            {"path": "a.yaml", "description": "", "model_name": "", "task_type": "",
             "datasets": [], "reward_functions": [], "peft_type": ""}
        ]
        with patch("oumi.mcp.server.get_all_configs", return_value=mock_configs), \
             patch("oumi.mcp.server.search_configs_service", return_value=mock_configs):
            result = search_configs()
        assert isinstance(result, list)
        assert len(result) == 1


# ------------------------------------------------------------------
# get_config tool
# ------------------------------------------------------------------


class TestGetConfigTool:
    def test_not_found(self):
        with patch("oumi.mcp.server.get_all_configs", return_value=[]), \
             patch("oumi.mcp.server.find_config_match", return_value=None):
            result = get_config("nonexistent")
        assert result["error"] != ""
        assert result["path"] == ""

    def test_found(self, tmp_path: Path):
        meta = {
            "path": "train.yaml", "description": "test", "model_name": "gpt2",
            "task_type": "sft", "datasets": [], "reward_functions": [], "peft_type": "",
        }
        p = tmp_path / "train.yaml"
        p.write_text("training:\n  learning_rate: 0.001\n")
        with patch("oumi.mcp.server.get_all_configs", return_value=[meta]), \
             patch("oumi.mcp.server.find_config_match", return_value=meta), \
             patch("oumi.mcp.server.get_configs_dir", return_value=tmp_path):
            result = get_config("train.yaml")
        assert result["error"] == ""
        assert result["path"] == "train.yaml"
        assert "learning_rate" in result["content"]


# ------------------------------------------------------------------
# list_categories tool
# ------------------------------------------------------------------


class TestListCategoriesTool:
    def test_returns_response(self, tmp_path: Path):
        (tmp_path / "recipes").mkdir()
        with patch("oumi.mcp.server.get_configs_dir", return_value=tmp_path), \
             patch("oumi.mcp.server.get_all_configs", return_value=[]), \
             patch("oumi.mcp.server.get_oumi_version", return_value="0.7"), \
             patch("oumi.mcp.server.get_configs_source", return_value="bundled:0.7"), \
             patch("oumi.mcp.server._build_version_warning", return_value=""):
            result = list_categories()
        assert "categories" in result
        assert "recipes" in result["categories"]
