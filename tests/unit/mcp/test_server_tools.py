# pyright: reportCallIssue=false
"""Tests for metadata extraction, warnings, error paths."""

from unittest.mock import patch

from oumi.mcp.environment_service import _build_missing_env_warning
from oumi.mcp.server import (
    _build_version_warning,
    _extract_job_metadata_from_cfg,
    get_config,
)


def test_extract_job_metadata_normal():
    cfg = {"model": {"model_name": "gpt2"}, "training": {"output_dir": "./out"}}
    model, output = _extract_job_metadata_from_cfg(cfg)
    assert model == "gpt2"
    assert output == "./out"


def test_extract_job_metadata_missing_model():
    model, output = _extract_job_metadata_from_cfg({})
    assert model == "unknown"
    assert output == "./output"


def test_extract_job_metadata_empty_model_name():
    model, _ = _extract_job_metadata_from_cfg({"model": {"model_name": ""}})
    assert model == "unknown"


def test_extract_job_metadata_non_dict_model():
    model, _ = _extract_job_metadata_from_cfg({"model": "not_a_dict"})
    assert model == "unknown"


def test_version_warning_none_for_unknown():
    with (
        patch("oumi.mcp.server.get_oumi_version", return_value="unknown"),
        patch("oumi.mcp.server.get_configs_source", return_value="bundled:0.7"),
    ):
        assert _build_version_warning() == ""


def test_version_warning_cache_main_with_release():
    with (
        patch("oumi.mcp.server.get_oumi_version", return_value="0.7"),
        patch("oumi.mcp.server.get_configs_source", return_value="cache:main"),
        patch("oumi.mcp.server.is_oumi_dev_build", return_value=False),
    ):
        assert "main branch" in _build_version_warning()


def test_version_warning_bundled_mismatch():
    with (
        patch("oumi.mcp.server.get_oumi_version", return_value="0.8"),
        patch("oumi.mcp.server.get_configs_source", return_value="bundled:0.7"),
        patch("oumi.mcp.server.is_oumi_dev_build", return_value=False),
    ):
        assert "bundled" in _build_version_warning()


def test_missing_env_no_warning_when_empty():
    with patch.dict("os.environ", {}, clear=True):
        assert _build_missing_env_warning(None) == ""


def test_missing_env_warns_when_not_forwarded():
    with patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True):
        assert "WANDB_API_KEY" in _build_missing_env_warning(None)


def test_missing_env_no_warning_when_forwarded():
    with patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True):
        assert _build_missing_env_warning({"WANDB_API_KEY": "secret"}) == ""


def test_get_config_not_found():
    with (
        patch("oumi.mcp.server.get_all_configs", return_value=[]),
        patch("oumi.mcp.server.find_config_match", return_value=None),
    ):
        result = get_config("nonexistent")
    assert result["error"] != ""
    assert result["path"] == ""
