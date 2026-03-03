# pyright: reportOperatorIssue=false
"""Tests for oumi.mcp.sync_service — version detection, URL building, sync flow."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest

from oumi.mcp.sync_service import (
    config_sync,
    get_configs_zip_url,
    get_oumi_git_tag,
    is_oumi_dev_build,
)


@pytest.mark.parametrize("version", ["0.8.dev35+ge2b81b3fe", "1.0.0.dev1", "0.7+local"])
def test_dev_versions(version: str):
    assert is_oumi_dev_build(version) is True


@pytest.mark.parametrize("version", ["0.7", "1.0.0", "0.8.1"])
def test_release_versions(version: str):
    assert is_oumi_dev_build(version) is False


def test_git_tag_dev_build_returns_none():
    with patch(
        "oumi.mcp.sync_service.get_package_version", return_value="0.8.dev35+g123"
    ):
        assert get_oumi_git_tag() is None


def test_git_tag_missing_returns_none():
    with patch("oumi.mcp.sync_service.get_package_version", return_value=None):
        assert get_oumi_git_tag() is None


def test_configs_zip_url_tagged():
    url, prefix = get_configs_zip_url("v0.7")
    assert "v0.7" in url
    assert prefix == "oumi-0.7/configs/"


def test_configs_zip_url_main_branch():
    url, prefix = get_configs_zip_url(None)
    assert "main" in url
    assert prefix == "oumi-main/configs/"


def test_config_sync_skips_when_fresh():
    with (
        patch("oumi.mcp.sync_service._is_cache_stale", return_value=False),
        patch("oumi.mcp.sync_service.get_configs_source", return_value="cache:0.7"),
    ):
        result = config_sync(force=False)
    assert result["ok"] is True
    assert result["skipped"] is True


def test_config_sync_force_ignores_freshness(tmp_path: Path):
    cache_dir = tmp_path / "cache"

    zip_path = tmp_path / "fake.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("oumi-0.7/configs/train.yaml", "model:\n  model_name: gpt2\n")

    zip_bytes = zip_path.read_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes
    mock_response.raise_for_status = MagicMock()

    mock_client_inst = MagicMock()
    mock_client_inst.get.return_value = mock_response

    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_inst)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("oumi.mcp.sync_service.get_cache_dir", return_value=cache_dir),
        patch("oumi.mcp.sync_service.get_oumi_version", return_value="0.7"),
        patch("oumi.mcp.sync_service.get_oumi_git_tag", return_value="v0.7"),
        patch("oumi.mcp.sync_service.clear_config_caches"),
        patch("oumi.mcp.sync_service.httpx.Client", mock_client_cls),
    ):
        result = config_sync(force=True)

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["configs_synced"] >= 1


def test_config_sync_http_error_returns_failure():
    import httpx

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = httpx.HTTPError("connection failed")

    with (
        patch("oumi.mcp.sync_service._is_cache_stale", return_value=True),
        patch("oumi.mcp.sync_service.get_cache_dir", return_value=Path("/tmp/fake")),
        patch("oumi.mcp.sync_service.get_oumi_version", return_value="0.7"),
        patch("oumi.mcp.sync_service.get_oumi_git_tag", return_value="v0.7"),
        patch("oumi.mcp.sync_service.httpx.Client", return_value=mock_client),
    ):
        result = config_sync(force=False)

    assert result["ok"] is False
    assert "connection failed" in result["error"]
