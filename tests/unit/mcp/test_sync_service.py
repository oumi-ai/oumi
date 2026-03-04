# pyright: reportOperatorIssue=false
"""Tests for oumi.mcp.sync_service — version detection, tree API sync flow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oumi.mcp.sync_service import (
    _fetch_yaml_paths,
    config_sync,
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


def test_git_tag_release():
    with patch("oumi.mcp.sync_service.get_package_version", return_value="0.7"):
        assert get_oumi_git_tag() == "v0.7"


def test_config_sync_skips_when_fresh():
    with (
        patch("oumi.mcp.sync_service._is_cache_stale", return_value=False),
        patch("oumi.mcp.sync_service.get_configs_source", return_value="cache:0.7"),
    ):
        result = config_sync(force=False)
    assert result["ok"] is True
    assert result["skipped"] is True


def _make_mock_client(yaml_paths: list[str]) -> MagicMock:
    """Build a mock httpx.Client that simulates the Trees API + raw downloads."""
    root_tree_response = MagicMock()
    root_tree_response.status_code = 200
    root_tree_response.raise_for_status = MagicMock()
    root_tree_response.json.return_value = {
        "tree": [
            {"path": "configs", "type": "tree", "sha": "abc123"},
            {"path": "src", "type": "tree", "sha": "def456"},
        ]
    }

    configs_tree_response = MagicMock()
    configs_tree_response.status_code = 200
    configs_tree_response.raise_for_status = MagicMock()
    configs_tree_response.json.return_value = {
        "truncated": False,
        "tree": [
            {"path": p, "type": "blob", "sha": f"sha_{i}"}
            for i, p in enumerate(yaml_paths)
        ],
    }

    raw_response = MagicMock()
    raw_response.status_code = 200
    raw_response.content = b"model:\n  model_name: gpt2\n"

    def mock_get(url, **_kwargs):
        if "/git/trees/abc123" in url:
            return configs_tree_response
        if "/git/trees/" in url:
            return root_tree_response
        if "raw.githubusercontent.com" in url:
            return raw_response
        return MagicMock(status_code=404)

    mock_client = MagicMock()
    mock_client.get.side_effect = mock_get
    return mock_client


def test_fetch_yaml_paths():
    yaml_paths = ["recipes/llama3/train.yaml", "apis/anthropic/infer.yaml"]
    mock_client = _make_mock_client(yaml_paths)
    result = _fetch_yaml_paths(mock_client, "main")
    assert result == yaml_paths


def test_fetch_yaml_paths_filters_non_yaml():
    root_resp = MagicMock()
    root_resp.status_code = 200
    root_resp.raise_for_status = MagicMock()
    root_resp.json.return_value = {
        "tree": [{"path": "configs", "type": "tree", "sha": "abc123"}]
    }
    configs_resp = MagicMock()
    configs_resp.status_code = 200
    configs_resp.raise_for_status = MagicMock()
    configs_resp.json.return_value = {
        "truncated": False,
        "tree": [
            {"path": "train.yaml", "type": "blob", "sha": "s1"},
            {"path": "README.md", "type": "blob", "sha": "s2"},
            {"path": "subdir", "type": "tree", "sha": "s3"},
        ],
    }

    def mock_get(url, **_kwargs):
        if "/git/trees/abc123" in url:
            return configs_resp
        return root_resp

    mock_client = MagicMock()
    mock_client.get.side_effect = mock_get
    result = _fetch_yaml_paths(mock_client, "main")
    assert result == ["train.yaml"]


def test_config_sync_force_downloads(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    yaml_paths = ["recipes/llama3/train.yaml"]
    mock_client = _make_mock_client(yaml_paths)

    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("oumi.mcp.sync_service.get_cache_dir", return_value=cache_dir),
        patch("oumi.mcp.sync_service.get_oumi_version", return_value="0.7"),
        patch("oumi.mcp.sync_service._get_git_ref", return_value=("v0.7", "tag:v0.7")),
        patch("oumi.mcp.sync_service.clear_config_caches"),
        patch("oumi.mcp.sync_service.httpx.Client", mock_client_cls),
    ):
        result = config_sync(force=True)

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["configs_synced"] == 1


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
        patch("oumi.mcp.sync_service._get_git_ref", return_value=("v0.7", "tag:v0.7")),
        patch("oumi.mcp.sync_service.httpx.Client", return_value=mock_client),
    ):
        result = config_sync(force=False)

    assert result["ok"] is False
    assert "connection failed" in result["error"]


def test_config_sync_no_configs_in_tree():
    """Tree API returns no YAML files."""
    mock_client = _make_mock_client([])

    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("oumi.mcp.sync_service._is_cache_stale", return_value=True),
        patch("oumi.mcp.sync_service.get_cache_dir", return_value=Path("/tmp/fake")),
        patch("oumi.mcp.sync_service.get_oumi_version", return_value="0.7"),
        patch("oumi.mcp.sync_service._get_git_ref", return_value=("v0.7", "tag:v0.7")),
        patch("oumi.mcp.sync_service.httpx.Client", mock_client_cls),
    ):
        result = config_sync(force=False)

    assert result["ok"] is False
    assert "No YAML configs" in result["error"]
