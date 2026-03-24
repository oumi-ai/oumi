# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config synchronization service for Oumi MCP.

Handles config version detection, cache staleness checks, and syncing
configs.
"""

import logging
import os
import shutil
from pathlib import Path

import httpx

from oumi.mcp.config_service import (
    clear_config_caches,
    get_bundled_configs_dir,
    get_cache_dir,
    get_package_version,
)
from oumi.mcp.constants import (
    BUNDLED_OUMI_VERSION,
    CONFIG_SYNC_TIMEOUT_SECONDS,
    CONFIGS_SYNC_MARKER,
    CONFIGS_VERSION_MARKER,
    GITHUB_API_URL,
    GITHUB_RAW_URL,
)
from oumi.mcp.models import ConfigSyncResponse

logger = logging.getLogger(__name__)


def get_oumi_version() -> str:
    """Return the installed oumi version, or "unknown"."""
    return get_package_version("oumi") or "unknown"


def is_oumi_dev_build(version: str) -> bool:
    """Return True if *version* looks like a setuptools_scm dev build."""
    return ".dev" in version or "+" in version


def get_oumi_git_tag() -> str | None:
    """Map the installed oumi version to the corresponding Git tag.

    Dev builds (e.g. ``0.8.dev35+ge2b81b3fe``) have no matching tag.
    Release versions (e.g. ``0.7``) map to ``v0.7``.
    """
    version = get_package_version("oumi")
    if not version or is_oumi_dev_build(version):
        return None
    return f"v{version}"


def _get_git_ref() -> tuple[str, str]:
    """Return ``(git_ref, source_label)`` for the current oumi version.

    Release builds use the matching Git tag; dev builds use ``main``.
    """
    tag = get_oumi_git_tag()
    if tag:
        return tag, f"tag:{tag}"
    return "main", "main"


def _read_version_marker() -> str:
    """Read the oumi version that the cached configs were synced for."""
    marker = get_cache_dir() / CONFIGS_VERSION_MARKER
    try:
        return marker.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_version_marker(version: str) -> None:
    """Record which oumi version the cached configs correspond to."""
    marker = get_cache_dir() / CONFIGS_VERSION_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(version, encoding="utf-8")


def _is_cache_stale() -> bool:
    """Check whether the cached configs need to be refreshed.

    Returns True if the cache directory doesn't exist, has no YAML files,
    or the installed oumi version no longer matches the cached version marker.
    Configs only change with releases, so there is no time-based expiry.
    """
    cache_dir = get_cache_dir()
    marker = cache_dir / CONFIGS_SYNC_MARKER

    if not cache_dir.exists() or not any(cache_dir.rglob("*.yaml")):
        return True

    if not marker.exists():
        return True

    cached_version = _read_version_marker()
    current_version = get_oumi_version()
    if cached_version and current_version != "unknown":
        if cached_version != current_version:
            logger.info(
                "Oumi version changed (%s -> %s); cache is stale",
                cached_version,
                current_version,
            )
            return True

    return False


def _touch_sync_marker() -> None:
    """Write/update the sync timestamp marker file."""
    marker = get_cache_dir() / CONFIGS_SYNC_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")


def get_configs_source() -> str:
    """Describe which source the current configs directory comes from.

    Possible values: ``"cache:<version>"``, ``"cache:main"``,
    ``"bundled:<version>"``, ``"env:<path>"``, or ``"unknown"``.
    """
    env_dir = os.environ.get("OUMI_MCP_CONFIGS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and any(p.rglob("*.yaml")):
            return f"env:{env_dir}"

    cache = get_cache_dir()
    if cache.is_dir() and any(cache.rglob("*.yaml")):
        cached_ver = _read_version_marker()
        return f"cache:{cached_ver}" if cached_ver else "cache:main"

    bundled = get_bundled_configs_dir()
    if bundled.is_dir():
        return f"bundled:{BUNDLED_OUMI_VERSION}"

    return "unknown"


def _fetch_yaml_paths(client: httpx.Client, ref: str) -> list[str]:
    """Fetch the list of YAML config paths using the Git Trees API.

    Makes 2 API calls:
    1. Get the root tree to find the ``configs/`` subtree SHA.
    2. Get the full recursive tree for ``configs/`` and filter to .yaml blobs.

    Args:
        client: An ``httpx.Client`` instance.
        ref: Git ref (tag like ``v0.7`` or branch like ``main``).

    Returns:
        List of relative paths (e.g. ``"recipes/llama3/sft/train.yaml"``).

    Raises:
        ValueError: If the ``configs/`` directory is not found in the tree.
        httpx.HTTPStatusError: On API errors.
    """
    root_resp = client.get(f"{GITHUB_API_URL}/git/trees/{ref}")
    root_resp.raise_for_status()
    root_tree = root_resp.json()

    configs_sha = None
    for entry in root_tree.get("tree", []):
        if entry["path"] == "configs" and entry["type"] == "tree":
            configs_sha = entry["sha"]
            break

    if not configs_sha:
        raise ValueError(f"configs/ directory not found in repo tree at ref {ref}")

    tree_resp = client.get(
        f"{GITHUB_API_URL}/git/trees/{configs_sha}",
        params={"recursive": "1"},
    )
    tree_resp.raise_for_status()
    tree_data = tree_resp.json()

    if tree_data.get("truncated"):
        logger.warning("Git tree response was truncated; some configs may be missing")

    return [
        entry["path"]
        for entry in tree_data.get("tree", [])
        if entry["type"] == "blob" and entry["path"].endswith(".yaml")
    ]


def _download_configs(
    client: httpx.Client,
    ref: str,
    yaml_paths: list[str],
    target_dir: Path,
) -> int:
    """Download YAML config files from raw.githubusercontent.com.

    Args:
        client: An ``httpx.Client`` instance.
        ref: Git ref (tag or branch).
        yaml_paths: Relative paths under ``configs/``.
        target_dir: Local directory to write files into.

    Returns:
        Number of files successfully downloaded.
    """
    downloaded = 0
    for path in yaml_paths:
        url = f"{GITHUB_RAW_URL}/{ref}/configs/{path}"
        resp = client.get(url)
        if resp.status_code != 200:
            logger.warning("Failed to download %s (HTTP %d)", path, resp.status_code)
            continue

        dest = target_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        downloaded += 1

    return downloaded


def config_sync(force: bool = False) -> ConfigSyncResponse:
    """Sync configs from the Oumi repository, matching the installed version.

    Uses the GitHub Git Trees API to discover config files, then downloads
    only YAML files from raw.githubusercontent.com. Release builds download
    from the matching Git tag; dev builds use main.

    Skips download if cache is fresh (unless force=True).

    Args:
        force: Sync regardless of cache freshness.
    """
    if not force and not _is_cache_stale():
        logger.info("Config cache is fresh, skipping sync")
        return {
            "ok": True,
            "skipped": True,
            "error": None,
            "configs_synced": 0,
            "source": get_configs_source(),
        }

    cache_dir = get_cache_dir()
    oumi_ver = get_oumi_version()
    ref, source_label = _get_git_ref()

    try:
        logger.info(
            "Starting config sync from oumi-ai/oumi (%s, oumi=%s)",
            source_label,
            oumi_ver,
        )

        with httpx.Client(
            follow_redirects=True,
            timeout=CONFIG_SYNC_TIMEOUT_SECONDS,
        ) as client:
            try:
                yaml_paths = _fetch_yaml_paths(client, ref)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404 and ref != "main":
                    logger.warning("Ref %s not found (404); falling back to main", ref)
                    ref = "main"
                    source_label = "main"
                    yaml_paths = _fetch_yaml_paths(client, ref)
                else:
                    raise

            if not yaml_paths:
                return {
                    "ok": False,
                    "skipped": False,
                    "error": "No YAML configs found in repository tree",
                    "configs_synced": 0,
                    "source": source_label,
                }

            logger.info("Found %d YAML configs, downloading...", len(yaml_paths))

            staging_dir = cache_dir.parent / (cache_dir.name + ".staging")
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True)

            try:
                downloaded = _download_configs(client, ref, yaml_paths, staging_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

        if downloaded == 0:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return {
                "ok": False,
                "skipped": False,
                "error": "All config downloads failed",
                "configs_synced": 0,
                "source": source_label,
            }

        backup_dir = cache_dir.parent / (cache_dir.name + ".backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        if cache_dir.exists():
            logger.info("Backing up old cache: %s -> %s", cache_dir, backup_dir)
            shutil.move(str(cache_dir), str(backup_dir))
        try:
            shutil.move(str(staging_dir), str(cache_dir))
        except Exception:
            if backup_dir.exists():
                logger.warning("Cache install failed; restoring backup")
                shutil.move(str(backup_dir), str(cache_dir))
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        clear_config_caches()
        _write_version_marker(oumi_ver)
        _touch_sync_marker()

        logger.info(
            "Successfully synced %d config files (%s)", downloaded, source_label
        )

        return {
            "ok": True,
            "skipped": False,
            "error": None,
            "configs_synced": downloaded,
            "source": source_label,
        }

    except httpx.HTTPError as e:
        error_msg = f"Failed to sync configs: {e}"
        logger.error(error_msg)
        return {
            "ok": False,
            "skipped": False,
            "error": error_msg,
            "configs_synced": 0,
            "source": source_label,
        }

    except Exception as e:
        error_msg = f"Config sync failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "ok": False,
            "skipped": False,
            "error": error_msg,
            "configs_synced": 0,
            "source": source_label,
        }
