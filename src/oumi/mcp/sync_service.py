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
configs from the Oumi GitHub repository.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

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
    GITHUB_CONFIGS_ZIP_URL,
    GITHUB_REPO_URL,
    GITHUB_ZIP_PREFIX,
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


def get_configs_zip_url(tag: str | None = None) -> tuple[str, str]:
    """Return ``(zip_url, zip_prefix)`` for downloading configs.

    If *tag* is provided, returns the tagged archive URL; otherwise falls back
    to the main branch.
    """
    if tag:
        url = f"{GITHUB_REPO_URL}/archive/refs/tags/{tag}.zip"
        prefix = f"oumi-{tag.lstrip('v')}/configs/"
        return url, prefix
    return GITHUB_CONFIGS_ZIP_URL, GITHUB_ZIP_PREFIX


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


def _safe_extract(zip_ref: ZipFile, members: list[str], target_dir: Path) -> None:
    """Extract *members* from *zip_ref* into *target_dir* safely.

    Validates that every extracted path stays within *target_dir* to
    prevent Zip Slip (path-traversal) attacks.
    """
    real_target = target_dir.resolve()
    for member in members:
        member_path = (real_target / member).resolve()
        if real_target not in member_path.parents and member_path != real_target:
            raise ValueError(f"Attempted path traversal in zip entry: {member}")
        zip_ref.extract(member, target_dir)


def config_sync(force: bool = False) -> ConfigSyncResponse:
    """Sync configs from the Oumi repository, matching the installed version.

    Release builds download from the matching Git tag; dev builds use main.
    Skips download if cache is fresh (unless force=True).

    Args:
        force: Sync regardless of cache age.
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
    temp_dir = None
    oumi_ver = get_oumi_version()
    tag = get_oumi_git_tag()

    zip_url, zip_prefix = get_configs_zip_url(tag)
    source_label = f"tag:{tag}" if tag else "main"

    try:
        logger.info(
            "Starting config sync from oumi-ai/oumi (%s, oumi=%s)",
            source_label,
            oumi_ver,
        )
        temp_dir = Path(tempfile.mkdtemp(prefix="oumi_config_sync_"))
        zip_path = temp_dir / "oumi.zip"

        # TODO: Use GitHub Contents API to fetch only the configs/ directory
        # instead of downloading the full repository archive.
        logger.info("Downloading configs from %s", zip_url)
        with httpx.Client(
            follow_redirects=True,
            timeout=CONFIG_SYNC_TIMEOUT_SECONDS,
        ) as client:
            response = client.get(zip_url)

            if response.status_code == 404 and tag:
                logger.warning(
                    "Tag %s not found (404); falling back to main branch", tag
                )
                zip_url, zip_prefix = get_configs_zip_url(None)
                source_label = "main"
                response = client.get(zip_url)

            response.raise_for_status()
            zip_path.write_bytes(response.content)

        logger.info("Downloaded %d bytes from %s", len(response.content), source_label)
        logger.info("Extracting configs from archive")

        archive_root = zip_prefix.split("/")[0]

        with ZipFile(zip_path, "r") as zip_ref:
            config_files = [
                name for name in zip_ref.namelist() if name.startswith(zip_prefix)
            ]

            if not config_files:
                return {
                    "ok": False,
                    "skipped": False,
                    "error": "No configs directory found in repository archive",
                    "configs_synced": 0,
                    "source": source_label,
                }

            _safe_extract(zip_ref, config_files, temp_dir)

        extracted_configs = temp_dir / archive_root / "configs"
        if not extracted_configs.exists():
            return {
                "ok": False,
                "skipped": False,
                "error": f"Extracted configs not found at {extracted_configs}",
                "configs_synced": 0,
                "source": source_label,
            }

        backup_dir = cache_dir.parent / (cache_dir.name + ".backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        if cache_dir.exists():
            logger.info("Backing up old cache: %s -> %s", cache_dir, backup_dir)
            shutil.move(str(cache_dir), str(backup_dir))
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Moving new configs to %s", cache_dir)
            shutil.move(str(extracted_configs), str(cache_dir))
        except Exception:
            if backup_dir.exists():
                logger.warning("New cache install failed; restoring backup")
                shutil.move(str(backup_dir), str(cache_dir))
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        clear_config_caches()

        _touch_sync_marker()
        _write_version_marker(oumi_ver)

        config_count = len(list(cache_dir.rglob("*.yaml")))
        logger.info(
            "Successfully synced %d config files (%s)", config_count, source_label
        )

        return {
            "ok": True,
            "skipped": False,
            "error": None,
            "configs_synced": config_count,
            "source": source_label,
        }

    except httpx.HTTPError as e:
        error_msg = f"Failed to download configs: {e}"
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

    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up temp directory %s: %s", temp_dir, e)
