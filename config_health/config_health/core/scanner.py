"""Config discovery — walks configs/ directory and finds all YAML files."""

from __future__ import annotations

import glob
import os

import yaml


# ── Shared YAML cache ──────────────────────────────────────────────
# Single cache used by all modules to avoid redundant disk reads + parses.
# On a 500-config --exhaustive run this eliminates ~3000 redundant parses.
_yaml_cache: dict[str, dict | None] = {}
_YAML_NOT_LOADED = object()


def load_yaml_cached(path: str) -> dict | None:
    """Load and cache a YAML file. Returns None on any error."""
    cached = _yaml_cache.get(path, _YAML_NOT_LOADED)
    if cached is not _YAML_NOT_LOADED:
        return cached  # type: ignore[return-value]
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        result = data if isinstance(data, dict) else None
    except Exception:
        result = None
    _yaml_cache[path] = result
    return result


def clear_yaml_cache() -> None:
    """Clear the shared YAML cache. Call between independent runs."""
    _yaml_cache.clear()


def scan_config_paths(repo_root: str) -> list[str]:
    """Find all YAML config files under configs/."""
    configs_dir = os.path.join(repo_root, "configs")
    if not os.path.isdir(configs_dir):
        return []
    pattern = os.path.join(configs_dir, "**", "*.yaml")
    return sorted(glob.glob(pattern, recursive=True))


def get_category(path: str, repo_root: str) -> str:
    """Extract top-level category from path (recipes, apis, examples, projects)."""
    rel = os.path.relpath(path, os.path.join(repo_root, "configs"))
    parts = rel.split(os.sep)
    return parts[0] if parts else "unknown"


def get_model_family(path: str, repo_root: str) -> str | None:
    """Extract model family from path."""
    rel = os.path.relpath(path, os.path.join(repo_root, "configs"))
    parts = rel.split(os.sep)

    # recipes/<model_family>/...
    if len(parts) > 1 and parts[0] == "recipes":
        return parts[1]

    # apis/<provider>/...
    if len(parts) > 1 and parts[0] == "apis":
        return parts[1]

    # projects/<project>/...
    if len(parts) > 1 and parts[0] == "projects":
        return parts[1]

    return None


def find_repo_root(start: str | None = None) -> str:
    """Walk up from start to find the repo root (contains configs/ dir)."""
    current = os.path.abspath(start or os.getcwd())
    while current != os.path.dirname(current):
        if os.path.isdir(os.path.join(current, "configs")) and os.path.isdir(
            os.path.join(current, "src", "oumi")
        ):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError(
        "Could not find oumi repo root (directory with configs/ and src/oumi/)"
    )
