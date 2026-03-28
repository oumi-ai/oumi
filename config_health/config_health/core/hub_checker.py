"""HuggingFace Hub existence checks with disk cache."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from config_health.core.models import (
    CheckResult,
    CheckStatus,
    ConfigEntry,
    Severity,
)

# Remote inference engines whose model_name values are provider-specific
# identifiers, not HuggingFace Hub repo IDs.
_REMOTE_ENGINES = frozenset(
    {
        "ANTHROPIC",
        "OPENAI",
        "GOOGLE",
        "GOOGLE_GEMINI",
        "GOOGLE_VERTEX",
        "OPENROUTER",
        "TOGETHER",
        "FIREWORKS",
        "PARASAIL",
        "LAMBDA",
        "REMOTE",
        "REMOTE_VLLM",
    }
)

_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days
_CACHE_DIR = os.path.expanduser("~/.cache/config_health")
_CACHE_FILE = os.path.join(_CACHE_DIR, "hub_cache.json")


@dataclass
class _CacheEntry:
    exists: bool
    checked_at: float


class HubChecker:
    """Check model/dataset existence on HuggingFace Hub with caching."""

    def __init__(self, offline: bool = False):
        self.offline = offline
        self._cache: dict[str, _CacheEntry] = {}
        self._load_cache()

    def check_config(self, entry: ConfigEntry) -> list[CheckResult]:
        """Check model and dataset existence for a config entry."""
        results: list[CheckResult] = []

        if entry.model_name:
            results.append(self._check_model(entry))

        for ds_name in entry.datasets:
            results.append(self._check_dataset(entry, ds_name))

        return results

    def _check_model(self, entry: ConfigEntry) -> CheckResult:
        """Check if model_name exists on HuggingFace Hub."""
        model_name = entry.model_name
        assert model_name is not None

        # Skip remote API engines — their model_name is a provider-specific
        # identifier (e.g. "gpt-4o", "claude-3-5-sonnet-latest"), not an HF repo.
        if entry.engine and entry.engine in _REMOTE_ENGINES:
            return CheckResult(
                config_path=entry.path,
                check_name="model_exists",
                status=CheckStatus.SKIP,
                message=f"Remote engine ({entry.engine}): {model_name}",
                severity=Severity.INFO,
            )

        # Skip local paths
        if model_name.startswith(("./", "/", "~")):
            return CheckResult(
                config_path=entry.path,
                check_name="model_exists",
                status=CheckStatus.SKIP,
                message=f"Local path: {model_name}",
                severity=Severity.INFO,
            )

        exists = self._check_hub_repo(f"model:{model_name}", model_name, "model")
        if exists is None:
            return CheckResult(
                config_path=entry.path,
                check_name="model_exists",
                status=CheckStatus.SKIP,
                message=f"Offline — cannot verify: {model_name}",
                severity=Severity.INFO,
            )
        if exists:
            return CheckResult(
                config_path=entry.path,
                check_name="model_exists",
                status=CheckStatus.PASS,
                message=f"Model exists: {model_name}",
                severity=Severity.INFO,
            )
        return CheckResult(
            config_path=entry.path,
            check_name="model_exists",
            status=CheckStatus.FAIL,
            message=f"Model not found on HF Hub: {model_name}",
            severity=Severity.ERROR,
        )

    def _check_dataset(self, entry: ConfigEntry, ds_name: str) -> CheckResult:
        """Check if dataset_name exists on HuggingFace Hub."""
        # Skip non-HF datasets
        if ds_name.startswith(("./", "/", "~")) or ":" in ds_name:
            return CheckResult(
                config_path=entry.path,
                check_name="dataset_exists",
                status=CheckStatus.SKIP,
                message=f"Non-HF dataset: {ds_name}",
                severity=Severity.INFO,
            )

        exists = self._check_hub_repo(f"dataset:{ds_name}", ds_name, "dataset")
        if exists is None:
            return CheckResult(
                config_path=entry.path,
                check_name="dataset_exists",
                status=CheckStatus.SKIP,
                message=f"Offline — cannot verify: {ds_name}",
                severity=Severity.INFO,
            )
        if exists:
            return CheckResult(
                config_path=entry.path,
                check_name="dataset_exists",
                status=CheckStatus.PASS,
                message=f"Dataset exists: {ds_name}",
                severity=Severity.INFO,
            )
        return CheckResult(
            config_path=entry.path,
            check_name="dataset_exists",
            status=CheckStatus.FAIL,
            message=f"Dataset not found on HF Hub: {ds_name}",
            severity=Severity.ERROR,
        )

    def _check_hub_repo(
        self, cache_key: str, repo_id: str, repo_type: str
    ) -> bool | None:
        """Check if a repo exists, with caching. Returns None if offline."""
        # Check cache first
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached.checked_at) < _CACHE_TTL_SECONDS:
            return cached.exists

        if self.offline:
            return cached.exists if cached else None

        try:
            from huggingface_hub import repo_info

            repo_info(repo_id, repo_type=repo_type if repo_type == "dataset" else None)
            exists = True
        except Exception:
            exists = False

        self._cache[cache_key] = _CacheEntry(exists=exists, checked_at=time.time())
        self._save_cache()
        return exists

    def _load_cache(self) -> None:
        if not os.path.exists(_CACHE_FILE):
            return
        try:
            with open(_CACHE_FILE) as f:
                raw = json.load(f)
            for key, val in raw.items():
                self._cache[key] = _CacheEntry(
                    exists=val["exists"], checked_at=val["checked_at"]
                )
        except Exception:
            pass

    def _save_cache(self) -> None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        raw = {
            k: {"exists": v.exists, "checked_at": v.checked_at}
            for k, v in self._cache.items()
        }
        try:
            with open(_CACHE_FILE, "w") as f:
                json.dump(raw, f, indent=2)
        except Exception:
            pass
