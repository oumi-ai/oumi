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

"""Configuration service for parsing and managing Oumi YAML configs."""

import copy
import logging
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import yaml

from oumi.core.configs import (
    AnalyzeConfig,
    AsyncEvaluationConfig,
    EvaluationConfig,
    InferenceConfig,
    JobConfig,
    JudgeConfig,
    QuantizationConfig,
    SynthesisConfig,
    TrainingConfig,
    TuningConfig,
)
from oumi.mcp.constants import (
    API_PROVIDERS_DIR,
    COMMENT_PREFIXES_TO_SKIP,
    CONFIGS_CACHE_SIZE,
    DATA_SPLITS,
    MODEL_FAMILIES_DIR,
    MODEL_KEYS,
    TRAIN_YAML,
    TRAINING_KEYS,
    YAML_CACHE_SIZE,
    PeftType,
    TaskType,
)
from oumi.mcp.models import (
    CategoriesResponse,
    ConfigMetadata,
    KeySettings,
)

logger = logging.getLogger(__name__)

TASK_MAPPING = {
    "analyze": AnalyzeConfig,
    "async_evaluation": AsyncEvaluationConfig,
    "evaluation": EvaluationConfig,
    "inference": InferenceConfig,
    "job": JobConfig,
    "judge": JudgeConfig,
    "quantization": QuantizationConfig,
    "synthesis": SynthesisConfig,
    "training": TrainingConfig,
    "tuning": TuningConfig,
}


def get_bundled_configs_dir() -> Path:
    """Get the path to the bundled configs shipped with the package.

    Returns:
        Path to the configs directory inside the installed package.
    """
    return Path(__file__).parent / "configs"


def get_cache_dir() -> Path:
    """Get the path to the user-level config cache directory.

    Returns:
        Path to ~/.cache/oumi-mcp/configs.
    """
    return Path.home() / ".cache" / "oumi-mcp" / "configs"


def get_configs_dir() -> Path:
    """Get the path to the configs directory.

    Resolution order (first that exists and contains YAML files wins):
    1. OUMI_MCP_CONFIGS_DIR environment variable (explicit override)
    2. ~/.cache/oumi-mcp/configs (synced from GitHub, fresher)
    3. Bundled configs shipped with the package (always available fallback)

    Returns:
        Path to the configs directory.
    """
    env_dir = os.environ.get("OUMI_MCP_CONFIGS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and any(p.rglob("*.yaml")):
            return p

    cache = get_cache_dir()
    if cache.is_dir() and any(cache.rglob("*.yaml")):
        return cache

    return get_bundled_configs_dir()


@lru_cache(maxsize=YAML_CACHE_SIZE)
def _parse_yaml_cached(path: str) -> dict[str, Any]:
    """Parse a YAML file (cached, internal). Callers should use parse_yaml()."""
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to parse YAML {path}: {e}")
        return {}


def parse_yaml(path: str) -> dict[str, Any]:
    """Parse a YAML file, returning empty dict on error.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed YAML as dict, or empty dict if parsing fails.
    """
    return copy.deepcopy(_parse_yaml_cached(path))


def extract_header_comment(path: Path) -> str:
    """Extract description from YAML header comments.

    Reads the first lines of the file that start with # and extracts
    the description, skipping lines that start with common prefixes
    like "Usage:", "See Also:", etc.

    Args:
        path: Path to the YAML file.

    Returns:
        Extracted description from header comments.
    """
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
        comments: list[str] = []
        for line in lines:
            if line.startswith("#"):
                clean = line.lstrip("#").strip()
                if clean and not any(
                    clean.startswith(prefix) for prefix in COMMENT_PREFIXES_TO_SKIP
                ):
                    comments.append(clean)
            elif line.strip():
                break
        return " ".join(comments[:2])
    except Exception as e:
        logger.warning(f"Failed to extract header from {path}: {e}")
        return ""


def infer_task_type(trainer_type: str, path: str) -> TaskType:
    """Infer task type from trainer_type or file path.

    Checks both the trainer_type field and the path for task indicators
    like grpo, dpo, sft, eval, infer, etc.

    Args:
        trainer_type: Value from config.training.trainer_type.
        path: Relative path to the config file.

    Returns:
        Inferred task type string.
    """
    t = trainer_type.lower()
    p = path.lower()

    task_mapping: list[tuple[list[str], TaskType]] = [
        (["grpo"], "grpo"),
        (["dpo"], "dpo"),
        (["kto"], "kto"),
        (["sft"], "sft"),
    ]

    for keywords, task in task_mapping:
        if any(kw in t or kw in p for kw in keywords):
            return task

    if "eval" in p:
        return "evaluation"
    if "infer" in p:
        return "inference"
    if "pretrain" in p:
        return "pretraining"
    if "synth" in p:
        return "synthesis"
    if "quant" in p:
        return "quantization"

    return ""


def extract_datasets(config: dict[str, Any]) -> list[str]:
    """Extract dataset names from config data section.

    Args:
        config: Parsed YAML config dict.

    Returns:
        List of dataset names used in train/validation/test splits.
    """
    datasets: list[str] = []
    data = config.get("data", {})

    for split in DATA_SPLITS:
        split_data = data.get(split, {})
        if isinstance(split_data, dict):
            for ds in split_data.get("datasets", []):
                if isinstance(ds, dict) and "dataset_name" in ds:
                    datasets.append(ds["dataset_name"])

    return datasets


def determine_peft_type(config: dict[str, Any], path: str) -> PeftType | None:
    """Determine PEFT type from config and path.

    Args:
        config: Parsed YAML config dict.
        path: Relative path to config file.

    Returns:
        "qlora", "lora", or None if not using PEFT.
    """
    peft = config.get("peft", {})
    if peft.get("lora_r"):
        if peft.get("q_lora") or "qlora" in path.lower():
            return "qlora"
        return "lora"
    return None


def build_metadata(config_path: Path, configs_dir: Path) -> ConfigMetadata:
    """Build metadata dict for a config file.

    Args:
        config_path: Absolute path to the YAML config file.
        configs_dir: Root configs directory for calculating relative path.

    Returns:
        ConfigMetadata with extracted information.
    """
    rel_path = str(config_path.relative_to(configs_dir))
    config = parse_yaml(str(config_path))

    model = config.get("model", {})
    training = config.get("training", {})

    return {
        "path": rel_path,
        "description": extract_header_comment(config_path),
        "model_name": model.get("model_name", "") or "",
        "task_type": infer_task_type(training.get("trainer_type", ""), rel_path),
        "datasets": extract_datasets(config) or [],
        "reward_functions": training.get("reward_functions") or [],
        "peft_type": determine_peft_type(config, rel_path) or "",
    }


@lru_cache(maxsize=CONFIGS_CACHE_SIZE)
def _get_all_configs_cached() -> list[ConfigMetadata]:
    """Get metadata for all configs (cached, internal)."""
    configs_dir = get_configs_dir()
    configs: list[ConfigMetadata] = []

    for path in configs_dir.rglob("*.yaml"):
        configs.append(build_metadata(path, configs_dir))

    return configs


def get_all_configs() -> list[ConfigMetadata]:
    """Get metadata for all configs (cached).

    Returns:
        List of ConfigMetadata for all YAML files in configs directory.
    """
    return copy.deepcopy(_get_all_configs_cached())


def clear_config_caches() -> None:
    """Invalidate all config LRU caches after config_sync replaces the cache dir."""
    _get_all_configs_cached.cache_clear()
    _parse_yaml_cached.cache_clear()


def extract_key_settings(config: dict[str, Any]) -> KeySettings:
    """Extract key training settings from config.

    Args:
        config: Parsed YAML config dict.

    Returns:
        KeySettings with important hyperparameters.
    """
    training = config.get("training", {})
    model_cfg = config.get("model", {})

    key_settings: KeySettings = {}

    for key in TRAINING_KEYS:
        if training.get(key) is not None:
            key_settings[key] = training[key]  # type: ignore[literal-required]

    for key in MODEL_KEYS:
        if model_cfg.get(key) is not None:
            if key == "torch_dtype_str":
                key_settings["torch_dtype"] = model_cfg[key]
            else:
                key_settings[key] = model_cfg[key]  # type: ignore[literal-required]

    return key_settings


def find_config_match(
    path_query: str, configs: list[ConfigMetadata]
) -> ConfigMetadata | None:
    """Find the best matching config for a path query.

    Args:
        path_query: Path query string (exact or partial match).
        configs: List of all configs to search.

    Returns:
        Best matching ConfigMetadata, or None if no match found.
    """
    path_lower = path_query.lower()
    candidates: list[ConfigMetadata] = []

    for cfg in configs:
        if cfg["path"] == path_query:
            return cfg
        if path_lower in cfg["path"].lower():
            candidates.append(cfg)

    if not candidates:
        return None
    for c in candidates:
        if c["path"].endswith(f"/{TRAIN_YAML}"):
            return c
    for c in candidates:
        if TRAIN_YAML in c["path"]:
            return c
    return candidates[0]


def search_configs(
    configs: list[ConfigMetadata],
    query: list[str] | None = None,
    content_match: list[str] | None = None,
    limit: int = 20,
) -> list[ConfigMetadata]:
    """Search for configs matching the given filters.

    Args:
        configs: List of all configs to search.
        query: Terms matched against config paths (AND logic, case-insensitive).
        content_match: Terms matched against YAML file content (AND logic,
            case-insensitive).
        limit: Maximum number of results to return.

    Returns:
        List of matching ConfigMetadata, sorted by relevance.
    """
    filters: list[str] = [t.lower().strip() for t in (query or []) if t.strip()]

    keywords: list[str] = [
        k.lower().strip() for k in (content_match or []) if k.strip()
    ]

    if not filters and not keywords:
        return sorted(configs, key=lambda x: x["path"])[:limit]

    matches: list[ConfigMetadata] = []
    configs_dir = get_configs_dir() if keywords else None
    for cfg in configs:
        path_lower = cfg["path"].lower()
        if not all(f in path_lower for f in filters):
            continue
        if keywords and configs_dir is not None:
            config_path = configs_dir / cfg["path"]
            try:
                content = config_path.read_text(encoding="utf-8").lower()
            except Exception:
                continue
            if not all(kw in content for kw in keywords):
                continue
        matches.append(cfg)

    matches.sort(key=lambda x: (-len(x["datasets"]), x["path"]))
    return matches[:limit]


def get_categories(
    configs_dir: Path,
    configs_count: int,
    *,
    oumi_version: str = "",
    configs_source: str = "",
    version_warning: str = "",
) -> CategoriesResponse:
    """List available config categories and model families.

    Args:
        configs_dir: Root configs directory.
        configs_count: Total number of configs.
        oumi_version: Installed oumi library version (populated by caller).
        configs_source: Where configs are loaded from (populated by caller).
        version_warning: Non-empty when configs may be mismatched (populated by caller).

    Returns:
        CategoriesResponse with all available categories.
    """
    categories: list[str] = []
    model_families: list[str] = []
    api_providers: list[str] = []

    for item in sorted(configs_dir.iterdir()):
        if item.is_dir():
            categories.append(item.name)
            if item.name == MODEL_FAMILIES_DIR:
                model_families = sorted(
                    d.name
                    for d in item.iterdir()
                    if d.is_dir() and d.name != "README.md"
                )
            elif item.name == API_PROVIDERS_DIR:
                api_providers = sorted(d.name for d in item.iterdir() if d.is_dir())

    return {
        "categories": categories,
        "model_families": model_families,
        "api_providers": api_providers,
        "total_configs": configs_count,
        "oumi_version": oumi_version,
        "configs_source": configs_source,
        "version_warning": version_warning,
    }


def get_package_version(package_name: str) -> str | None:
    """Return the installed version string for *package_name*, or None."""
    try:
        return _pkg_version(package_name)
    except PackageNotFoundError:
        return None


def load_yaml_strict(config_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load YAML config and return a user-facing error when invalid."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        return None, f"Invalid YAML config: {exc}"
    if cfg is None:
        return None, "Config file is empty."
    if not isinstance(cfg, dict):
        return None, "Config root must be a mapping/object."
    return cfg, None


def resolve_path(raw: str, client_cwd: Path) -> Path:
    """Resolve a path string against the client's working directory.

    Absolute paths (after ``~`` expansion) are returned as-is.
    Relative paths are resolved against *client_cwd*.
    """
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (client_cwd / expanded).resolve()


def resolve_config_path(config: str, client_cwd: str) -> tuple[Path, str | None]:
    """Resolve and validate a config file path against the client's CWD.

    Requires *client_cwd* to be absolute. Relative *config* paths are
    resolved against it. Returns ``(resolved_path, None)`` on success,
    or ``(Path(), error_message)`` on failure.
    """
    cwd = Path(client_cwd).expanduser()
    if not cwd.is_absolute():
        return Path(), (
            f"client_cwd must be absolute, got: '{client_cwd}'. "
            "Provide the full path to your project directory."
        )
    p = resolve_path(config, cwd)
    if not p.exists():
        return Path(), f"Config file not found: {p}"
    if not p.is_file():
        return Path(), f"Config path is not a file: {p}"
    return p, None
