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

"""Config index for fast metadata lookup and filtering."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from oumi.cli.alias import AliasType, _ALIASES
from oumi.core.configs.metadata import (
    ConfigMetadata,
    ConfigType,
    FinetuningType,
    MetadataExtractor,
    TrainingMethod,
    estimate_vram_from_config,
    get_recommended_gpus,
)
from oumi.utils.logging import logger

# Version of the index format
INDEX_VERSION = "1.0.0"

# Default index file location
DEFAULT_INDEX_PATH = Path(__file__).parent.parent.parent.parent / "configs" / ".oumi" / "config_index.json"


def _alias_type_to_config_type(alias_type: AliasType) -> ConfigType:
    """Map AliasType to ConfigType."""
    mapping = {
        AliasType.TRAIN: ConfigType.TRAINING,
        AliasType.EVAL: ConfigType.EVALUATION,
        AliasType.INFER: ConfigType.INFERENCE,
        AliasType.JOB: ConfigType.JOB,
        AliasType.JUDGE: ConfigType.JUDGE,
        AliasType.QUANTIZE: ConfigType.QUANTIZE,
    }
    return mapping.get(alias_type, ConfigType.TRAINING)


def load_config_index(index_path: Optional[Path] = None) -> dict:
    """Load the config index from disk.

    Args:
        index_path: Path to the index file. Uses default if not provided.

    Returns:
        Dictionary with index data, or empty dict with version if file doesn't exist.
    """
    path = index_path or DEFAULT_INDEX_PATH

    if not path.exists():
        return {"version": INDEX_VERSION, "generated_at": None, "configs": {}}

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load config index: {e}")
        return {"version": INDEX_VERSION, "generated_at": None, "configs": {}}


def save_config_index(index: dict, index_path: Optional[Path] = None) -> None:
    """Save the config index to disk.

    Args:
        index: The index dictionary to save.
        index_path: Path to save the index. Uses default if not provided.
    """
    path = index_path or DEFAULT_INDEX_PATH

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def generate_config_index_from_aliases(
    verbose: bool = False,
) -> dict:
    """Generate config index from all aliased configs.

    This function iterates through all aliases and extracts metadata
    using lightweight path-based extraction (doesn't load full configs).

    Args:
        verbose: Whether to print progress messages.

    Returns:
        Dictionary with config index data.
    """
    index = {
        "version": INDEX_VERSION,
        "generated_at": datetime.now().isoformat(),
        "configs": {},
    }

    for alias, paths in _ALIASES.items():
        for alias_type, config_path in paths.items():
            try:
                # Use path-based extraction (lightweight)
                config_type = _alias_type_to_config_type(alias_type)
                metadata = MetadataExtractor.extract_from_path(
                    config_path, config_type_hint=config_type
                )

                # Create unique key for this alias+type combination
                if len(paths) == 1:
                    key = alias
                else:
                    key = f"{alias}:{alias_type.value}"

                index["configs"][key] = {
                    "alias": alias,
                    "alias_type": alias_type.value,
                    "path": config_path,
                    **metadata.to_dict(),
                }

                if verbose:
                    logger.info(f"Indexed: {key}")

            except Exception as e:
                logger.warning(f"Failed to extract metadata for {alias}: {e}")
                # Still add basic entry
                index["configs"][alias] = {
                    "alias": alias,
                    "alias_type": alias_type.value,
                    "path": config_path,
                    "config_type": config_type.value,
                    "error": str(e),
                }

    return index


def filter_configs(
    index: dict,
    config_type: Optional[ConfigType] = None,
    model_family: Optional[str] = None,
    training_method: Optional[TrainingMethod] = None,
    finetuning_type: Optional[FinetuningType] = None,
    max_vram_gb: Optional[float] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Filter configs by metadata criteria.

    Args:
        index: The config index dictionary.
        config_type: Filter by config type.
        model_family: Filter by model family (case-insensitive).
        training_method: Filter by training method.
        finetuning_type: Filter by finetuning type.
        max_vram_gb: Filter by maximum VRAM requirement.
        tags: Filter by tags (any match).

    Returns:
        Filtered dictionary of matching configs.
    """
    configs = index.get("configs", {})
    result = {}

    for key, config in configs.items():
        # Skip configs with errors
        if "error" in config:
            continue

        # Filter by config type
        if config_type and config.get("config_type") != config_type.value:
            continue

        # Filter by model family (case-insensitive)
        if model_family:
            cfg_family = config.get("model_family")
            if not cfg_family or model_family.lower() not in cfg_family.lower():
                continue

        # Filter by training method
        if training_method and config.get("training_method") != training_method.value:
            continue

        # Filter by finetuning type
        if finetuning_type and config.get("finetuning_type") != finetuning_type.value:
            continue

        # Filter by max VRAM
        if max_vram_gb is not None:
            cfg_vram = config.get("min_vram_gb")
            if cfg_vram is not None and cfg_vram > max_vram_gb:
                continue

        # Filter by tags (any match)
        if tags:
            cfg_tags = config.get("tags", [])
            if not any(tag in cfg_tags for tag in tags):
                continue

        result[key] = config

    return result


def parse_filter_expression(filter_str: str) -> dict:
    """Parse a filter expression string into filter criteria.

    Supports expressions like:
    - "vram<24" or "vram<=24"
    - "family=llama"
    - "type=qlora"
    - "method=sft"
    - "tag=vision"
    - "config=training"

    Multiple filters can be combined with commas:
    - "vram<24,family=llama,type=qlora"

    Args:
        filter_str: The filter expression string.

    Returns:
        Dictionary of filter criteria suitable for filter_configs().
    """
    criteria = {}

    for part in filter_str.split(","):
        part = part.strip()
        if not part:
            continue

        # Parse operators
        if "<=" in part:
            key, value = part.split("<=", 1)
            op = "<="
        elif "<" in part:
            key, value = part.split("<", 1)
            op = "<"
        elif ">=" in part:
            key, value = part.split(">=", 1)
            op = ">="
        elif ">" in part:
            key, value = part.split(">", 1)
            op = ">"
        elif "=" in part:
            key, value = part.split("=", 1)
            op = "="
        else:
            continue

        key = key.strip().lower()
        value = value.strip()

        # Map filter keys to criteria
        if key == "vram":
            try:
                vram_val = float(value)
                if op in ("<", "<="):
                    criteria["max_vram_gb"] = vram_val
            except ValueError:
                pass

        elif key == "family":
            criteria["model_family"] = value

        elif key == "type":
            try:
                criteria["finetuning_type"] = FinetuningType(value.lower())
            except ValueError:
                pass

        elif key == "method":
            try:
                criteria["training_method"] = TrainingMethod(value.lower())
            except ValueError:
                pass

        elif key == "tag":
            criteria.setdefault("tags", []).append(value)

        elif key == "config":
            try:
                criteria["config_type"] = ConfigType(value.lower())
            except ValueError:
                pass

    return criteria


def get_config_metadata(alias: str, index: Optional[dict] = None) -> Optional[ConfigMetadata]:
    """Get metadata for a specific config alias.

    Args:
        alias: The config alias.
        index: Optional pre-loaded index. Will load from disk if not provided.

    Returns:
        ConfigMetadata for the alias, or None if not found.
    """
    if index is None:
        index = load_config_index()

    configs = index.get("configs", {})

    # Try exact match first
    if alias in configs:
        return ConfigMetadata.from_dict(configs[alias])

    # Try with alias types
    for alias_type in AliasType:
        key = f"{alias}:{alias_type.value}"
        if key in configs:
            return ConfigMetadata.from_dict(configs[key])

    return None
