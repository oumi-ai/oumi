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

"""Parser for @meta comments in YAML config files."""

import re
from pathlib import Path
from typing import Union

# Pattern to match @meta comments: # @meta key: value
META_PATTERN = re.compile(r"^#\s*@meta\s+(\w+):\s*(.+)$")


def parse_metadata_comments(config_path: Union[str, Path]) -> dict[str, str]:
    """Parse @meta comments from the top of a config file.

    Reads the config file and extracts metadata from lines matching
    the pattern `# @meta key: value`. Stops parsing when encountering
    a non-comment, non-empty line.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary mapping metadata keys to their string values.

    Example:
        Given a config file starting with:
        ```yaml
        # @meta training_method: sft
        # @meta finetuning_type: lora
        # @meta min_vram_gb: 20
        # @meta tags: beginner-friendly, single-gpu
        # @meta description: LoRA fine-tuning for Llama 3.1 8B

        model:
          model_name: "meta-llama/Llama-3.1-8B-Instruct"
        ```

        Returns:
        ```python
        {
            "training_method": "sft",
            "finetuning_type": "lora",
            "min_vram_gb": "20",
            "tags": "beginner-friendly, single-gpu",
            "description": "LoRA fine-tuning for Llama 3.1 8B"
        }
        ```
    """
    metadata: dict[str, str] = {}
    config_path = Path(config_path)

    if not config_path.exists():
        return metadata

    with open(config_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Skip empty lines at the top
            if not stripped:
                continue

            # Stop at first non-comment line
            if not stripped.startswith("#"):
                break

            # Try to match @meta pattern
            match = META_PATTERN.match(stripped)
            if match:
                key, value = match.groups()
                metadata[key] = value.strip()

    return metadata


def parse_tags(tags_str: str) -> list[str]:
    """Parse a comma-separated tags string into a list.

    Args:
        tags_str: Comma-separated string of tags.

    Returns:
        List of individual tags, stripped of whitespace.

    Example:
        >>> parse_tags("beginner-friendly, single-gpu, vision")
        ["beginner-friendly", "single-gpu", "vision"]
    """
    if not tags_str:
        return []
    return [tag.strip() for tag in tags_str.split(",") if tag.strip()]
