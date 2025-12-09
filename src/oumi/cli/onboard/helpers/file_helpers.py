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

"""File detection and analysis helper functions for the onboard wizard."""

import json
from pathlib import Path

from ..dataclasses import SUPPORTED_EXTENSIONS


def detect_files_in_directory(dir_path: Path) -> list[dict]:
    """Detect supported files in a directory.

    Args:
        dir_path: Path to the directory.

    Returns:
        List of file info dicts with path, name, extension.
    """
    files = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append({
                "path": file_path,
                "name": file_path.name,
                "extension": file_path.suffix.lower(),
            })
    return sorted(files, key=lambda x: x["name"])


def analyze_file_purposes(files: list[dict], analyzer, llm_analyzer) -> list[dict]:
    """Analyze file purposes using LLM.

    Args:
        files: List of file info dicts.
        analyzer: DataAnalyzer instance.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Updated files with suggested_purpose, suggested_role, role_reason.
    """
    for f in files:
        if f.get("schema") and f["schema"].columns:
            cols = [c.name for c in f["schema"].columns[:10]]
            sample = ""
            if f["schema"].sample_rows:
                sample = json.dumps(f["schema"].sample_rows[0], indent=2)[:500]

            prompt = f"""Analyze this file and determine its purpose for ML training.

File: {f['name']} ({f['extension']})
Columns: {', '.join(cols)}
Sample data:
{sample}

Return JSON:
{{
    "purpose": "Brief description of what this file contains",
    "role": "primary|reference|rules|examples",
    "reason": "Why this role fits"
}}

Return ONLY the JSON object."""

            try:
                result = llm_analyzer._invoke_json(prompt)
                f["suggested_purpose"] = result.get("purpose", "")
                f["suggested_role"] = result.get("role", "primary")
                f["role_reason"] = result.get("reason", "")
            except Exception:
                f["suggested_purpose"] = "Data file"
                f["suggested_role"] = "primary"
                f["role_reason"] = "Default assignment"
        else:
            f["suggested_purpose"] = "Document"
            f["suggested_role"] = "reference"
            f["role_reason"] = "Non-tabular file"

    return files
