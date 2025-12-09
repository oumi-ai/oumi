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

"""Input detection helper functions for the onboard wizard."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataclasses import WizardState

from ..dataclasses import INPUT_FORMATS


def detect_input_source(state: "WizardState", llm_analyzer) -> dict:
    """Use AI to detect the best input source for the task.

    Analyzes columns and sample data to find meaningful input columns,
    excluding things like UUIDs, IDs, timestamps that don't make sense as inputs.

    Args:
        state: Wizard state with task and schema info.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with: source_column, format, samples, reasoning
    """
    if not state.primary_schema or not state.primary_schema.columns:
        return {
            "source_column": "",
            "format": "single_turn",
            "samples": [],
            "reasoning": "No schema available",
        }

    cols_info = []
    for col in state.primary_schema.columns[:15]:
        col_info = {"name": col.name, "dtype": col.dtype}
        if col.avg_length:
            col_info["avg_length"] = col.avg_length
        if state.primary_schema.sample_rows:
            sample_vals = [
                str(row.get(col.name, ""))[:100]
                for row in state.primary_schema.sample_rows[:3]
                if row.get(col.name)
            ]
            col_info["sample_values"] = sample_vals
        cols_info.append(col_info)

    prompt = f"""Analyze this data to determine the best INPUT column(s) for a machine learning task.

Task: {state.task.description}

Available columns:
{json.dumps(cols_info, indent=2)}

Select the column(s) that should be used as INPUT to the model. Consider:
1. The task description - what data does the model need to perform this task?
2. Exclude meaningless columns like: UUIDs, IDs, timestamps, row numbers, internal identifiers
3. Prefer columns with actual text content, questions, or structured data relevant to the task
4. If multiple columns are needed together (e.g., "title" + "body"), suggest combining them

Return JSON:
{{
    "source_column": "column_name",
    "format": "single_turn|multi_turn|document|structured|instruction",
    "reasoning": "Brief explanation of why this column makes sense for the task"
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        source_col = result.get("source_column", "")
        fmt = result.get("format", "single_turn")
        reasoning = result.get("reasoning", "")

        samples = []
        if source_col and state.primary_schema.sample_rows:
            cols_to_use = [c.strip() for c in source_col.split(",")]
            for row in state.primary_schema.sample_rows[:3]:
                if len(cols_to_use) == 1:
                    val = row.get(cols_to_use[0], "")
                    if val:
                        samples.append(str(val)[:200])
                else:
                    parts = [str(row.get(c, "")) for c in cols_to_use if row.get(c)]
                    if parts:
                        samples.append(" | ".join(parts)[:200])

        return {
            "source_column": source_col,
            "format": fmt if fmt in INPUT_FORMATS else "single_turn",
            "samples": samples,
            "reasoning": reasoning,
        }
    except Exception:
        return fallback_input_detection(state)


def fallback_input_detection(state: "WizardState") -> dict:
    """Fallback input detection using simple heuristics."""
    if not state.primary_schema or not state.primary_schema.columns:
        return {
            "source_column": "",
            "format": "single_turn",
            "samples": [],
            "reasoning": "No schema available",
        }

    skip_patterns = ["id", "uuid", "_id", "created", "updated", "timestamp", "index"]

    input_col = None
    for col in state.primary_schema.columns:
        name_lower = col.name.lower()
        if any(pat in name_lower for pat in skip_patterns):
            continue
        if any(kw in name_lower for kw in ["question", "input", "query", "prompt", "text", "content", "message"]):
            input_col = col.name
            break

    if not input_col:
        for col in state.primary_schema.columns:
            name_lower = col.name.lower()
            if not any(pat in name_lower for pat in skip_patterns):
                input_col = col.name
                break

    if not input_col and state.primary_schema.columns:
        input_col = state.primary_schema.columns[0].name

    samples = []
    if input_col and state.primary_schema.sample_rows:
        samples = [
            str(row.get(input_col, ""))[:200]
            for row in state.primary_schema.sample_rows[:3]
            if row.get(input_col)
        ]

    return {
        "source_column": input_col or "",
        "format": "single_turn",
        "samples": samples,
        "reasoning": "Selected using keyword heuristics",
    }
