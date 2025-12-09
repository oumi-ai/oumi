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

"""Output quality helper functions for the onboard wizard."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataclasses import WizardState

from ..prompts import load_prompt


def suggest_quality_criteria(state: "WizardState", llm_analyzer) -> list[str]:
    """Suggest quality criteria based on task.

    Args:
        state: Wizard state with task description.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        List of suggested criteria strings.
    """
    prompt = load_prompt(
        "suggest_quality_criteria",
        task_description=state.task.description,
    )

    try:
        result = llm_analyzer._invoke_json(prompt)
        if isinstance(result, list):
            return [str(c) for c in result[:5]]
    except Exception:
        pass
    return ["accurate", "helpful", "clear"]


def merge_criteria(extracted: list[str], generated: list[str]) -> tuple[list[str], dict]:
    """Merge extracted and generated criteria, deduplicating case-insensitively."""
    merged: list[str] = []
    sources: dict[str, str] = {}
    seen_lower = set()

    for criterion in extracted:
        crit_lower = criterion.lower().strip()
        if crit_lower in seen_lower:
            continue
        merged.append(criterion)
        sources[criterion] = "extracted"
        seen_lower.add(crit_lower)

    for criterion in generated:
        crit_lower = criterion.lower().strip()
        if crit_lower in seen_lower:
            continue
        merged.append(criterion)
        sources[criterion] = "generated"
        seen_lower.add(crit_lower)

    return merged, sources
