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


def suggest_quality_criteria(state: "WizardState", llm_analyzer) -> list[str]:
    """Suggest quality criteria based on task.

    Args:
        state: Wizard state with task description.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        List of suggested criteria strings.
    """
    prompt = f"""For this task, suggest 3-5 quality criteria for evaluating responses.

Task: {state.task.description}

Return a JSON array of short criteria, e.g.:
["accurate", "helpful", "clear", "complete"]

Return ONLY the JSON array."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        if isinstance(result, list):
            return [str(c) for c in result[:5]]
    except Exception:
        pass
    return ["accurate", "helpful", "clear"]
