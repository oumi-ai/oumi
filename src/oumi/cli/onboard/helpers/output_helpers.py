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


def suggest_quality_criteria(state: "WizardState", llm_analyzer) -> tuple[list[str], dict[str, str]]:
    """Suggest quality criteria based on task.

    Args:
        state: Wizard state with task description.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Tuple of (criteria_names, criteria_descriptions) where descriptions maps name -> description.
    """
    # Build context for better criteria suggestions
    context = {
        "task_description": state.task.description,
        "task_type": state.task.task_type if state.task and state.task.task_type else "generation",
    }

    # Add domain context if available
    if state.domain_analysis:
        context["domain"] = state.domain_analysis.domain
        if state.domain_analysis.quality_signals:
            context["quality_signals"] = ", ".join(state.domain_analysis.quality_signals[:5])

    # Add system prompt for additional context
    if state.task and state.task.system_prompt:
        context["system_prompt"] = state.task.system_prompt[:200]  # First 200 chars for context

    prompt = load_prompt("suggest_quality_criteria", **context)

    try:
        result = llm_analyzer._invoke_json(prompt)
        if isinstance(result, list) and len(result) > 0:
            # Parse new format with name + description
            criteria_names = []
            criteria_descriptions = {}

            for item in result[:5]:
                if isinstance(item, dict) and "name" in item:
                    name = str(item["name"])
                    criteria_names.append(name)
                    if "description" in item:
                        criteria_descriptions[name] = str(item["description"])
                elif isinstance(item, str):
                    # Backward compatibility: accept plain strings
                    criteria_names.append(str(item))

            if criteria_names:
                return criteria_names, criteria_descriptions
    except Exception:
        pass

    # Fallback: return task-specific defaults with descriptions
    task_type = context.get("task_type", "generation")
    fallback_data = {
        "extraction": {
            "factually_accurate": "Checks that extracted information matches the source without invention or distortion.",
            "no_hallucination": "Verifies the model does not extract entities or facts that don't exist in the input.",
            "complete": "Ensures all required fields are extracted without omissions.",
        },
        "classification": {
            "correct_category": "Verifies the assigned category/label matches the input content.",
            "well_reasoned": "Checks that the classification reasoning is logical and follows from the input.",
            "confident": "Ensures the model provides clear classification without hedging when it should be certain.",
        },
        "qa": {
            "addresses_question": "Verifies the response directly answers what was asked.",
            "factually_correct": "Checks that all factual claims in the answer are accurate and verifiable.",
            "no_fabrication": "Ensures the model doesn't invent information not present in the context.",
        },
        "transformation": {
            "preserves_meaning": "Checks that the core meaning and intent are maintained through transformation.",
            "follows_format": "Verifies output adheres to the required format/structure.",
            "complete": "Ensures no important information is lost during transformation.",
        },
        "generation": {
            "on_topic": "Checks that generated content stays relevant to the task and doesn't drift.",
            "factually_sound": "Verifies generated claims are plausible and not obviously false.",
            "appropriate_tone": "Ensures the tone matches the context and intended audience.",
        },
    }

    fallback_criteria = fallback_data.get(
        task_type,
        {
            "factually_accurate": "Checks that information is correct without fabrication.",
            "on_topic": "Verifies content stays relevant to the task.",
            "appropriate_format": "Ensures output follows expected structure and conventions.",
        }
    )

    return list(fallback_criteria.keys()), fallback_criteria


def merge_criteria(
    extracted: list[str],
    generated: list[str],
    extracted_descriptions: dict[str, str] | None = None,
    generated_descriptions: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Merge extracted and generated criteria, deduplicating case-insensitively.

    Args:
        extracted: List of extracted criterion names.
        generated: List of generated criterion names.
        extracted_descriptions: Optional descriptions for extracted criteria.
        generated_descriptions: Optional descriptions for generated criteria.

    Returns:
        Tuple of (merged_names, sources, descriptions) where:
        - merged_names: Deduplicated list of criterion names
        - sources: Maps name -> "extracted" or "generated"
        - descriptions: Maps name -> description text
    """
    merged: list[str] = []
    sources: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    seen_lower = set()

    extracted_descriptions = extracted_descriptions or {}
    generated_descriptions = generated_descriptions or {}

    for criterion in extracted:
        crit_lower = criterion.lower().strip()
        if crit_lower in seen_lower:
            continue
        merged.append(criterion)
        sources[criterion] = "extracted"
        if criterion in extracted_descriptions:
            descriptions[criterion] = extracted_descriptions[criterion]
        seen_lower.add(crit_lower)

    for criterion in generated:
        crit_lower = criterion.lower().strip()
        if crit_lower in seen_lower:
            continue
        merged.append(criterion)
        sources[criterion] = "generated"
        if criterion in generated_descriptions:
            descriptions[criterion] = generated_descriptions[criterion]
        seen_lower.add(crit_lower)

    return merged, sources, descriptions
