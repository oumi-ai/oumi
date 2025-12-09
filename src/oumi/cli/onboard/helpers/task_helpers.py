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

"""Task analysis helper functions for the onboard wizard."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataclasses import WizardState

from ..dataclasses import TASK_TYPES


def derive_task_name(description: str) -> str:
    """Derive a short task name from a description.

    Args:
        description: Full task description.

    Returns:
        Short task name (max ~50 chars).
    """
    if not description:
        return "Custom Task"

    desc = description.strip()

    for end_char in [".", "!", "?"]:
        idx = desc.find(end_char)
        if 0 < idx < 80:
            return desc[: idx + 1]

    if len(desc) <= 50:
        return desc

    truncated = desc[:50]
    last_space = truncated.rfind(" ")
    if last_space > 20:
        return truncated[:last_space] + "..."
    return truncated + "..."


def infer_task_type(description: str, system_prompt: str, llm_analyzer) -> tuple[str, str]:
    """Infer the task type from description and system prompt.

    Args:
        description: Task description.
        system_prompt: Generated system prompt.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Tuple of (task_type, example_output_format).
    """
    task_types_str = "\n".join([
        f"- {key}: {info['description']}"
        for key, info in TASK_TYPES.items()
    ])

    prompt = f"""Analyze this task and determine its type.

Task description: {description}

System prompt: {system_prompt}

Available task types:
{task_types_str}

Based on the task, determine:
1. Which task type best fits
2. What format the OUTPUT should be in (e.g., JSON schema, label format, text format)

Return JSON:
{{
    "task_type": "<one of: extraction, classification, generation, transformation, qa>",
    "output_format": "<brief description of expected output format>"
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        task_type = result.get("task_type", "generation")
        output_format = result.get("output_format", "")

        if task_type not in TASK_TYPES:
            task_type = "generation"

        return task_type, output_format
    except Exception:
        return "generation", ""


def analyze_task_from_files(files: list[dict], llm_analyzer) -> dict:
    """Analyze files to suggest what task the user is trying to accomplish.

    Args:
        files: List of file info dicts with analysis.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with task analysis.
    """
    file_summaries = []
    for f in files:
        summary = f"- {f['name']} ({f['extension']})"
        if f.get("suggested_purpose"):
            summary += f": {f['suggested_purpose']}"
        if f.get("schema") and f["schema"].columns:
            cols = [c.name for c in f["schema"].columns[:8]]
            summary += f" [columns: {', '.join(cols)}]"
        if f.get("schema") and f["schema"].sample_rows:
            sample = json.dumps(f["schema"].sample_rows[0], indent=2)[:300]
            summary += f"\n    Sample: {sample}"
        file_summaries.append(summary)

    prompt = f"""Analyze these files to understand what ML task the user is trying to build.

FILES:
{chr(10).join(file_summaries)}

Based on the file contents, column names, and sample data:

1. What task is the user trying to accomplish?
2. What would typical inputs look like?
3. What would ideal outputs look like?

Return JSON:
{{
    "primary_task": "Short descriptive name (e.g., 'Q&A System', 'Customer Support Bot')",
    "task_description": "Clear description of what the model will do (2-3 sentences)",
    "example_input": "A realistic example of what users will send",
    "example_output": "What the model should respond with",
    "reasoning": "Why you concluded this based on the file contents"
}}

Return ONLY the JSON object."""

    try:
        return llm_analyzer._invoke_json(prompt)
    except Exception as e:
        return {
            "primary_task": "Custom Task",
            "task_description": "Unable to determine - please describe your task",
            "example_input": "",
            "example_output": "",
            "reasoning": f"Analysis failed: {str(e)[:100]}",
        }


def generate_system_prompt(state: "WizardState", llm_analyzer) -> str:
    """Generate a system prompt based on task description.

    Args:
        state: Wizard state with task description.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Generated system prompt string.
    """
    context_parts = [f"Task: {state.task.description}"]

    if state.domain_analysis:
        context_parts.append(f"Domain: {state.domain_analysis.domain}")

    if state.primary_schema and state.primary_schema.sample_rows:
        sample = json.dumps(state.primary_schema.sample_rows[0], indent=2)[:500]
        context_parts.append(f"Sample data:\n{sample}")

    prompt = f"""Create a concise system prompt for an AI assistant.

Context:
{chr(10).join(context_parts)}

The system prompt should:
1. Define the AI's role clearly
2. Set expectations for response style
3. Be 2-4 sentences maximum

Return ONLY the system prompt text, nothing else."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return f"You are a helpful assistant. {state.task.description}"
