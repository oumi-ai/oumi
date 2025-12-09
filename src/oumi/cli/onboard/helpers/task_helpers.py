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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..dataclasses import WizardState

from ..dataclasses import TASK_TYPES
from ..prompts import load_prompt


@dataclass
class ExtractedUseCase:
    """Use case specification extracted from customer documentation."""

    has_explicit_use_case: bool = False
    task_description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    output_schema: Optional[str] = None
    output_example: Optional[str] = None
    input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)


def extract_use_case_from_documents(
    files: list[dict], llm_analyzer
) -> Optional[ExtractedUseCase]:
    """Extract explicit use case specification from customer documents.

    Customers often provide documentation describing what they want their model to do,
    including system prompts, output schemas, and examples. This function extracts
    those elements rather than generating new ones.

    Args:
        files: List of file info dicts with schema containing raw_text.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        ExtractedUseCase if explicit use case found, None otherwise.
    """
    # Collect full document content from all files
    # We need the complete document to extract all use case details
    # (system prompts, schemas, examples may be spread throughout)
    document_contents = []
    document_infos = []
    for i, f in enumerate(files, 1):
        schema = f.get("schema")
        if schema and hasattr(schema, "raw_text") and schema.raw_text:
            document_contents.append(schema.raw_text)
            # Build info about this document
            char_count = len(schema.raw_text)
            doc_info = f"- Document {i}: {char_count:,} characters"
            if hasattr(schema, "row_count") and schema.row_count:
                doc_info += f", {schema.row_count} paragraphs/sections"
            document_infos.append(doc_info)
        elif schema and hasattr(schema, "row_count") and schema.row_count:
            # Tabular data
            col_count = len(schema.columns) if schema.columns else 0
            doc_info = f"- File {i}: {schema.row_count:,} rows, {col_count} columns"
            if schema.sample_rows:
                sample_text = json.dumps(schema.sample_rows[0], indent=2)
                document_contents.append(f"[Tabular data sample]\n{sample_text}")
            document_infos.append(doc_info)

    if not document_contents:
        return None

    # Combine all document content and info
    combined_content = "\n\n---\n\n".join(document_contents)
    combined_info = "\n".join(document_infos) if document_infos else "No size info available"

    prompt = load_prompt(
        "extract_use_case",
        document_content=combined_content,
        document_info=combined_info,
    )

    try:
        result = llm_analyzer._invoke_json(prompt)

        if not result.get("has_explicit_use_case", False):
            return None

        return ExtractedUseCase(
            has_explicit_use_case=True,
            task_description=result.get("task_description"),
            system_prompt=result.get("system_prompt"),
            user_prompt_template=result.get("user_prompt_template"),
            output_schema=result.get("output_schema"),
            output_example=result.get("output_example"),
            input_fields=result.get("input_fields", []),
            output_fields=result.get("output_fields", []),
        )
    except Exception:
        return None


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

    prompt = load_prompt(
        "infer_task_type",
        description=description,
        system_prompt=system_prompt,
        task_types_str=task_types_str,
    )

    try:
        result = llm_analyzer._invoke_json(prompt)
        task_type = result.get("task_type", "generation")
        output_format = result.get("output_format", "")

        if task_type not in TASK_TYPES:
            task_type = "generation"

        return task_type, output_format
    except Exception:
        return "generation", ""


def analyze_task_from_files(files: list[dict], llm_analyzer, domain_analysis=None) -> dict:
    """Analyze files to suggest multiple potential tasks the user might want.

    Args:
        files: List of file info dicts with analysis.
        llm_analyzer: LLMAnalyzer instance.
        domain_analysis: Optional DomainAnalysis with rich context about the data.

    Returns:
        Dict with task_options list.
    """
    file_summaries = []
    for i, f in enumerate(files, 1):
        # Use generic file identifiers instead of actual filenames
        file_type = f["extension"].lstrip(".").upper() or "FILE"
        summary_parts = [f"- File {i} ({file_type})"]

        # Include size information (helps distinguish training vs eval data)
        schema = f.get("schema")
        if schema:
            if hasattr(schema, "raw_text") and schema.raw_text:
                char_count = len(schema.raw_text)
                summary_parts.append(f"  Size: {char_count:,} characters")
            elif hasattr(schema, "row_count") and schema.row_count:
                summary_parts.append(f"  Size: {schema.row_count:,} rows")

        # Include the analyzed purpose (rich semantic description)
        if f.get("suggested_purpose"):
            summary_parts.append(f"  Purpose: {f['suggested_purpose']}")

        # Include role and reasoning
        if f.get("suggested_role"):
            role_desc = f"  Role: {f['suggested_role']}"
            if f.get("role_reason"):
                role_desc += f" - {f['role_reason']}"
            summary_parts.append(role_desc)

        # Include field count for structured data
        if schema and schema.columns:
            col_count = len(schema.columns)
            summary_parts.append(f"  Structure: {col_count} fields")

        file_summaries.append("\n".join(summary_parts))

    # Build domain context section if available
    domain_context = ""
    if domain_analysis:
        domain_parts = []
        if domain_analysis.domain:
            domain_parts.append(f"Domain: {domain_analysis.domain}")
        if domain_analysis.description:
            domain_parts.append(f"Description: {domain_analysis.description}")
        if domain_analysis.data_purpose:
            domain_parts.append(f"Data purpose: {domain_analysis.data_purpose}")
        if domain_analysis.terminology:
            terms = ", ".join(domain_analysis.terminology[:8])
            domain_parts.append(f"Key terminology: {terms}")
        if domain_parts:
            domain_context = "\n".join(domain_parts)

    prompt = load_prompt(
        "analyze_task_from_files",
        file_summaries="\n".join(file_summaries),
        domain_context=domain_context,
    )

    try:
        result = llm_analyzer._invoke_json(prompt)
        tasks = result.get("tasks", [])
        if not tasks or not isinstance(tasks, list):
            tasks = ["Analyze and respond to user queries based on the provided data"]
        return {"task_options": tasks}
    except Exception:
        return {
            "task_options": [
                "Analyze and respond to user queries based on the provided data"
            ]
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

    # Provide schema structure without sample values that might contain identifying info
    if state.primary_schema and state.primary_schema.columns:
        field_types = []
        for col in state.primary_schema.columns[:6]:
            if col.is_text:
                field_types.append("text")
            elif col.is_categorical:
                field_types.append("category")
            elif col.is_conversation:
                field_types.append("conversation")
            else:
                field_types.append(col.dtype)
        context_parts.append(f"Data structure: {', '.join(field_types)} fields")

    # Add task type hint for better prompt generation
    if state.task.task_type:
        context_parts.append(f"Task type: {state.task.task_type}")

    prompt = load_prompt(
        "generate_system_prompt",
        context="\n".join(context_parts),
    )

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return f"You are a helpful assistant. {state.task.description}"
