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
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..dataclasses import WizardState

from ..dataclasses import TASK_TYPES, DetectionResult
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


def infer_task_type(description: str, system_prompt: str, llm_analyzer, domain_analysis=None) -> tuple[str, str]:
    """Infer the task type from description and system prompt.

    Args:
        description: Task description.
        system_prompt: Generated system prompt.
        llm_analyzer: LLMAnalyzer instance.
        domain_analysis: Optional DomainAnalysis for better classification.

    Returns:
        Tuple of (task_type, example_output_format).
    """
    task_types_str = "\n".join([
        f"- {key}: {info['description']}"
        for key, info in TASK_TYPES.items()
    ])

    # Build context for prompt
    context = {
        "description": description,
        "system_prompt": system_prompt,
        "task_types_str": task_types_str,
    }

    # Add domain context if available
    if domain_analysis:
        context["domain"] = domain_analysis.domain
        if domain_analysis.terminology:
            context["terminology"] = ", ".join(domain_analysis.terminology[:5])

    prompt = load_prompt("infer_task_type", **context)

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


# =============================================================================
# Detection Functions for the new wizard flow
# =============================================================================


def detect_all_elements(
    files: list[dict],
    primary_schema: Any,
    llm_analyzer,
    domain_analysis: Any = None,
) -> DetectionResult:
    """Run all detection passes to identify what the customer has provided.

    This is the main detection function that orchestrates all individual
    detection functions and returns a comprehensive DetectionResult.

    Args:
        files: List of file info dicts with schemas.
        primary_schema: The primary data schema.
        llm_analyzer: LLMAnalyzer instance.
        domain_analysis: Optional domain analysis for context.

    Returns:
        DetectionResult with all detected elements.
    """
    result = DetectionResult()

    # 1. Detect labeled examples (input-output pairs)
    labeled_result = detect_labeled_examples(
        files, primary_schema, llm_analyzer, domain_analysis
    )
    if labeled_result:
        result.has_labeled_examples = labeled_result.get("has_labeled_examples", False)
        result.labels_confidence = labeled_result.get("confidence", 0.0)
        result.labeled_examples = labeled_result.get("examples", [])
        result.input_column = labeled_result.get("input_column")
        result.output_column = labeled_result.get("output_column")

    # 2. Detect unlabeled prompts (inputs without outputs)
    # Only check if no labeled examples found
    if not result.has_labeled_examples:
        unlabeled_result = detect_unlabeled_prompts(
            files, primary_schema, llm_analyzer, domain_analysis
        )
        if unlabeled_result:
            result.has_unlabeled_prompts = unlabeled_result.get(
                "has_unlabeled_prompts", False
            )
            result.prompts_confidence = unlabeled_result.get("confidence", 0.0)
            result.unlabeled_prompts = unlabeled_result.get("prompts", [])
            result.prompt_column = unlabeled_result.get("prompt_column")

    # 3. Extract use case (task definition, system prompt, template)
    use_case = extract_use_case_from_documents(files, llm_analyzer)
    if use_case and use_case.has_explicit_use_case:
        if use_case.task_description:
            result.has_task_definition = True
            result.task_definition = use_case.task_description
            result.task_confidence = 0.8

        if use_case.system_prompt:
            result.system_prompt = use_case.system_prompt
            result.task_confidence = max(result.task_confidence, 0.9)

        if use_case.user_prompt_template:
            result.has_user_prompt_template = True
            result.user_prompt_template = use_case.user_prompt_template
            result.template_variables = _extract_template_variables(
                use_case.user_prompt_template
            )
            result.template_confidence = 0.8

    # 4. Extract user prompt template from documents (if not found in use case)
    if not result.has_user_prompt_template:
        template_result = extract_user_prompt_template(files, primary_schema, llm_analyzer)
        if template_result:
            result.has_user_prompt_template = template_result.get("has_template", False)
            result.user_prompt_template = template_result.get("template")
            result.template_variables = template_result.get("variables", [])
            result.template_mapping = template_result.get("suggested_mapping", {})
            result.template_confidence = template_result.get("confidence", 0.0)

    # 5. Extract evaluation criteria
    eval_result = extract_evaluation_criteria(
        files, llm_analyzer, task_context=result.task_definition
    )
    if eval_result:
        result.has_eval_criteria = eval_result.get("has_eval_criteria", False)
        result.eval_criteria = [
            c.get("name", c) if isinstance(c, dict) else c
            for c in eval_result.get("criteria", [])
        ]
        result.eval_confidence = eval_result.get("confidence", 0.0)

    # 6. Identify seed columns for diversity
    seed_result = identify_seed_columns(
        primary_schema, llm_analyzer, task_context=result.task_definition
    )
    if seed_result:
        result.has_seed_data = seed_result.get("has_seed_data", False)
        result.seed_columns = [
            c.get("column", c) if isinstance(c, dict) else c
            for c in seed_result.get("seed_columns", [])
        ]

    return result


def detect_labeled_examples(
    files: list[dict],
    primary_schema: Any,
    llm_analyzer,
    domain_analysis: Any = None,
) -> Optional[dict]:
    """Detect if the data contains labeled training examples (input-output pairs).

    Args:
        files: List of file info dicts.
        primary_schema: Primary data schema.
        llm_analyzer: LLMAnalyzer instance.
        domain_analysis: Optional domain analysis for context.

    Returns:
        Dict with detection results, or None if detection fails.
    """
    if not primary_schema or not primary_schema.columns:
        return None

    # Build schema info for the prompt
    schema_info = _build_schema_info(primary_schema)
    sample_rows = json.dumps(primary_schema.sample_rows[:5], indent=2, default=str)

    prompt = load_prompt(
        "detect_labeled_examples",
        schema_info=schema_info,
        sample_rows=sample_rows,
    )

    try:
        result = llm_analyzer._invoke_json(prompt)

        if result.get("has_labeled_examples", False):
            # Extract actual examples from the data
            input_col = result.get("input_column")
            output_col = result.get("output_column")
            examples = []

            if input_col and output_col and primary_schema.sample_rows:
                for row in primary_schema.sample_rows[:10]:
                    if input_col in row and output_col in row:
                        examples.append({
                            "input": str(row[input_col]),
                            "output": str(row[output_col]),
                        })

            result["examples"] = examples

        return result
    except Exception:
        return None


def detect_unlabeled_prompts(
    files: list[dict],
    primary_schema: Any,
    llm_analyzer,
    domain_analysis: Any = None,
) -> Optional[dict]:
    """Detect if the data contains unlabeled prompts (inputs without outputs).

    Args:
        files: List of file info dicts.
        primary_schema: Primary data schema.
        llm_analyzer: LLMAnalyzer instance.
        domain_analysis: Optional domain analysis for context.

    Returns:
        Dict with detection results, or None if detection fails.
    """
    if not primary_schema or not primary_schema.columns:
        return None

    schema_info = _build_schema_info(primary_schema)
    sample_rows = json.dumps(primary_schema.sample_rows[:5], indent=2, default=str)

    prompt = load_prompt(
        "detect_unlabeled_prompts",
        schema_info=schema_info,
        sample_rows=sample_rows,
    )

    try:
        result = llm_analyzer._invoke_json(prompt)

        if result.get("has_unlabeled_prompts", False):
            # Extract prompts from the data
            prompt_col = result.get("prompt_column")
            prompts = []

            if prompt_col and primary_schema.sample_rows:
                for row in primary_schema.sample_rows[:20]:
                    if prompt_col in row and row[prompt_col]:
                        prompts.append(str(row[prompt_col]))

            result["prompts"] = prompts

        return result
    except Exception:
        return None


def extract_user_prompt_template(
    files: list[dict], primary_schema: Any, llm_analyzer
) -> Optional[dict]:
    """Extract user prompt template with placeholders from documents.

    Args:
        files: List of file info dicts.
        primary_schema: Primary data schema for column names.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with template info, or None if not found.
    """
    # Collect document content
    document_contents = []
    for f in files:
        schema = f.get("schema")
        if schema and hasattr(schema, "raw_text") and schema.raw_text:
            document_contents.append(schema.raw_text[:5000])

    if not document_contents:
        return None

    # Get column names for mapping
    column_names = []
    if primary_schema and primary_schema.columns:
        column_names = [col.name for col in primary_schema.columns]

    prompt = load_prompt(
        "extract_user_prompt_template",
        document_content="\n\n---\n\n".join(document_contents),
        column_names=", ".join(column_names) if column_names else "No columns available",
    )

    try:
        result = llm_analyzer._invoke_json(prompt)
        return result
    except Exception:
        return None


def extract_evaluation_criteria(
    files: list[dict], llm_analyzer, task_context: Optional[str] = None
) -> Optional[dict]:
    """Extract evaluation criteria from customer documents.

    Args:
        files: List of file info dicts.
        llm_analyzer: LLMAnalyzer instance.
        task_context: Optional task description for context.

    Returns:
        Dict with criteria info, or None if not found.
    """
    # Collect document content
    document_contents = []
    for f in files:
        schema = f.get("schema")
        if schema and hasattr(schema, "raw_text") and schema.raw_text:
            document_contents.append(schema.raw_text[:5000])

    if not document_contents:
        return None

    prompt = load_prompt(
        "extract_evaluation_criteria",
        document_content="\n\n---\n\n".join(document_contents),
        task_context=task_context or "Not yet defined",
    )

    try:
        result = llm_analyzer._invoke_json(prompt)
        return result
    except Exception:
        return None


def identify_seed_columns(
    primary_schema: Any, llm_analyzer, task_context: Optional[str] = None
) -> Optional[dict]:
    """Identify columns that can be used as seed data for diversity.

    Args:
        primary_schema: Primary data schema.
        llm_analyzer: LLMAnalyzer instance.
        task_context: Optional task description for context.

    Returns:
        Dict with seed column info, or None if not found.
    """
    if not primary_schema or not primary_schema.columns:
        return None

    schema_info = _build_schema_info(primary_schema)

    # Build sample values for each column
    sample_values = {}
    if primary_schema.sample_rows:
        for col in primary_schema.columns:
            values = []
            for row in primary_schema.sample_rows[:10]:
                if col.name in row and row[col.name] is not None:
                    val = str(row[col.name])[:100]
                    if val not in values:
                        values.append(val)
            if values:
                sample_values[col.name] = values[:5]

    prompt = load_prompt(
        "identify_seed_columns",
        schema_info=schema_info,
        sample_values=json.dumps(sample_values, indent=2, default=str),
        task_context=task_context or "Not yet defined",
    )

    try:
        result = llm_analyzer._invoke_json(prompt)
        return result
    except Exception:
        return None


def _build_schema_info(schema: Any) -> str:
    """Build a schema info string for prompts.

    Args:
        schema: DataSchema instance.

    Returns:
        Formatted schema info string.
    """
    if not schema or not schema.columns:
        return "No schema available"

    lines = [f"Row count: {schema.row_count}"]
    lines.append("Columns:")
    for col in schema.columns:
        col_info = f"  - {col.name} ({col.dtype})"
        if col.is_text:
            col_info += " [TEXT]"
        if col.is_categorical:
            col_info += " [CATEGORICAL]"
        if col.is_conversation:
            col_info += " [CONVERSATION]"
        if col.unique_count:
            col_info += f" - {col.unique_count} unique values"
        lines.append(col_info)

    return "\n".join(lines)


def _extract_template_variables(template: str) -> list[str]:
    """Extract variable names from a template string.

    Supports {var}, {{var}}, and $var syntax.

    Args:
        template: Template string with placeholders.

    Returns:
        List of variable names found.
    """
    if not template:
        return []

    variables = set()

    # Match {variable} or {{variable}}
    for match in re.finditer(r"\{+(\w+)\}+", template):
        variables.add(match.group(1))

    # Match $variable
    for match in re.finditer(r"\$(\w+)", template):
        variables.add(match.group(1))

    return list(variables)
