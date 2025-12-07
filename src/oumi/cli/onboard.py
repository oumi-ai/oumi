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

"""CLI commands for customer onboarding.

This module provides interactive wizard and config generation commands
to help customers quickly set up Oumi for their use cases.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

import oumi.cli.cli_utils as cli_utils

# Goal choices
GOAL_CHOICES = ["synth", "judge", "train", "pipeline"]
SYNTH_GOAL_CHOICES = ["qa", "conversation", "augmentation", "instruction"]
JUDGE_TYPE_CHOICES = ["generic", "compliance", "relevance", "safety", "groundedness"]

# Supported file extensions for auto-detection
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".docx", ".doc"}

# File role descriptions for the wizard
FILE_ROLE_GUIDANCE = {
    "primary": {
        "title": "PRIMARY DATA",
        "description": "The main data you want to process",
        "help": [
            "This is what you'll synthesize from, evaluate, or train on",
            "Usually your largest dataset",
            "Examples: customer conversations, invoices, documents to classify",
        ],
    },
    "reference": {
        "title": "REFERENCE DATA",
        "description": "Valid values to match against",
        "help": [
            "Lookup tables, catalogs, or lists of valid options",
            "Used to check if primary data matches known values",
            "Examples: product catalog, service types, approved terms",
        ],
    },
    "rules": {
        "title": "RULES/GUIDELINES",
        "description": "Evaluation criteria and standards",
        "help": [
            "Documents describing what makes data valid/invalid",
            "Quality standards or processing rules",
            "Examples: style guide, validation rules, compliance requirements",
        ],
    },
    "examples": {
        "title": "LABELED EXAMPLES",
        "description": "Known good/bad samples",
        "help": [
            "Examples with labels like 'valid', 'invalid', 'approved'",
            "Helps train classifiers and calibrate judges",
            "Examples: approved_responses.csv, rejected_items.txt",
        ],
    },
}

# Column role descriptions for granular assignment
COLUMN_ROLE_GUIDANCE = {
    "context": {
        "title": "CONTEXT",
        "description": "Main content to generate from",
        "help": "Text column containing the source material for synthesis",
    },
    "question": {
        "title": "QUESTION",
        "description": "Existing questions to use or augment",
        "help": "Column containing questions (if you have existing Q&A data)",
    },
    "answer": {
        "title": "ANSWER",
        "description": "Existing answers to use or augment",
        "help": "Column containing answers (if you have existing Q&A data)",
    },
    "reference_values": {
        "title": "REFERENCE VALUES",
        "description": "Valid values for validation",
        "help": "Column with valid/approved values to match against",
    },
    "label": {
        "title": "LABEL",
        "description": "Classification labels",
        "help": "Column with labels like 'valid', 'invalid', 'category'",
    },
    "metadata": {
        "title": "METADATA",
        "description": "Additional context",
        "help": "Supporting information to include in prompts",
    },
}

# Tabular file extensions that support column selection
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}


def _detect_files_in_directory(dir_path: Path) -> list[dict]:
    """Scan directory for supported data files.

    Args:
        dir_path: Path to the directory to scan.

    Returns:
        List of file info dicts with path, name, extension, and size.
    """
    files = []
    for item in dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append({
                "path": item,
                "name": item.name,
                "extension": item.suffix.lower(),
                "size": item.stat().st_size,
                "suggested_purpose": None,  # Will be filled by LLM analysis
                "suggested_role": None,
            })
    # Sort by size descending (largest files first)
    return sorted(files, key=lambda f: f["size"], reverse=True)


def _analyze_file_purposes(files: list[dict], analyzer, llm_analyzer) -> list[dict]:
    """Use LLM to analyze each file and suggest its purpose.

    Args:
        files: List of file info dicts.
        analyzer: DataAnalyzer instance.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Updated files list with suggested purposes.
    """
    import json

    for f in files:
        try:
            # Get schema for data files
            schema = None
            sample_content = ""

            if f["extension"] in {".csv", ".json", ".jsonl", ".xlsx", ".xls"}:
                schema = analyzer.analyze(f["path"])
                sample_content = json.dumps(schema.sample_rows[:2], indent=2) if schema.sample_rows else ""
                columns = [c.name for c in schema.columns] if schema.columns else []
            elif f["extension"] in {".docx", ".doc"}:
                schema = analyzer.analyze(f["path"])
                sample_content = schema.raw_text[:1000] if schema.raw_text else ""
                columns = []

            # Build analysis prompt with detailed role explanations
            prompt = f"""Analyze this file and suggest what role it should play in an ML training pipeline.

FILE: {f['name']}
TYPE: {f['extension']}
"""
            if columns:
                prompt += f"COLUMNS: {columns}\n"
            if sample_content:
                prompt += f"\nSAMPLE CONTENT:\n{sample_content[:1500]}\n"

            prompt += """
AVAILABLE ROLES:
- primary: Main data to process - this is the core dataset for synthesis, evaluation, or training
- reference: Lookup/validation data - catalogs, valid values lists, or mapping tables
- rules: Guidelines documents - style guides, policies, or criteria definitions
- examples: Labeled samples - data with known good/bad labels for training or calibration
- context: Supporting information - background docs that inform but aren't directly processed

Return a JSON object with:
{
    "purpose": "Brief description of what this file contains (1-2 sentences)",
    "suggested_role": "primary|reference|rules|examples|context",
    "role_explanation": "A clear 1-sentence explanation of WHY this role fits, citing specific evidence from the file (e.g., column names, content patterns, file structure)"
}

Be specific in the role_explanation - cite actual column names or content that led to your recommendation.

Return ONLY the JSON object."""

            result = llm_analyzer._invoke_json(prompt)
            f["suggested_purpose"] = result.get("purpose", "Unknown")
            f["suggested_role"] = result.get("suggested_role", "context")
            f["role_reason"] = result.get("role_explanation", result.get("role_reason", ""))
            f["schema"] = schema

        except Exception as e:
            f["suggested_purpose"] = f"(Analysis failed: {str(e)[:50]})"
            f["suggested_role"] = "context"
            f["role_reason"] = "Default assignment due to analysis failure"

    return files


def _display_columns_for_file(file_info: dict, schema=None) -> list[str]:
    """Display columns available in a tabular file.

    Args:
        file_info: File info dict with path and extension.
        schema: Optional pre-analyzed schema.

    Returns:
        List of column names.
    """
    if file_info["extension"] not in TABULAR_EXTENSIONS:
        return []

    columns = []
    if schema and schema.columns:
        columns = [c.name for c in schema.columns]
    elif file_info.get("schema") and file_info["schema"].columns:
        columns = [c.name for c in file_info["schema"].columns]

    if columns:
        table = Table(
            title=f"Columns in {file_info['name']}",
            show_edge=False,
            title_style="cyan",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Column Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Sample Value", style="dim", max_width=40)

        schema_obj = schema or file_info.get("schema")
        for i, col_name in enumerate(columns, 1):
            col_info = None
            if schema_obj:
                col_info = next(
                    (c for c in schema_obj.columns if c.name == col_name), None
                )

            dtype = col_info.dtype if col_info else "unknown"
            sample = ""
            if col_info and col_info.sample_values:
                sample = str(col_info.sample_values[0])[:40]

            table.add_row(str(i), col_name, dtype, sample)

        cli_utils.CONSOLE.print(table)

    return columns


def _analyze_column_roles(all_columns: list[dict], llm_analyzer) -> list[dict]:
    """Use AI to analyze columns and suggest roles.

    Args:
        all_columns: List of column info dicts.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Updated column list with suggested roles.
    """
    import json

    # Build column descriptions for the LLM
    columns_desc = []
    for i, col_data in enumerate(all_columns, 1):
        col_info = col_data["col_info"]
        sample_vals = col_info.sample_values[:2] if col_info.sample_values else []
        columns_desc.append({
            "index": i,
            "file": col_data["file"]["name"],
            "column": col_data["column"],
            "dtype": col_info.dtype,
            "is_text": col_info.is_text,
            "avg_length": col_info.avg_length,
            "sample_values": [str(v)[:100] for v in sample_vals],
        })

    prompt = f"""Analyze these columns and suggest the best role for each in an ML data pipeline.

COLUMNS:
{json.dumps(columns_desc, indent=2)}

AVAILABLE ROLES:
- context: Main text content for synthesis/generation (longest, most informative text)
- question: Contains questions or queries
- answer: Contains answers or responses
- reference_values: Lookup values, categories, or valid options for validation
- metadata: Supporting information like IDs, dates, categories
- label: Classification labels (valid/invalid, categories, scores)
- skip: Column not useful for ML pipeline

For each column, suggest ONE role based on:
1. Column name patterns (e.g., "query" → question, "response" → answer)
2. Data type and content (text vs numeric vs categorical)
3. Average length (longer text = more likely context/answer)
4. Sample values

Return a JSON object:
{{
    "column_roles": [
        {{"index": 1, "role": "context", "reason": "Why this role fits"}},
        {{"index": 2, "role": "metadata", "reason": "Why this role fits"}}
    ]
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        role_map = {
            item["index"]: {"role": item["role"], "reason": item.get("reason", "")}
            for item in result.get("column_roles", [])
        }

        for i, col_data in enumerate(all_columns, 1):
            if i in role_map:
                col_data["suggested_role"] = role_map[i]["role"]
                col_data["role_reason"] = role_map[i]["reason"]
            else:
                col_data["suggested_role"] = "metadata"
                col_data["role_reason"] = "Default assignment"

    except Exception as e:
        cli_utils.CONSOLE.print(
            f"[yellow]Warning: AI column analysis failed: {e}[/yellow]"
        )
        # Fallback: use heuristics
        for col_data in all_columns:
            col_info = col_data["col_info"]
            col_name = col_data["column"].lower()

            if col_info.is_text and (col_info.avg_length or 0) > 100:
                col_data["suggested_role"] = "context"
                col_data["role_reason"] = "Long text content"
            elif "question" in col_name or "query" in col_name:
                col_data["suggested_role"] = "question"
                col_data["role_reason"] = "Column name suggests questions"
            elif "answer" in col_name or "response" in col_name:
                col_data["suggested_role"] = "answer"
                col_data["role_reason"] = "Column name suggests answers"
            elif "label" in col_name or "class" in col_name or "category" in col_name:
                col_data["suggested_role"] = "label"
                col_data["role_reason"] = "Column name suggests labels"
            elif col_info.is_categorical:
                col_data["suggested_role"] = "reference_values"
                col_data["role_reason"] = "Categorical values"
            else:
                col_data["suggested_role"] = "metadata"
                col_data["role_reason"] = "Supporting information"

    return all_columns


def _prompt_column_roles(
    files: list[dict], analyzer, llm_analyzer=None, verbose: bool = False
) -> dict[str, dict]:
    """Prompt user to assign roles to specific columns from tabular files.

    Args:
        files: List of file info dicts.
        analyzer: DataAnalyzer instance.
        llm_analyzer: Optional LLMAnalyzer for AI suggestions.
        verbose: Whether to show detailed output.

    Returns:
        Dict mapping role names to {"path": Path, "column": str} or {"path": Path}.
    """
    column_assignments = {}

    # First, analyze all tabular files to get their columns
    tabular_files = [
        f for f in files if f["extension"] in TABULAR_EXTENSIONS
    ]
    doc_files = [
        f for f in files if f["extension"] not in TABULAR_EXTENSIONS
    ]

    if not tabular_files:
        cli_utils.CONSOLE.print(
            "[yellow]No tabular files found. Using document-level assignments.[/yellow]"
        )
        return _prompt_file_roles(files)

    # Collect all columns from all files
    all_columns = []  # List of {"file": file_info, "column": str, "schema": schema}
    for f in tabular_files:
        try:
            schema = f.get("schema") or analyzer.analyze(f["path"])
            f["schema"] = schema
            if schema.columns:
                for col in schema.columns:
                    all_columns.append({
                        "file": f,
                        "column": col.name,
                        "col_info": col,
                        "schema": schema,
                    })
        except Exception as e:
            cli_utils.CONSOLE.print(
                f"[yellow]Warning: Could not analyze {f['name']}: {e}[/yellow]"
            )

    if not all_columns:
        cli_utils.CONSOLE.print(
            "[yellow]No columns found in tabular files.[/yellow]"
        )
        return _prompt_file_roles(files)

    # Use AI to suggest column roles if available
    if llm_analyzer:
        with cli_utils.CONSOLE.status(
            "[dim]Analyzing columns with AI...[/dim]", spinner="dots"
        ):
            all_columns = _analyze_column_roles(all_columns, llm_analyzer)

    # Display columns - compact or verbose
    if verbose:
        cli_utils.CONSOLE.print(
            Panel(
                "[bold]Column-Level Role Assignment[/bold]\n\n"
                "[dim]The AI has analyzed your columns and suggested roles.\n"
                "You can accept these suggestions or customize them.[/dim]",
                border_style="cyan",
            )
        )
        # Full table with all details
        table = Table(title="Available Columns", show_edge=False, expand=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("File", style="green", no_wrap=True)
        table.add_column("Column", style="yellow", no_wrap=True)
        table.add_column("Type", style="dim", no_wrap=True)
        table.add_column("AI Role", style="magenta", no_wrap=True)
        table.add_column("Why?", style="dim")

        for i, col_data in enumerate(all_columns, 1):
            suggested_role = col_data.get("suggested_role", "").upper()
            role_reason = col_data.get("role_reason", "")
            table.add_row(
                str(i),
                col_data["file"]["name"],
                col_data["column"],
                col_data["col_info"].dtype,
                suggested_role,
                role_reason,
            )
        cli_utils.CONSOLE.print(table)
    else:
        # Compact display
        cli_utils.CONSOLE.print("\n[bold]Column Roles:[/bold]")
        for i, col_data in enumerate(all_columns, 1):
            suggested_role = col_data.get("suggested_role", "").upper()
            cli_utils.CONSOLE.print(
                f"  [{i}] [yellow]{col_data['column']}[/yellow] → [magenta]{suggested_role}[/magenta]"
            )

    # Ask if user wants to use AI suggestions or customize
    use_ai = Confirm.ask(
        "\nUse AI-suggested column roles?",
        default=True,
    )

    if use_ai and llm_analyzer:
        # Build assignments from AI suggestions
        for col_data in all_columns:
            role = col_data.get("suggested_role", "").lower()
            if role in ("context", "question", "answer"):
                if role not in column_assignments:
                    column_assignments[role] = {
                        "path": col_data["file"]["path"],
                        "column": col_data["column"],
                        "file_name": col_data["file"]["name"],
                    }
            elif role == "reference_values":
                if "reference_values" not in column_assignments:
                    column_assignments["reference_values"] = {
                        "path": col_data["file"]["path"],
                        "column": col_data["column"],
                        "file_name": col_data["file"]["name"],
                    }
            elif role == "label":
                if "label" not in column_assignments:
                    column_assignments["label"] = {
                        "path": col_data["file"]["path"],
                        "column": col_data["column"],
                        "file_name": col_data["file"]["name"],
                    }
            elif role == "metadata":
                if "metadata" not in column_assignments:
                    column_assignments["metadata"] = []
                column_assignments["metadata"].append({
                    "path": col_data["file"]["path"],
                    "column": col_data["column"],
                    "file_name": col_data["file"]["name"],
                })

        # Ensure we have a primary file and context
        if "context" in column_assignments:
            column_assignments["primary"] = {
                "path": column_assignments["context"]["path"],
                "schema": next(
                    c["schema"] for c in all_columns
                    if c["column"] == column_assignments["context"]["column"]
                ),
            }
        elif all_columns:
            # Fallback to first column
            column_assignments["context"] = {
                "path": all_columns[0]["file"]["path"],
                "column": all_columns[0]["column"],
                "file_name": all_columns[0]["file"]["name"],
            }
            column_assignments["primary"] = {
                "path": all_columns[0]["file"]["path"],
                "schema": all_columns[0]["schema"],
            }

    else:
        # Manual assignment flow
        col_nums = [str(i) for i in range(1, len(all_columns) + 1)]

        # Prompt for context column (required)
        cli_utils.CONSOLE.print(
            "\n[bold cyan]CONTEXT[/bold cyan] - Main content for synthesis\n"
            "[dim]Which column contains the text you want to generate from?[/dim]"
        )

        context_idx = IntPrompt.ask(
            "Select context column",
            choices=col_nums,
            default="1",
        )
        context_col = all_columns[int(context_idx) - 1]
        column_assignments["context"] = {
            "path": context_col["file"]["path"],
            "column": context_col["column"],
            "file_name": context_col["file"]["name"],
        }

        # Set primary to the file containing context
        column_assignments["primary"] = {
            "path": context_col["file"]["path"],
            "schema": context_col["schema"],
        }

        remaining_nums = [n for n in col_nums if n != str(context_idx)]

        # Optional: Reference values column
        if remaining_nums:
            cli_utils.CONSOLE.print(
                "\n[bold cyan]REFERENCE VALUES[/bold cyan] - Valid values for validation (optional)\n"
                "[dim]Column with approved values, categories, or lookup data.[/dim]"
            )
            if Confirm.ask("Do you have a reference values column?", default=False):
                ref_idx = Prompt.ask("Select column", choices=remaining_nums)
                ref_col = all_columns[int(ref_idx) - 1]
                column_assignments["reference_values"] = {
                    "path": ref_col["file"]["path"],
                    "column": ref_col["column"],
                    "file_name": ref_col["file"]["name"],
                }
                remaining_nums = [n for n in remaining_nums if n != ref_idx]

        # Optional: Metadata column
        if remaining_nums:
            cli_utils.CONSOLE.print(
                "\n[bold cyan]METADATA[/bold cyan] - Additional context (optional)\n"
                "[dim]Extra information to include in prompts (e.g., category, type).[/dim]"
            )
            if Confirm.ask("Do you have metadata columns to include?", default=False):
                # Allow multiple metadata columns
                metadata_cols = []
                while remaining_nums:
                    meta_idx = Prompt.ask(
                        "Select metadata column (or 'done')",
                        default="done",
                    )
                    if meta_idx.lower() == "done":
                        break
                    if meta_idx in remaining_nums:
                        meta_col = all_columns[int(meta_idx) - 1]
                        metadata_cols.append({
                            "path": meta_col["file"]["path"],
                            "column": meta_col["column"],
                            "file_name": meta_col["file"]["name"],
                        })
                        remaining_nums = [n for n in remaining_nums if n != meta_idx]

                if metadata_cols:
                    column_assignments["metadata"] = metadata_cols

        # Optional: Label column (for training/classification)
        if remaining_nums:
            cli_utils.CONSOLE.print(
                "\n[bold cyan]LABEL[/bold cyan] - Classification labels (optional)\n"
                "[dim]Column with labels like 'valid', 'invalid', or categories.[/dim]"
            )
            if Confirm.ask("Do you have a label column?", default=False):
                label_idx = Prompt.ask("Select column", choices=remaining_nums)
                label_col = all_columns[int(label_idx) - 1]
                column_assignments["label"] = {
                    "path": label_col["file"]["path"],
                    "column": label_col["column"],
                    "file_name": label_col["file"]["name"],
                }

    # Handle document files separately
    if doc_files:
        cli_utils.CONSOLE.print(
            "\n[bold cyan]Document Files[/bold cyan]\n"
            "[dim]These non-tabular files can be assigned to roles:[/dim]"
        )
        for i, f in enumerate(doc_files, 1):
            cli_utils.CONSOLE.print(f"  [{i}] {f['name']}")

        if Confirm.ask("\nDo any of these contain rules or guidelines?", default=False):
            rules_idx = Prompt.ask(
                "Which file?",
                choices=[str(i) for i in range(1, len(doc_files) + 1)],
            )
            column_assignments["rules"] = {
                "path": doc_files[int(rules_idx) - 1]["path"],
            }

    # Display summary
    cli_utils.CONSOLE.print("\n[bold green]Column Assignments Summary:[/bold green]")
    for role, data in column_assignments.items():
        if role == "metadata" and isinstance(data, list):
            cols = ", ".join(d["column"] for d in data)
            cli_utils.CONSOLE.print(f"  [cyan]{role}:[/cyan] {cols}")
        elif isinstance(data, dict) and "column" in data:
            cli_utils.CONSOLE.print(
                f"  [cyan]{role}:[/cyan] {data.get('file_name', 'file')}.{data['column']}"
            )
        elif isinstance(data, dict) and "path" in data:
            cli_utils.CONSOLE.print(
                f"  [cyan]{role}:[/cyan] {data['path'].name if hasattr(data['path'], 'name') else data['path']}"
            )

    return column_assignments


def _analyze_columns_with_llm(
    files: list[dict], analyzer, llm_analyzer
) -> dict[str, dict]:
    """Use LLM to analyze and suggest column roles.

    Args:
        files: List of file info dicts with schemas.
        analyzer: DataAnalyzer instance.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with suggested column assignments.
    """
    import json

    # Gather all column info
    all_columns = []
    for f in files:
        if f["extension"] not in TABULAR_EXTENSIONS:
            continue

        schema = f.get("schema")
        if not schema:
            try:
                schema = analyzer.analyze(f["path"])
                f["schema"] = schema
            except Exception:
                continue

        if schema.columns:
            for col in schema.columns:
                sample = col.sample_values[0] if col.sample_values else ""
                all_columns.append({
                    "file": f["name"],
                    "column": col.name,
                    "type": col.dtype,
                    "is_text": col.is_text,
                    "avg_length": col.avg_length,
                    "sample": str(sample)[:100],
                })

    if not all_columns:
        return {}

    prompt = f"""Analyze these columns and suggest their roles for an ML training pipeline.

COLUMNS:
{json.dumps(all_columns, indent=2)}

AVAILABLE ROLES:
- context: Main text content for synthesis (required, pick the best text column)
- reference_values: Lookup values, valid options, categories
- metadata: Supporting information (can pick multiple)
- label: Classification labels like 'valid', 'invalid'

Return JSON:
{{
    "suggestions": [
        {{
            "file": "filename",
            "column": "column_name",
            "suggested_role": "context|reference_values|metadata|label",
            "confidence": 0.0-1.0,
            "reason": "Why this role fits"
        }}
    ],
    "primary_context": {{
        "file": "best file for context",
        "column": "best column for context"
    }}
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        return result
    except Exception as e:
        cli_utils.CONSOLE.print(
            f"[yellow]Warning: Column analysis failed: {e}[/yellow]"
        )
        return {}


def _display_file_listing(files: list[dict], analyzer, show_ai_analysis: bool = False):
    """Display detected files with type info.

    Args:
        files: List of file info dicts.
        analyzer: DataAnalyzer instance for quick schema detection.
        show_ai_analysis: Whether to show AI-analyzed purposes.
    """
    table = Table(title="Files Detected", show_edge=False)
    table.add_column("#", style="cyan", width=3)
    table.add_column("File", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Size", style="dim")

    if show_ai_analysis:
        table.add_column("AI Analysis", style="white", max_width=50)
        table.add_column("Suggested Role", style="magenta")
    else:
        table.add_column("Details", style="dim")

    for i, f in enumerate(files, 1):
        # Get basic info without full analysis
        ext = f["extension"]
        type_map = {
            ".csv": "CSV",
            ".json": "JSON",
            ".jsonl": "JSONL",
            ".xlsx": "Excel",
            ".xls": "Excel",
            ".docx": "Word",
            ".doc": "Word",
        }
        file_type = type_map.get(ext, ext.upper())

        # Format size
        size = f["size"]
        if size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        elif size > 1000:
            size_str = f"{size / 1000:.1f} KB"
        else:
            size_str = f"{size} B"

        if show_ai_analysis and f.get("suggested_purpose"):
            # Show AI analysis
            purpose = f.get("suggested_purpose", "")
            role = f.get("suggested_role", "unknown").upper()
            table.add_row(str(i), f["name"], file_type, size_str, purpose, role)
        else:
            # Quick peek at structure if it's a data file
            details = ""
            if ext in {".csv", ".json", ".jsonl", ".xlsx", ".xls"}:
                try:
                    schema = f.get("schema") or analyzer.analyze(f["path"])
                    details = f"{schema.row_count} rows, {len(schema.columns)} cols"
                except Exception:
                    details = "(unable to read)"
            table.add_row(str(i), f["name"], file_type, size_str, details)

    cli_utils.CONSOLE.print(table)


def _iterative_system_prompt_builder(
    schema, domain, llm_analyzer, file_roles: dict = None
) -> str:
    """Interactively build a system prompt with user feedback.

    Args:
        schema: DataSchema of primary data.
        domain: DomainAnalysis from LLM.
        llm_analyzer: LLMAnalyzer instance.
        file_roles: Dict of file roles for context.

    Returns:
        Final approved system prompt.
    """
    import json

    cli_utils.CONSOLE.print(
        "\n[bold cyan]Building System Prompt[/bold cyan]\n"
        "[dim]Let's create an expert persona for your AI. "
        "I'll generate a draft and we'll refine it together.[/dim]\n"
    )

    # Build context for prompt generation
    context = {
        "domain": domain.domain if domain else "general",
        "description": domain.description if domain else "",
        "terminology": domain.terminology if domain else [],
        "columns": [c.name for c in schema.columns] if schema.columns else [],
        "sample": schema.sample_rows[0] if schema.sample_rows else {},
    }

    if file_roles:
        context["files"] = {k: str(v.name) if hasattr(v, 'name') else str(v) for k, v in file_roles.items()}

    # Generate initial system prompt
    prompt = f"""Create a system prompt for an AI expert working in the {context['domain']} domain.

CONTEXT:
- Domain: {context['domain']}
- Description: {context['description']}
- Key terminology: {context['terminology']}
- Data columns: {context['columns']}
- Sample data: {json.dumps(context['sample'], indent=2)[:500]}

Create a detailed system prompt that:
1. Establishes an expert persona specific to this domain
2. Sets clear expectations for output quality
3. Incorporates domain terminology naturally
4. Is 3-5 sentences long

Return ONLY the system prompt text, no JSON or formatting."""

    current_prompt = llm_analyzer._invoke(prompt).strip()

    # Show initial version
    cli_utils.CONSOLE.print(
        Panel(
            current_prompt,
            title="[green]Draft System Prompt[/green]",
            border_style="green",
        )
    )

    # Iterative refinement loop
    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        cli_utils.CONSOLE.print(
            "\n[bold]Options:[/bold]\n"
            "  [cyan][1][/cyan] Accept this prompt\n"
            "  [cyan][2][/cyan] Make it more formal/professional\n"
            "  [cyan][3][/cyan] Make it more conversational/friendly\n"
            "  [cyan][4][/cyan] Add specific expertise or focus\n"
            "  [cyan][5][/cyan] Custom feedback (describe what to change)\n"
        )

        choice = Prompt.ask(
            "Your choice",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )

        if choice == "1":
            break

        feedback = ""
        if choice == "2":
            feedback = "Make this more formal and professional in tone."
        elif choice == "3":
            feedback = "Make this more conversational and approachable."
        elif choice == "4":
            expertise = Prompt.ask("What expertise or focus should be added?")
            feedback = f"Add specific expertise in: {expertise}"
        elif choice == "5":
            feedback = Prompt.ask("Describe what changes you'd like")

        # Refine the prompt
        refine_prompt = f"""Revise this system prompt based on the feedback.

CURRENT PROMPT:
{current_prompt}

FEEDBACK: {feedback}

Return ONLY the revised system prompt text, no JSON or formatting."""

        current_prompt = llm_analyzer._invoke(refine_prompt).strip()

        cli_utils.CONSOLE.print(
            Panel(
                current_prompt,
                title=f"[green]Revised System Prompt (v{iteration + 2})[/green]",
                border_style="green",
            )
        )

        iteration += 1

    cli_utils.CONSOLE.print("[green]✓ System prompt finalized[/green]")
    return current_prompt


def _iterative_question_template_builder(
    schema, domain, system_prompt: str, llm_analyzer
) -> str:
    """Interactively build a question/instruction template with user feedback.

    Args:
        schema: DataSchema of primary data.
        domain: DomainAnalysis from LLM.
        system_prompt: The finalized system prompt.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Final approved question template.
    """
    import json

    cli_utils.CONSOLE.print(
        "\n[bold cyan]Building User Question Template[/bold cyan]\n"
        "[dim]Now let's create the template for user questions/instructions. "
        "This defines what the AI will be asked to do.[/dim]\n"
    )

    # Get column names for placeholders
    columns = [c.name for c in schema.columns] if schema.columns else []
    placeholder_hint = ", ".join(f"{{{c}}}" for c in columns[:5])

    cli_utils.CONSOLE.print(
        f"[dim]Available placeholders from your data: {placeholder_hint}[/dim]\n"
    )

    # Generate initial question template
    context = {
        "domain": domain.domain if domain else "general",
        "terminology": domain.terminology if domain else [],
        "columns": columns,
        "sample": schema.sample_rows[0] if schema.sample_rows else {},
        "system_prompt": system_prompt,
    }

    prompt = f"""Create a user instruction template for generating training data.

CONTEXT:
- Domain: {context['domain']}
- System prompt: {context['system_prompt'][:200]}...
- Available columns: {context['columns']}
- Sample row: {json.dumps(context['sample'], indent=2)[:400]}

Create an instruction template that:
1. Uses {{column_name}} placeholders for data fields
2. Clearly describes what output is expected
3. Is specific to the {context['domain']} domain
4. Will produce high-quality training examples

Example format:
"Based on the following [data type]: {{column_name}}

Generate a [output type] that [specific requirement]."

Return ONLY the template text with placeholders, no JSON."""

    current_template = llm_analyzer._invoke(prompt).strip()

    # Show initial version
    cli_utils.CONSOLE.print(
        Panel(
            current_template,
            title="[yellow]Draft Question Template[/yellow]",
            border_style="yellow",
        )
    )

    # Iterative refinement loop
    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        cli_utils.CONSOLE.print(
            "\n[bold]Options:[/bold]\n"
            "  [cyan][1][/cyan] Accept this template\n"
            "  [cyan][2][/cyan] Add more specific instructions\n"
            "  [cyan][3][/cyan] Simplify the template\n"
            "  [cyan][4][/cyan] Change the output format requested\n"
            "  [cyan][5][/cyan] Custom feedback\n"
        )

        choice = Prompt.ask(
            "Your choice",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )

        if choice == "1":
            break

        feedback = ""
        if choice == "2":
            details = Prompt.ask("What specific instructions should be added?")
            feedback = f"Add more specific instructions about: {details}"
        elif choice == "3":
            feedback = "Simplify this template - make it more concise and direct."
        elif choice == "4":
            new_format = Prompt.ask("What output format should be requested?")
            feedback = f"Change to request this output format: {new_format}"
        elif choice == "5":
            feedback = Prompt.ask("Describe what changes you'd like")

        # Refine the template
        refine_prompt = f"""Revise this user instruction template based on the feedback.

CURRENT TEMPLATE:
{current_template}

AVAILABLE COLUMNS: {columns}

FEEDBACK: {feedback}

Return ONLY the revised template text with {{placeholders}}, no JSON."""

        current_template = llm_analyzer._invoke(refine_prompt).strip()

        cli_utils.CONSOLE.print(
            Panel(
                current_template,
                title=f"[yellow]Revised Question Template (v{iteration + 2})[/yellow]",
                border_style="yellow",
            )
        )

        iteration += 1

    cli_utils.CONSOLE.print("[green]✓ Question template finalized[/green]")
    return current_template


def _iterative_answer_template_builder(
    schema, domain, system_prompt: str, question_template: str, llm_analyzer
) -> tuple[str, dict]:
    """Interactively build an answer template with user feedback.

    Args:
        schema: DataSchema of primary data.
        domain: DomainAnalysis from LLM.
        system_prompt: The finalized system prompt.
        question_template: The finalized question template.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Tuple of (answer template, postprocessing config).
    """
    import json

    cli_utils.CONSOLE.print(
        "\n[bold cyan]Building Answer Template[/bold cyan]\n"
        "[dim]Finally, let's define how the AI should format its answers. "
        "This ensures consistent, parseable outputs.[/dim]\n"
    )

    columns = [c.name for c in schema.columns] if schema.columns else []

    context = {
        "domain": domain.domain if domain else "general",
        "system_prompt": system_prompt,
        "question_template": question_template,
    }

    prompt = f"""Create an answer generation instruction and format.

CONTEXT:
- Domain: {context['domain']}
- System prompt: {context['system_prompt'][:150]}...
- Question template: {context['question_template'][:200]}...

Create an answer instruction that:
1. Tells the AI how to generate the response
2. Specifies a clear output format with a prefix (e.g., "Answer:", "Response:")
3. Ensures consistent, parseable outputs

Return JSON with:
{{
    "instruction": "The instruction for generating the answer (can reference {{question}} and data placeholders)",
    "output_prefix": "The prefix the answer should start with (e.g., 'Answer:')",
    "format_guidance": "Brief guidance on answer format"
}}

Return ONLY the JSON object."""

    result = llm_analyzer._invoke_json(prompt)

    instruction = result.get("instruction", "Provide a helpful response to: {question}")
    output_prefix = result.get("output_prefix", "Answer:")
    format_guidance = result.get("format_guidance", "")

    # Build display template
    current_template = f"{instruction}\n\nFormat your response as:\n{output_prefix} <your response>"

    # Show initial version
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Instruction:[/bold]\n{instruction}\n\n"
            f"[bold]Output prefix:[/bold] {output_prefix}\n\n"
            f"[bold]Format guidance:[/bold] {format_guidance}",
            title="[magenta]Draft Answer Template[/magenta]",
            border_style="magenta",
        )
    )

    # Iterative refinement loop
    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        cli_utils.CONSOLE.print(
            "\n[bold]Options:[/bold]\n"
            "  [cyan][1][/cyan] Accept this template\n"
            "  [cyan][2][/cyan] Request longer/more detailed answers\n"
            "  [cyan][3][/cyan] Request shorter/concise answers\n"
            "  [cyan][4][/cyan] Change the output prefix\n"
            "  [cyan][5][/cyan] Custom feedback\n"
        )

        choice = Prompt.ask(
            "Your choice",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )

        if choice == "1":
            break

        feedback = ""
        if choice == "2":
            feedback = "Request longer, more detailed and comprehensive answers."
        elif choice == "3":
            feedback = "Request shorter, more concise answers - just the essential information."
        elif choice == "4":
            new_prefix = Prompt.ask("What prefix should answers use?")
            output_prefix = new_prefix
            feedback = f"Use this output prefix: {new_prefix}"
        elif choice == "5":
            feedback = Prompt.ask("Describe what changes you'd like")

        # Refine the template
        refine_prompt = f"""Revise this answer generation instruction based on the feedback.

CURRENT:
- Instruction: {instruction}
- Output prefix: {output_prefix}
- Format guidance: {format_guidance}

FEEDBACK: {feedback}

Return JSON with:
{{
    "instruction": "revised instruction",
    "output_prefix": "{output_prefix}",
    "format_guidance": "revised format guidance"
}}

Return ONLY the JSON object."""

        result = llm_analyzer._invoke_json(refine_prompt)
        instruction = result.get("instruction", instruction)
        output_prefix = result.get("output_prefix", output_prefix)
        format_guidance = result.get("format_guidance", format_guidance)

        cli_utils.CONSOLE.print(
            Panel(
                f"[bold]Instruction:[/bold]\n{instruction}\n\n"
                f"[bold]Output prefix:[/bold] {output_prefix}\n\n"
                f"[bold]Format guidance:[/bold] {format_guidance}",
                title=f"[magenta]Revised Answer Template (v{iteration + 2})[/magenta]",
                border_style="magenta",
            )
        )

        iteration += 1

    cli_utils.CONSOLE.print("[green]✓ Answer template finalized[/green]")

    # Build final template and postprocessing config
    final_template = f"{instruction}\n\nFormat your response as:\n{output_prefix} <your response>"
    postprocessing = {
        "cut_prefix": output_prefix,
        "strip_whitespace": True,
    }

    return final_template, postprocessing


def _show_role_guidance(role: str):
    """Display contextual help for a file role.

    Args:
        role: The role to show guidance for (primary, reference, rules, examples).
    """
    guidance = FILE_ROLE_GUIDANCE.get(role)
    if not guidance:
        return

    help_text = "\n".join(f"  • {h}" for h in guidance.get("help", []))

    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]{guidance['title']}[/bold] - {guidance['description']}\n\n"
            f"[dim]{help_text}[/dim]",
            border_style="blue",
        )
    )


def _prompt_file_roles(files: list[dict]) -> dict[str, Path]:
    """Interactive prompts to assign file roles.

    Args:
        files: List of detected files.

    Returns:
        Dict mapping role names to file paths.
    """
    roles = {}
    file_nums = [str(i) for i in range(1, len(files) + 1)]

    # Primary data (required)
    _show_role_guidance("primary")
    primary_idx = IntPrompt.ask(
        "\nWhich file is your [bold]PRIMARY[/bold] data?",
        choices=file_nums,
        default="1",
    )
    roles["primary"] = files[int(primary_idx) - 1]["path"]

    # Get remaining file numbers
    remaining_nums = [n for n in file_nums if n != str(primary_idx)]

    if remaining_nums:
        # Reference data (optional)
        _show_role_guidance("reference")
        if Confirm.ask("\nDo you have [bold]reference data[/bold]?", default=False):
            ref_idx = Prompt.ask(
                "Which file?",
                choices=remaining_nums,
            )
            roles["reference"] = files[int(ref_idx) - 1]["path"]
            remaining_nums = [n for n in remaining_nums if n != ref_idx]

    if remaining_nums:
        # Rules/guidelines (optional)
        _show_role_guidance("rules")
        if Confirm.ask("\nDo you have [bold]rules or guidelines[/bold]?", default=False):
            rules_idx = Prompt.ask(
                "Which file?",
                choices=remaining_nums,
            )
            roles["rules"] = files[int(rules_idx) - 1]["path"]
            remaining_nums = [n for n in remaining_nums if n != rules_idx]

    if remaining_nums:
        # Labeled examples (optional)
        _show_role_guidance("examples")
        if Confirm.ask("\nDo you have [bold]labeled examples[/bold]?", default=False):
            examples_idx = Prompt.ask(
                "Which file?",
                choices=remaining_nums,
            )
            roles["examples"] = files[int(examples_idx) - 1]["path"]

    return roles


def _display_multi_file_analysis(analysis):
    """Display results of multi-file analysis.

    Args:
        analysis: MultiFileAnalysis object with relationships and suggestions.
    """
    # Purpose
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Primary Purpose:[/bold] {analysis.primary_purpose}",
            title="[cyan]Multi-File Analysis[/cyan]",
            border_style="cyan",
        )
    )

    # Relationships
    if analysis.relationships:
        rel_table = Table(title="File Relationships", show_edge=False)
        rel_table.add_column("From", style="cyan")
        rel_table.add_column("To", style="green")
        rel_table.add_column("Type", style="yellow")

        for rel in analysis.relationships:
            from_str = f"{rel.get('from_file', '?')}.{rel.get('from_col', '?')}"
            to_str = f"{rel.get('to_file', '?')}.{rel.get('to_col', '?')}"
            rel_table.add_row(from_str, to_str, rel.get("type", "?"))

        cli_utils.CONSOLE.print(rel_table)

    # Extracted rules
    if analysis.extracted_rules:
        rules_text = "\n".join(f"  • {r}" for r in analysis.extracted_rules[:5])
        cli_utils.CONSOLE.print(
            Panel(
                f"[dim]{rules_text}[/dim]",
                title="[yellow]Extracted Rules[/yellow]",
                border_style="yellow",
            )
        )

    # Quality patterns
    if analysis.quality_patterns:
        good_patterns = analysis.quality_patterns.get("good", [])
        bad_patterns = analysis.quality_patterns.get("bad", [])

        patterns_text = ""
        if good_patterns:
            patterns_text += "[green]Good patterns:[/green]\n"
            patterns_text += "\n".join(f"  ✓ {p}" for p in good_patterns[:3])
        if bad_patterns:
            if patterns_text:
                patterns_text += "\n\n"
            patterns_text += "[red]Bad patterns:[/red]\n"
            patterns_text += "\n".join(f"  ✗ {p}" for p in bad_patterns[:3])

        if patterns_text:
            cli_utils.CONSOLE.print(
                Panel(patterns_text, title="Quality Patterns", border_style="dim")
            )

    # Suggested pipeline
    if analysis.suggested_pipeline:
        pipeline = analysis.suggested_pipeline
        pipeline_text = []

        if pipeline.get("synth", {}).get("enabled"):
            synth_info = pipeline["synth"]
            pipeline_text.append(
                f"[cyan]1. Synth:[/cyan] {synth_info.get('goal', 'qa')} "
                f"from {synth_info.get('source_file', 'primary data')}"
            )

        if pipeline.get("judge", {}).get("enabled"):
            judge_info = pipeline["judge"]
            pipeline_text.append(
                f"[cyan]2. Judge:[/cyan] {judge_info.get('type', 'generic')} "
                f"evaluation"
            )

        if pipeline.get("train", {}).get("enabled"):
            train_info = pipeline["train"]
            pipeline_text.append(
                f"[cyan]3. Train:[/cyan] {train_info.get('task', 'fine-tuning')}"
            )

        if pipeline_text:
            cli_utils.CONSOLE.print(
                Panel(
                    "\n".join(pipeline_text),
                    title="[magenta]Suggested Pipeline[/magenta]",
                    border_style="magenta",
                )
            )


def _display_annotated_synth_config(
    config, config_path: Path, schema, goal: str, verbose: bool = False
):
    """Display synth config with helpful annotations."""
    # Simple output for non-verbose mode
    if not verbose:
        cli_utils.CONSOLE.print(
            f"\n[green]Created:[/green] {config_path}"
        )
        cli_utils.CONSOLE.print(
            f"  [dim]Strategy: {goal} | Samples: {config.num_samples} | "
            f"Model: {config.inference_config.model.model_name}[/dim]"
        )
        return

    from rich.syntax import Syntax

    # Build annotated config display
    annotations = []

    # Input data section
    annotations.append(
        "[bold cyan]# INPUT DATA[/bold cyan]\n"
        f"[dim]# Your data file: {schema.source_path}[/dim]\n"
        f"[dim]# Rows: {schema.row_count}, Columns: {len(schema.columns)}[/dim]"
    )

    # Strategy section with detailed explanations
    strategy_details = {
        "qa": (
            "Generates question-answer pairs from your content",
            "The LLM reads each row of your data and:\n"
            "#   1. Creates a relevant question about the content\n"
            "#   2. Generates an accurate answer based on the source\n"
            "#   Output: {question, answer, context} for each sample"
        ),
        "conversation": (
            "Creates multi-turn chat dialogues",
            "The LLM reads each row and generates a realistic conversation:\n"
            "#   1. User asks about the content\n"
            "#   2. Assistant responds helpfully\n"
            "#   3. May include follow-up turns\n"
            "#   Output: {messages: [{role, content}, ...]} for each sample"
        ),
        "augmentation": (
            "Produces variations of existing examples",
            "The LLM takes your existing data and creates variations:\n"
            "#   1. Rephrases while preserving meaning\n"
            "#   2. Changes style, tone, or perspective\n"
            "#   3. Maintains factual accuracy\n"
            "#   Output: Same format as input, with varied content"
        ),
        "instruction": (
            "Generates instruction-following data",
            "The LLM creates task instructions with outputs:\n"
            "#   1. Generates a clear instruction/task\n"
            "#   2. Provides the expected output\n"
            "#   3. Teaches the model to follow procedures\n"
            "#   Output: {instruction, input, output} for each sample"
        ),
    }
    summary, details = strategy_details.get(
        goal, ("Custom synthesis", "Generates examples based on your data")
    )
    annotations.append(
        f"\n[bold cyan]# SYNTHESIS STRATEGY[/bold cyan]\n"
        f"[dim]# Goal: {goal} - {summary}[/dim]\n"
        f"[dim]# {details}[/dim]"
    )

    # Model section
    annotations.append(
        "\n[bold cyan]# LLM MODEL[/bold cyan]\n"
        f"[dim]# Using: {config.inference_config.model.model_name}[/dim]\n"
        "[dim]# Change 'engine' to VLLM/LLAMACPP for local models[/dim]\n"
        "[dim]# Change 'model_name' to use a different model[/dim]"
    )

    # Generation params
    annotations.append(
        "\n[bold cyan]# GENERATION SETTINGS[/bold cyan]\n"
        f"[dim]# num_samples: {config.num_samples} examples to generate[/dim]\n"
        f"[dim]# temperature: {config.inference_config.generation.temperature} "
        f"(higher = more creative, lower = more consistent)[/dim]\n"
        f"[dim]# output_path: {config.output_path}[/dim]"
    )

    # Field mappings
    if config.strategy_params and config.strategy_params.input_data:
        input_data = config.strategy_params.input_data[0]
        if hasattr(input_data, "attribute_map") and input_data.attribute_map:
            mapping_str = ", ".join(
                f"{k} -> {{{v}}}" for k, v in input_data.attribute_map.items()
            )
            annotations.append(
                "\n[bold cyan]# FIELD MAPPINGS[/bold cyan]\n"
                f"[dim]# Your columns mapped to placeholders: {mapping_str}[/dim]\n"
                "[dim]# These placeholders are used in the prompts below[/dim]"
            )

    cli_utils.CONSOLE.print(
        Panel(
            "\n".join(annotations),
            title=f"[green]Config: {config_path}[/green]",
            border_style="green",
        )
    )

    # Show key editable sections
    cli_utils.CONSOLE.print("\n[bold]Key sections you can customize:[/bold]")
    cli_utils.CONSOLE.print(
        "  [cyan]inference_config.model.model_name[/cyan] - Change the LLM\n"
        "  [cyan]inference_config.engine[/cyan] - ANTHROPIC, OPENAI, VLLM, LLAMACPP\n"
        "  [cyan]inference_config.generation.temperature[/cyan] - Creativity (0.0-1.0)\n"
        "  [cyan]num_samples[/cyan] - Number of examples to generate\n"
        "  [cyan]strategy_params.generated_attributes[/cyan] - Modify prompts"
    )


def _display_annotated_judge_config(
    config, config_path: Path, schema, judge_type: str, verbose: bool = False
):
    """Display judge config with helpful annotations."""
    # Simple output for non-verbose mode
    if not verbose:
        cli_utils.CONSOLE.print(
            f"\n[green]Created:[/green] {config_path}"
        )
        cli_utils.CONSOLE.print(
            f"  [dim]Type: {judge_type} | "
            f"Model: {config.inference_config.model.model_name}[/dim]"
        )
        return

    judge_descriptions = {
        "generic": "Evaluates overall quality, coherence, and helpfulness",
        "compliance": "Checks if responses follow specific guidelines",
        "relevance": "Measures how well answers address questions",
        "safety": "Detects harmful or inappropriate content",
        "groundedness": "Verifies claims are supported by context",
    }

    annotations = [
        "[bold cyan]# JUDGE CONFIGURATION[/bold cyan]\n"
        f"[dim]# Type: {judge_type} - {judge_descriptions.get(judge_type, '')}[/dim]",
        "\n[bold cyan]# INPUT[/bold cyan]\n"
        f"[dim]# Data to evaluate: {schema.source_path}[/dim]",
        "\n[bold cyan]# LLM MODEL[/bold cyan]\n"
        f"[dim]# Judge model: {config.inference_config.model.model_name}[/dim]\n"
        "[dim]# The judge LLM scores each example based on criteria[/dim]",
        "\n[bold cyan]# EVALUATION CRITERIA[/bold cyan]\n"
        f"[dim]# Built-in criteria for '{judge_type}' evaluation[/dim]\n"
        "[dim]# Edit 'judge_params.evaluation_criteria' to customize[/dim]",
    ]

    cli_utils.CONSOLE.print(
        Panel(
            "\n".join(annotations),
            title=f"[green]Config: {config_path}[/green]",
            border_style="green",
        )
    )

    cli_utils.CONSOLE.print("\n[bold]Key sections you can customize:[/bold]")
    cli_utils.CONSOLE.print(
        "  [cyan]inference_config.model.model_name[/cyan] - Change the judge LLM\n"
        "  [cyan]judge_params.evaluation_criteria[/cyan] - Custom scoring criteria\n"
        "  [cyan]judge_params.score_threshold[/cyan] - Min score to pass"
    )


def _display_annotated_train_config(
    config, config_path: Path, base_model: str, use_lora: bool, verbose: bool = False
):
    """Display training config with helpful annotations."""
    # Simple output for non-verbose mode
    if not verbose:
        cli_utils.CONSOLE.print(
            f"\n[green]Created:[/green] {config_path}"
        )
        method = "LoRA" if use_lora else "Full"
        cli_utils.CONSOLE.print(
            f"  [dim]Model: {base_model} | Method: {method} | "
            f"Steps: {config.training.max_steps}[/dim]"
        )
        return

    # Get dataset name safely
    dataset_name = "your_data.jsonl"
    if config.data and config.data.train and config.data.train.datasets:
        ds = config.data.train.datasets[0]
        if hasattr(ds, "dataset_name") and ds.dataset_name:
            dataset_name = ds.dataset_name

    annotations = [
        "[bold cyan]# MODEL[/bold cyan]\n"
        f"[dim]# Base model: {base_model}[/dim]\n"
        "[dim]# The pre-trained model to fine-tune[/dim]",
        "\n[bold cyan]# TRAINING METHOD[/bold cyan]\n"
        f"[dim]# LoRA enabled: {use_lora}[/dim]\n"
        + (
            "[dim]# LoRA trains small adapter weights (efficient, preserves base model)[/dim]"
            if use_lora
            else "[dim]# Full fine-tuning (modifies all weights, needs more memory)[/dim]"
        ),
        "\n[bold cyan]# TRAINING DATA[/bold cyan]\n"
        f"[dim]# Dataset: {dataset_name}[/dim]\n"
        "[dim]# Format: conversation (chat format with user/assistant turns)[/dim]",
        "\n[bold cyan]# HYPERPARAMETERS[/bold cyan]\n"
        f"[dim]# max_steps: {config.training.max_steps} (total training iterations)[/dim]\n"
        f"[dim]# learning_rate: {config.training.learning_rate}[/dim]\n"
        f"[dim]# batch_size: {config.training.per_device_train_batch_size}[/dim]",
        "\n[bold cyan]# OUTPUT[/bold cyan]\n"
        f"[dim]# Model saved to: {config.training.output_dir}[/dim]\n"
        "[dim]# Checkpoints saved during training[/dim]",
    ]

    cli_utils.CONSOLE.print(
        Panel(
            "\n".join(annotations),
            title=f"[green]Config: {config_path}[/green]",
            border_style="green",
        )
    )

    cli_utils.CONSOLE.print("\n[bold]Key sections you can customize:[/bold]")
    cli_utils.CONSOLE.print(
        "  [cyan]model.model_name[/cyan] - Change base model\n"
        "  [cyan]training.max_steps[/cyan] - More steps = more learning\n"
        "  [cyan]training.learning_rate[/cyan] - Speed of learning (default works well)\n"
        "  [cyan]training.per_device_train_batch_size[/cyan] - Increase if GPU has memory\n"
        "  [cyan]peft.lora_r[/cyan] - LoRA rank (higher = more capacity, more memory)"
    )


def wizard(
    ctx: typer.Context,
    data: Annotated[
        str,
        typer.Option(
            "--data",
            "-d",
            help="Path to your data file or directory (CSV, JSON, Excel, or Word).",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save generated configs.",
        ),
    ] = "./oumi_configs",
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Use LLM to analyze data and infer domain-specific config.",
        ),
    ] = False,
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            "-e",
            help="LLM inference engine to use (ANTHROPIC, OPENAI, DEEPSEEK, TOGETHER).",
        ),
    ] = "ANTHROPIC",
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name to use. If not specified, uses default for the engine.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed output including AI explanations and extra context.",
        ),
    ] = False,
):
    """Interactive wizard to guide you through Oumi setup.

    This wizard analyzes your data and helps you create configurations
    for synthesis, evaluation, and training.

    Examples:
        # Single file
        oumi onboard wizard --data ./my_data.csv

        # Directory with multiple files
        oumi onboard wizard --data ./customer_data/

        # With AI analysis using different engines
        oumi onboard wizard --data ./data/ --llm --engine ANTHROPIC
        oumi onboard wizard --data ./data/ --llm --engine OPENAI --model gpt-4o

        # Show detailed output
        oumi onboard wizard --data ./data/ --llm --verbose
    """
    # Delayed imports
    from oumi.onboarding import DataAnalyzer, FieldMapper
    from oumi.onboarding.config_builder import (
        JudgeConfigBuilder,
        SynthConfigBuilder,
        TrainConfigBuilder,
    )

    # Welcome message - concise by default
    if verbose:
        cli_utils.CONSOLE.print(
            Panel(
                "[bold green]Welcome to the Oumi Onboarding Wizard![/bold green]\n\n"
                "This wizard will help you create configurations for:\n"
                "  [cyan]oumi synth[/cyan]  - Generate synthetic training data\n"
                "  [cyan]oumi judge[/cyan]  - Evaluate and score data quality\n"
                "  [cyan]oumi train[/cyan]  - Fine-tune language models\n\n"
                "[dim]The wizard will analyze your data and suggest the best options.\n"
                "You can accept defaults or customize each setting.[/dim]",
                title="Oumi Onboard",
                border_style="green",
            )
        )
    else:
        cli_utils.CONSOLE.print(
            "[bold green]Oumi Onboarding Wizard[/bold green] "
            "[dim](use --verbose for detailed output)[/dim]\n"
        )

    # Check if input is a directory or file
    data_path = Path(data)
    if not data_path.exists():
        cli_utils.CONSOLE.print(f"[red]Error: Path not found: {data}[/red]")
        raise typer.Exit(1)

    analyzer = DataAnalyzer()
    file_roles = {}
    multi_file_analysis = None
    column_assignments = None  # For column-level role assignment

    # Initialize LLM analyzer early if needed
    llm_analyzer_instance = None
    if use_llm:
        # Validate engine choice
        valid_engines = ["ANTHROPIC", "OPENAI", "DEEPSEEK", "TOGETHER"]
        engine_upper = engine.upper()
        if engine_upper not in valid_engines:
            cli_utils.CONSOLE.print(
                f"[yellow]Warning: Unknown engine '{engine}'. "
                f"Valid options: {', '.join(valid_engines)}. Using ANTHROPIC.[/yellow]"
            )
            engine_upper = "ANTHROPIC"

        try:
            from oumi.onboarding.llm_analyzer import FileContext, LLMAnalyzer

            llm_analyzer_instance = LLMAnalyzer(
                engine=engine_upper,
                model=model,
            )
            cli_utils.CONSOLE.print(
                f"[dim]Using {engine_upper} engine"
                f"{f' with model {model}' if model else ''}[/dim]\n"
            )
        except ImportError as e:
            cli_utils.CONSOLE.print(
                f"[yellow]Warning: Could not load LLM analyzer: {e}[/yellow]\n"
                "[dim]Make sure you have the required API key set.[/dim]"
            )
            use_llm = False
        except Exception as e:
            cli_utils.CONSOLE.print(
                f"[yellow]Warning: LLM initialization failed: {e}[/yellow]"
            )
            use_llm = False

    # Handle directory input (multi-file mode)
    if data_path.is_dir():
        cli_utils.CONSOLE.print(
            "\n[bold cyan]Step 1/5: Scanning directory...[/bold cyan]"
        )

        files = _detect_files_in_directory(data_path)
        if not files:
            cli_utils.CONSOLE.print(
                f"[red]Error: No supported files found in {data_path}[/red]\n"
                f"[dim]Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}[/dim]"
            )
            raise typer.Exit(1)

        cli_utils.CONSOLE.print(f"[green]Found {len(files)} file(s)[/green]")

        # AI analysis of each file if LLM enabled
        if use_llm and llm_analyzer_instance:
            with cli_utils.CONSOLE.status(
                "[dim]Analyzing files with AI...[/dim]", spinner="dots"
            ):
                files = _analyze_file_purposes(files, analyzer, llm_analyzer_instance)

            # Display files with AI analysis - verbose shows full table
            if verbose:
                _display_file_listing(files, analyzer, show_ai_analysis=True)

            # Show AI-suggested role assignments
            cli_utils.CONSOLE.print("\n[bold]File Roles:[/bold]")
            for f in files:
                role = f.get("suggested_role", "unknown").upper()
                reason = f.get("role_reason") or ""
                if verbose:
                    cli_utils.CONSOLE.print(
                        f"  [cyan]{f['name']}[/cyan] → [magenta]{role}[/magenta]"
                    )
                    if reason:
                        cli_utils.CONSOLE.print(f"    [dim]{reason}[/dim]")
                else:
                    cli_utils.CONSOLE.print(
                        f"  [cyan]{f['name']}[/cyan] → [magenta]{role}[/magenta]"
                    )

            # Check if we have tabular files for column-level assignment
            has_tabular_files = any(
                f["extension"] in TABULAR_EXTENSIONS for f in files
            )

            # Ask what level of assignment - simplified prompt
            cli_utils.CONSOLE.print("")
            if has_tabular_files:
                assignment_mode = Prompt.ask(
                    "Assignment mode",
                    choices=["ai", "file", "column"],
                    default="ai",
                )
                if verbose:
                    cli_utils.CONSOLE.print(
                        "[dim]  ai     = Use AI-suggested file roles\n"
                        "  file   = Manually assign entire files to roles\n"
                        "  column = Pick specific columns from tabular files[/dim]"
                    )
            else:
                use_ai_roles = Confirm.ask(
                    "Use AI-suggested file roles?",
                    default=True,
                )
                assignment_mode = "ai" if use_ai_roles else "file"

            column_assignments = None

            if assignment_mode == "ai":
                # Build file_roles from AI suggestions
                file_roles = {}
                for f in files:
                    role = f.get("suggested_role", "context")
                    if role == "primary" and "primary" not in file_roles:
                        file_roles["primary"] = f["path"]
                    elif role == "reference" and "reference" not in file_roles:
                        file_roles["reference"] = f["path"]
                    elif role == "rules" and "rules" not in file_roles:
                        file_roles["rules"] = f["path"]
                    elif role == "examples" and "examples" not in file_roles:
                        file_roles["examples"] = f["path"]

                # Ensure we have a primary
                if "primary" not in file_roles and files:
                    file_roles["primary"] = files[0]["path"]

            elif assignment_mode == "column":
                # Column-level assignment for tabular files
                column_assignments = _prompt_column_roles(
                    files, analyzer, llm_analyzer_instance, verbose=verbose
                )
                # Extract file_roles from column_assignments
                file_roles = {}
                if "primary" in column_assignments:
                    file_roles["primary"] = column_assignments["primary"]["path"]
                if "rules" in column_assignments:
                    file_roles["rules"] = column_assignments["rules"]["path"]

            else:
                # Manual file-level role assignment
                cli_utils.CONSOLE.print(
                    "\n[bold cyan]Manual File Role Assignment[/bold cyan]\n"
                )
                file_roles = _prompt_file_roles(files)
        else:
            # No LLM - display basic listing and manual assignment
            with cli_utils.CONSOLE.status(
                "[green]Analyzing files...[/green]", spinner="dots"
            ):
                _display_file_listing(files, analyzer)

            # Check if we have tabular files for column-level assignment
            has_tabular_files = any(
                f["extension"] in TABULAR_EXTENSIONS for f in files
            )

            column_assignments = None

            if has_tabular_files:
                cli_utils.CONSOLE.print(
                    "\n[bold cyan]Assignment Mode[/bold cyan]\n"
                    "[dim]You can assign entire files to roles, or pick specific columns.[/dim]\n"
                )
                assignment_mode = Prompt.ask(
                    "How would you like to assign data?",
                    choices=["file", "column"],
                    default="file",
                )
                cli_utils.CONSOLE.print(
                    "[dim]  file   = Assign entire files to roles\n"
                    "  column = Pick specific columns from tabular files[/dim]\n"
                )

                if assignment_mode == "column":
                    column_assignments = _prompt_column_roles(
                        files, analyzer, None, verbose=verbose
                    )
                    # Extract file_roles from column_assignments
                    file_roles = {}
                    if "primary" in column_assignments:
                        file_roles["primary"] = column_assignments["primary"]["path"]
                    if "rules" in column_assignments:
                        file_roles["rules"] = column_assignments["rules"]["path"]
                else:
                    cli_utils.CONSOLE.print(
                        "\n[bold cyan]Identify File Roles[/bold cyan]\n"
                        "[dim]Help us understand how each file should be used.[/dim]\n"
                    )
                    file_roles = _prompt_file_roles(files)
            else:
                cli_utils.CONSOLE.print(
                    "\n[bold cyan]Identify File Roles[/bold cyan]\n"
                    "[dim]Help us understand how each file should be used.[/dim]\n"
                )
                file_roles = _prompt_file_roles(files)

        # Use primary file as the main schema
        primary_path = file_roles["primary"]
        with cli_utils.CONSOLE.status(
            "[green]Analyzing primary data...[/green]", spinner="dots"
        ):
            try:
                schema = analyzer.analyze(primary_path)
            except Exception as e:
                cli_utils.CONSOLE.print(f"[red]Error analyzing primary data: {e}[/red]")
                raise typer.Exit(1)

        # Display schema info for primary file
        _display_schema_info(schema)

        # Multi-file LLM analysis if enabled and multiple roles assigned
        if use_llm and llm_analyzer_instance and len(file_roles) > 1:
            cli_utils.CONSOLE.print(
                "\n[bold cyan]AI Analysis: Analyzing file relationships...[/bold cyan]"
            )
            try:
                from oumi.onboarding.llm_analyzer import FileContext

                # Build file contexts
                file_contexts = []
                for role, path in file_roles.items():
                    ctx_schema = None
                    try:
                        ctx_schema = analyzer.analyze(path)
                    except Exception:
                        pass

                    file_contexts.append(
                        FileContext(
                            path=str(path),
                            role=role,
                            schema=ctx_schema,
                            summary=f"{role} data file",
                        )
                    )

                with cli_utils.CONSOLE.status(
                    "[green]Analyzing relationships between files...[/green]",
                    spinner="dots",
                ):
                    multi_file_analysis = llm_analyzer_instance.analyze_multi_file(file_contexts)

                _display_multi_file_analysis(multi_file_analysis)

            except Exception as e:
                cli_utils.CONSOLE.print(
                    f"[yellow]Warning: Multi-file analysis failed: {e}[/yellow]\n"
                    "[dim]Continuing with single-file config generation.[/dim]"
                )

    else:
        # Single file mode (original behavior)
        cli_utils.CONSOLE.print(
            "\n[bold cyan]Step 1/5: Analyzing your data...[/bold cyan]"
        )

        with cli_utils.CONSOLE.status("[green]Analyzing...[/green]", spinner="dots"):
            try:
                schema = analyzer.analyze(data_path)
            except Exception as e:
                cli_utils.CONSOLE.print(f"[red]Error analyzing data: {e}[/red]")
                raise typer.Exit(1)

        # Display schema info
        _display_schema_info(schema)

    # Optional: LLM-based domain analysis (reuse llm_analyzer_instance if available)
    domain = None
    llm_analyzer = llm_analyzer_instance  # Use the instance created earlier if any
    if use_llm and llm_analyzer is not None:
        cli_utils.CONSOLE.print(
            "\n[bold cyan]AI Analysis: Analyzing your data with Claude...[/bold cyan]"
        )
        try:
            with cli_utils.CONSOLE.status(
                "[green]Analyzing domain and terminology...[/green]", spinner="dots"
            ):
                domain = llm_analyzer.analyze(schema)

            # Display domain analysis results
            _display_domain_analysis(domain)

        except Exception as e:
            cli_utils.CONSOLE.print(
                f"[yellow]Warning: LLM analysis failed: {e}[/yellow]\n"
                "[dim]Continuing with template-based config generation.[/dim]"
            )

    # Step 2: Select goal
    cli_utils.CONSOLE.print("\n[bold cyan]Step 2/5: What would you like to do?[/bold cyan]")
    cli_utils.CONSOLE.print(
        Panel(
            "[bold white][1] Generate synthetic training data[/bold white] (oumi synth)\n"
            "    [dim]Create new training examples from your data using an LLM.\n"
            "    Best for: Expanding small datasets, creating Q&A pairs, augmenting conversations.[/dim]\n\n"
            "[bold white][2] Evaluate/judge data quality[/bold white] (oumi judge)\n"
            "    [dim]Score and filter your data based on quality criteria.\n"
            "    Best for: Quality control, compliance checking, filtering bad examples.[/dim]\n\n"
            "[bold white][3] Train a model[/bold white] (oumi train)\n"
            "    [dim]Fine-tune a language model on your data.\n"
            "    Best for: Creating a custom model for your specific use case.[/dim]\n\n"
            "[bold white][4] Full pipeline: synth -> judge -> train[/bold white]\n"
            "    [dim]Run all three steps in sequence.\n"
            "    Best for: End-to-end workflow from raw data to trained model.[/dim]",
            title="Choose your goal",
            border_style="blue",
        )
    )

    choice = IntPrompt.ask("\nSelect an option", choices=["1", "2", "3", "4"], default="1")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 3-4: Configure based on choice
    if choice == 1:
        commands = _wizard_synth(
            schema, output_path, analyzer, domain, llm_analyzer, file_roles,
            column_assignments=column_assignments, verbose=verbose,
        )
    elif choice == 2:
        commands = _wizard_judge(schema, output_path, domain, llm_analyzer, verbose=verbose)
    elif choice == 3:
        commands = _wizard_train(schema, output_path, verbose=verbose)
    elif choice == 4:
        commands = _wizard_pipeline(
            schema, output_path, analyzer, domain, llm_analyzer, file_roles,
            column_assignments=column_assignments, verbose=verbose,
        )

    # Step 5: Show runnable command(s)
    cli_utils.CONSOLE.print("\n[bold cyan]Step 5/5: Ready to run![/bold cyan]")

    if len(commands) == 1:
        cli_utils.CONSOLE.print(
            Panel(
                f"[bold white]{commands[0]}[/bold white]",
                title="[green]Run this command[/green]",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        commands_text = "\n".join(
            f"[bold white]{i+1}. {cmd}[/bold white]" for i, cmd in enumerate(commands)
        )
        cli_utils.CONSOLE.print(
            Panel(
                commands_text,
                title="[green]Run these commands in order[/green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    cli_utils.CONSOLE.print(f"\n[dim]Configs saved to: {output_path}[/dim]")

    # Show prerequisites based on what was configured
    prereqs = []
    if choice in [1, 2, 4]:  # synth, judge, or pipeline
        prereqs.append(
            "[yellow]Synth/Judge:[/yellow] Set ANTHROPIC_API_KEY or OPENAI_API_KEY env var"
        )
    if choice in [3, 4]:  # train or pipeline
        prereqs.append(
            "[yellow]Training:[/yellow] Requires GPU with sufficient VRAM (see model selection)"
        )

    if prereqs:
        cli_utils.CONSOLE.print(
            Panel(
                "\n".join(prereqs) + "\n\n"
                "[dim]To use a local model for synth, edit the config and change:\n"
                "  inference_config.engine: VLLM\n"
                "  inference_config.model.model_name: <local-model-path>[/dim]",
                title="Prerequisites",
                border_style="yellow",
            )
        )


def _display_domain_analysis(domain):
    """Display LLM-inferred domain analysis."""
    terminology_str = ", ".join(domain.terminology[:8]) if domain.terminology else "None detected"
    quality_str = "\n    ".join(f"- {s}" for s in domain.quality_signals[:4]) if domain.quality_signals else "- General quality"
    issues_str = "\n    ".join(f"- {s}" for s in domain.common_issues[:4]) if domain.common_issues else "- No specific issues"

    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Domain:[/bold] [green]{domain.domain}[/green]\n"
            f"[bold]Description:[/bold] {domain.description}\n\n"
            f"[bold]Key Terminology:[/bold]\n    {terminology_str}\n\n"
            f"[bold]Quality Signals:[/bold]\n    {quality_str}\n\n"
            f"[bold]Common Issues to Watch:[/bold]\n    {issues_str}",
            title="[cyan]AI Analysis of Your Data[/cyan]",
            border_style="cyan",
        )
    )

    if domain.suggested_persona:
        cli_utils.CONSOLE.print(
            f"\n[dim]Suggested AI persona: {domain.suggested_persona[:150]}...[/dim]"
        )


def _display_schema_info(schema):
    """Display analyzed schema information."""
    from oumi.onboarding.data_analyzer import DataSchema

    table = Table(title=f"Data Analysis: {schema.source_path}", show_edge=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Format", schema.detected_format)
    table.add_row("Rows", str(schema.row_count))
    table.add_row("Columns", str(len(schema.columns)))

    if schema.conversation_columns:
        table.add_row("Conversation cols", ", ".join(schema.conversation_columns))
    if schema.text_columns:
        table.add_row("Text cols", ", ".join(schema.text_columns[:5]))
    if schema.categorical_columns:
        table.add_row("Categorical cols", ", ".join(schema.categorical_columns[:5]))

    cli_utils.CONSOLE.print(table)

    # Show column details
    if schema.columns:
        col_table = Table(title="Column Details", show_edge=False)
        col_table.add_column("Column", style="cyan")
        col_table.add_column("Type", style="yellow")
        col_table.add_column("Characteristics", style="green")

        for col in schema.columns[:10]:  # Show first 10 columns
            chars = []
            if col.is_text:
                chars.append("text")
            if col.is_conversation:
                chars.append("conversation")
            if col.is_categorical:
                chars.append("categorical")
            col_table.add_row(col.name, col.dtype, ", ".join(chars) or "-")

        cli_utils.CONSOLE.print(col_table)


def _wizard_synth(
    schema, output_path: Path, analyzer, domain=None, llm_analyzer=None, file_roles=None,
    column_assignments=None, verbose: bool = False,
):
    """Configure synthesis.

    Args:
        schema: DataSchema for the primary data file.
        output_path: Output directory for generated configs.
        analyzer: DataAnalyzer instance.
        domain: Optional DomainAnalysis from LLM analysis.
        llm_analyzer: Optional LLMAnalyzer instance.
        file_roles: Optional dict mapping roles to file paths.
        column_assignments: Optional dict with column-level role assignments.
            Format: {"context": {"path": Path, "column": str}, "metadata": [...], ...}
    """
    from oumi.onboarding.config_builder import SynthConfigBuilder

    cli_utils.CONSOLE.print("\n[bold cyan]Step 3/5: Configuring synthesis...[/bold cyan]")

    # Suggest goal based on data
    suggested_goal = analyzer.suggest_goal(schema)
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Recommended: [green]{suggested_goal}[/green][/bold] "
            f"(based on your data structure)\n\n"
            "[bold white][1] qa[/bold white] - Generate Question-Answer Pairs\n"
            "    [dim]Reads your content and creates questions about it,\n"
            "    then generates accurate answers based on the source material.\n"
            "    Output: {question, answer, context}[/dim]\n\n"
            "[bold white][2] conversation[/bold white] - Generate Multi-Turn Dialogues\n"
            "    [dim]Creates realistic back-and-forth conversations\n"
            "    with multiple turns between user and assistant.\n"
            "    Output: {messages: [{role, content}, ...]}[/dim]\n\n"
            "[bold white][3] augmentation[/bold white] - Create Variations of Existing Data\n"
            "    [dim]Takes your existing examples and creates new variations\n"
            "    that preserve meaning but change wording, style, or perspective.\n"
            "    Output: Same format as input, with variations[/dim]\n\n"
            "[bold white][4] instruction[/bold white] - Generate Instruction-Following Data\n"
            "    [dim]Creates task instructions paired with correct outputs,\n"
            "    teaching the model to follow specific guidelines or procedures.\n"
            "    Output: {instruction, input, output}[/dim]",
            title="Synthesis Goal - What should the LLM generate?",
            border_style="blue",
        )
    )

    goal_map = {"1": "qa", "2": "conversation", "3": "augmentation", "4": "instruction"}
    goal_choice = Prompt.ask(
        "\nSelect synthesis goal",
        choices=["1", "2", "3", "4"],
        default=str(list(goal_map.values()).index(suggested_goal) + 1),
    )
    goal = goal_map[goal_choice]

    cli_utils.CONSOLE.print(
        f"\n[dim]Tip: Start with a small number (10-50) to verify quality, "
        f"then scale up.[/dim]"
    )
    num_samples = IntPrompt.ask("Number of samples to generate", default=100)

    cli_utils.CONSOLE.print("\n[bold cyan]Step 4/5: Generating configuration...[/bold cyan]")

    builder = SynthConfigBuilder()

    # Build attribute_map from column_assignments if provided
    attribute_map = None
    if column_assignments:
        attribute_map = {}
        if "context" in column_assignments:
            attribute_map["context"] = column_assignments["context"]["column"]
        if "question" in column_assignments:
            attribute_map["question"] = column_assignments["question"]["column"]
        if "answer" in column_assignments:
            attribute_map["answer"] = column_assignments["answer"]["column"]
        if "reference_values" in column_assignments:
            attribute_map["reference"] = column_assignments["reference_values"]["column"]
        if "label" in column_assignments:
            attribute_map["label"] = column_assignments["label"]["column"]
        # Handle metadata as list of columns
        if "metadata" in column_assignments and isinstance(column_assignments["metadata"], list):
            metadata_cols = [m["column"] for m in column_assignments["metadata"]]
            attribute_map["metadata"] = metadata_cols

        if attribute_map:
            cli_utils.CONSOLE.print(
                f"[dim]Using column mappings: {attribute_map}[/dim]"
            )

    # Use LLM-inferred config if domain analysis is available
    if domain is not None and llm_analyzer is not None:
        # Offer interactive prompt building
        cli_utils.CONSOLE.print(
            Panel(
                "[bold]Configuration Mode[/bold]\n\n"
                "[cyan][1][/cyan] [bold]Quick[/bold] - Auto-generate prompts based on your data\n"
                "    [dim]AI creates everything automatically. Good for fast iteration.[/dim]\n\n"
                "[cyan][2][/cyan] [bold]Interactive[/bold] - Build prompts step by step with AI\n"
                "    [dim]You'll refine system prompt, questions, and answers together.\n"
                "    Best for high-quality, customized outputs.[/dim]",
                title="Choose Configuration Mode",
                border_style="magenta",
            )
        )

        mode_choice = Prompt.ask(
            "Select mode",
            choices=["1", "2"],
            default="1",
        )

        if mode_choice == "2":
            # Interactive prompt building mode
            cli_utils.CONSOLE.print(
                "\n[bold magenta]Interactive Prompt Building[/bold magenta]\n"
                "[dim]We'll work together to create the perfect prompts for your use case.[/dim]"
            )

            # Step 1: Build system prompt interactively
            system_prompt = _iterative_system_prompt_builder(
                schema, domain, llm_analyzer, file_roles
            )

            # Step 2: Build question template interactively
            question_template = _iterative_question_template_builder(
                schema, domain, system_prompt, llm_analyzer
            )

            # Step 3: Build answer template interactively
            answer_template, postprocessing = _iterative_answer_template_builder(
                schema, domain, system_prompt, question_template, llm_analyzer
            )

            # Build config with custom prompts
            config = builder.from_schema_with_custom_prompts(
                schema,
                goal=goal,
                num_samples=num_samples,
                output_path=str(output_path / "synth_output.jsonl"),
                system_prompt=system_prompt,
                question_template=question_template,
                answer_template=answer_template,
                postprocessing=postprocessing,
                attribute_map=attribute_map,
            )
        else:
            # Quick auto-generation mode
            cli_utils.CONSOLE.print(
                "[dim]Using AI-inferred domain knowledge for config generation...[/dim]"
            )
            with cli_utils.CONSOLE.status(
                "[green]Generating domain-specific prompts...[/green]", spinner="dots"
            ):
                config = builder.from_schema_with_inference(
                    schema,
                    goal=goal,
                    num_samples=num_samples,
                    output_path=str(output_path / "synth_output.jsonl"),
                    domain=domain,
                    llm_analyzer=llm_analyzer,
                    attribute_map=attribute_map,
                )
    else:
        config = builder.from_schema(
            schema,
            goal=goal,
            num_samples=num_samples,
            output_path=str(output_path / "synth_output.jsonl"),
            attribute_map=attribute_map,
        )

    config_path = output_path / "synth_config.yaml"
    config.to_yaml(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]Synth config saved to: {config_path}[/green]")

    # Display annotated config
    _display_annotated_synth_config(config, config_path, schema, goal, verbose=verbose)

    return [f"oumi synth -c {config_path}"]


def _wizard_judge(schema, output_path: Path, domain=None, llm_analyzer=None, verbose: bool = False):
    """Configure judge."""
    from oumi.onboarding.config_builder import JudgeConfigBuilder

    cli_utils.CONSOLE.print("\n[bold cyan]Step 3/5: Configuring evaluation...[/bold cyan]")

    cli_utils.CONSOLE.print(
        Panel(
            "[bold white][1] generic[/bold white] - General quality evaluation\n"
            "    [dim]Evaluates overall quality: coherence, helpfulness, and clarity.\n"
            "    Use when you want a broad quality score without specific criteria.[/dim]\n\n"
            "[bold white][2] compliance[/bold white] - Check guideline adherence\n"
            "    [dim]Verifies responses follow specific rules or guidelines.\n"
            "    Example: \"Does the agent follow the refund policy?\"[/dim]\n\n"
            "[bold white][3] relevance[/bold white] - Check answer relevance\n"
            "    [dim]Measures how well answers address the question asked.\n"
            "    Useful for Q&A systems and search result evaluation.[/dim]\n\n"
            "[bold white][4] safety[/bold white] - Check content safety\n"
            "    [dim]Detects harmful, biased, or inappropriate content.\n"
            "    Essential for production deployments and content moderation.[/dim]\n\n"
            "[bold white][5] groundedness[/bold white] - Check factual accuracy\n"
            "    [dim]Verifies claims are supported by provided context.\n"
            "    Critical for RAG systems to detect hallucinations.[/dim]",
            title="Judge Type",
            border_style="blue",
        )
    )

    type_map = {
        "1": "generic",
        "2": "compliance",
        "3": "relevance",
        "4": "safety",
        "5": "groundedness",
    }
    type_choice = Prompt.ask(
        "\nSelect judge type", choices=["1", "2", "3", "4", "5"], default="1"
    )
    judge_type = type_map[type_choice]

    cli_utils.CONSOLE.print(
        "\n[dim]You can add custom criteria to evaluate domain-specific requirements.[/dim]"
    )
    custom_criteria = None
    if Confirm.ask("Add custom evaluation criteria?", default=False):
        cli_utils.CONSOLE.print(
            "[dim]Enter your criteria as a clear question or statement.\n"
            "Example: \"Is the response professional and empathetic?\"[/dim]"
        )
        custom_criteria = Prompt.ask("Enter your evaluation criteria")

    cli_utils.CONSOLE.print("\n[bold cyan]Step 4/5: Generating configuration...[/bold cyan]")

    builder = JudgeConfigBuilder()

    # Use LLM-inferred config if domain analysis is available
    if domain is not None and llm_analyzer is not None and custom_criteria is None:
        cli_utils.CONSOLE.print(
            "[dim]Using AI-inferred domain knowledge for evaluation criteria...[/dim]"
        )
        with cli_utils.CONSOLE.status(
            "[green]Generating domain-specific evaluation prompts...[/green]",
            spinner="dots",
        ):
            config = builder.from_schema_with_inference(
                schema,
                judge_type=judge_type,
                domain=domain,
                llm_analyzer=llm_analyzer,
            )
    else:
        config = builder.from_schema(
            schema,
            judge_type=judge_type,
            custom_criteria=custom_criteria,
        )

    config_path = output_path / "judge_config.yaml"
    config.to_yaml(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]Judge config saved to: {config_path}[/green]")

    # Display annotated config
    _display_annotated_judge_config(config, config_path, schema, judge_type, verbose=verbose)

    return [f"oumi judge dataset -c {config_path} --input {schema.source_path}"]


def _wizard_train(schema, output_path: Path, verbose: bool = False):
    """Configure training."""
    from oumi.onboarding.config_builder import TrainConfigBuilder

    cli_utils.CONSOLE.print("\n[bold cyan]Step 3/5: Configuring training...[/bold cyan]")

    cli_utils.CONSOLE.print(
        Panel(
            "[bold white][1] small[/bold white] - Llama 3.2 1B Instruct\n"
            "    [dim]Fastest training, lowest resource usage.\n"
            "    GPU: 8GB+ VRAM (RTX 3070, T4, etc.)\n"
            "    Best for: Quick experiments, simple tasks, limited hardware.[/dim]\n\n"
            "[bold white][2] medium[/bold white] - Llama 3.2 3B Instruct\n"
            "    [dim]Good balance of speed and capability.\n"
            "    GPU: 16GB+ VRAM (RTX 4080, A10, etc.)\n"
            "    Best for: Most use cases, production deployments.[/dim]\n\n"
            "[bold white][3] large[/bold white] - Llama 3.1 8B Instruct\n"
            "    [dim]Most capable, best quality outputs.\n"
            "    GPU: 24GB+ VRAM (RTX 4090, A100, etc.)\n"
            "    Best for: Complex tasks, highest quality requirements.[/dim]",
            title="Base Model",
            border_style="blue",
        )
    )

    model_map = {
        "1": "meta-llama/Llama-3.2-1B-Instruct",
        "2": "meta-llama/Llama-3.2-3B-Instruct",
        "3": "meta-llama/Llama-3.1-8B-Instruct",
    }
    model_choice = Prompt.ask("\nSelect model", choices=["1", "2", "3"], default="1")
    base_model = model_map[model_choice]

    cli_utils.CONSOLE.print(
        Panel(
            "[bold]LoRA (Low-Rank Adaptation)[/bold]\n\n"
            "[green]Recommended: Yes[/green]\n\n"
            "[dim]LoRA trains only a small set of adapter weights instead of the full model.\n"
            "Benefits:\n"
            "  - 10-100x less memory usage\n"
            "  - 2-3x faster training\n"
            "  - Preserves base model capabilities\n"
            "  - Easy to swap adapters for different tasks\n\n"
            "Use full fine-tuning only if you need maximum customization\n"
            "and have abundant GPU resources.[/dim]",
            title="Training Method",
            border_style="blue",
        )
    )
    use_lora = Confirm.ask("\nUse LoRA for efficient fine-tuning?", default=True)

    cli_utils.CONSOLE.print(
        "\n[dim]Training steps: More steps = better learning, but risk of overfitting.\n"
        "Rule of thumb: 1-3 epochs over your data. For 1000 samples, ~500-1500 steps.[/dim]"
    )
    max_steps = IntPrompt.ask("Maximum training steps", default=1000)

    cli_utils.CONSOLE.print("\n[bold cyan]Step 4/5: Generating configuration...[/bold cyan]")

    builder = TrainConfigBuilder()
    config = builder.from_data_path(
        schema.source_path,
        base_model=base_model,
        use_lora=use_lora,
        max_steps=max_steps,
        output_dir=str(output_path / "model_output"),
    )

    config_path = output_path / "train_config.yaml"
    config.to_yaml(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]Train config saved to: {config_path}[/green]")

    # Display annotated config
    _display_annotated_train_config(config, config_path, base_model, use_lora, verbose=verbose)

    return [f"oumi train -c {config_path}"]


def _wizard_pipeline(
    schema, output_path: Path, analyzer, domain=None, llm_analyzer=None, file_roles=None,
    column_assignments=None, verbose: bool = False,
):
    """Configure full pipeline.

    Args:
        schema: DataSchema for the primary data file.
        output_path: Output directory for generated configs.
        analyzer: DataAnalyzer instance.
        domain: Optional DomainAnalysis from LLM analysis.
        llm_analyzer: Optional LLMAnalyzer instance.
        file_roles: Optional dict mapping roles to file paths.
        column_assignments: Optional dict with column-level role assignments.
    """
    from oumi.onboarding.config_builder import (
        JudgeConfigBuilder,
        SynthConfigBuilder,
        TrainConfigBuilder,
    )

    cli_utils.CONSOLE.print("\n[bold cyan]Step 3/5: Configuring pipeline...[/bold cyan]")

    cli_utils.CONSOLE.print(
        Panel(
            "[bold]Pipeline Overview[/bold]\n\n"
            "[cyan]1. Synthesis[/cyan] - Generate training data from your raw data\n"
            "[cyan]2. Evaluation[/cyan] - Score and filter generated data for quality\n"
            "[cyan]3. Training[/cyan] - Fine-tune a model on the high-quality data\n\n"
            "[dim]Each step produces output that feeds into the next step.[/dim]",
            title="Full Pipeline: Synth -> Judge -> Train",
            border_style="magenta",
        )
    )

    # Synth config
    suggested_goal = analyzer.suggest_goal(schema)
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Recommended goal: [green]{suggested_goal}[/green][/bold]\n\n"
            "[dim]Goals: qa (Q&A pairs), conversation (dialogues),\n"
            "augmentation (variations), instruction (task data)[/dim]",
            title="Step 1: Synthesis",
            border_style="cyan",
        )
    )
    use_suggested = Confirm.ask(f"Use suggested goal '{suggested_goal}'?", default=True)
    goal = suggested_goal if use_suggested else Prompt.ask(
        "Enter goal", choices=SYNTH_GOAL_CHOICES
    )
    num_samples = IntPrompt.ask("Number of samples to generate", default=100)

    synth_builder = SynthConfigBuilder()

    # Build attribute_map from column_assignments if provided
    attribute_map = None
    if column_assignments:
        attribute_map = {}
        if "context" in column_assignments:
            attribute_map["context"] = column_assignments["context"]["column"]
        if "question" in column_assignments:
            attribute_map["question"] = column_assignments["question"]["column"]
        if "answer" in column_assignments:
            attribute_map["answer"] = column_assignments["answer"]["column"]
        if "reference_values" in column_assignments:
            attribute_map["reference"] = column_assignments["reference_values"]["column"]
        if "label" in column_assignments:
            attribute_map["label"] = column_assignments["label"]["column"]
        if "metadata" in column_assignments and isinstance(column_assignments["metadata"], list):
            metadata_cols = [m["column"] for m in column_assignments["metadata"]]
            attribute_map["metadata"] = metadata_cols

        if attribute_map:
            cli_utils.CONSOLE.print(
                f"[dim]Using column mappings: {attribute_map}[/dim]"
            )

    # Use LLM-inferred config if domain analysis is available
    if domain is not None and llm_analyzer is not None:
        cli_utils.CONSOLE.print(
            "[dim]Using AI-inferred domain knowledge for synth config...[/dim]"
        )
        with cli_utils.CONSOLE.status(
            "[green]Generating domain-specific synth prompts...[/green]", spinner="dots"
        ):
            synth_config = synth_builder.from_schema_with_inference(
                schema,
                goal=goal,
                num_samples=num_samples,
                output_path=str(output_path / "synth_output.jsonl"),
                domain=domain,
                llm_analyzer=llm_analyzer,
                attribute_map=attribute_map,
            )
    else:
        synth_config = synth_builder.from_schema(
            schema,
            goal=goal,
            num_samples=num_samples,
            output_path=str(output_path / "synth_output.jsonl"),
            attribute_map=attribute_map,
        )

    # Judge config
    cli_utils.CONSOLE.print(
        Panel(
            "[dim]Judge types:\n"
            "  generic - Overall quality\n"
            "  compliance - Guideline adherence\n"
            "  relevance - Answer relevance\n"
            "  safety - Content safety\n"
            "  groundedness - Factual accuracy[/dim]",
            title="Step 2: Evaluation",
            border_style="cyan",
        )
    )
    judge_type = Prompt.ask(
        "Judge type", choices=JUDGE_TYPE_CHOICES, default="generic"
    )
    judge_builder = JudgeConfigBuilder()

    # Use LLM-inferred config if domain analysis is available
    if domain is not None and llm_analyzer is not None:
        cli_utils.CONSOLE.print(
            "[dim]Using AI-inferred domain knowledge for judge config...[/dim]"
        )
        with cli_utils.CONSOLE.status(
            "[green]Generating domain-specific evaluation prompts...[/green]",
            spinner="dots",
        ):
            judge_config = judge_builder.from_schema_with_inference(
                schema,
                judge_type=judge_type,
                domain=domain,
                llm_analyzer=llm_analyzer,
            )
    else:
        judge_config = judge_builder.from_schema(schema, judge_type=judge_type)

    # Train config
    cli_utils.CONSOLE.print(
        Panel(
            "[dim]LoRA: Efficient fine-tuning with less memory (recommended)\n"
            "Steps: More = better learning, but risk overfitting[/dim]",
            title="Step 3: Training",
            border_style="cyan",
        )
    )
    use_lora = Confirm.ask("Use LoRA for efficient fine-tuning?", default=True)
    max_steps = IntPrompt.ask("Max training steps", default=1000)

    train_builder = TrainConfigBuilder()
    train_config = train_builder.from_data_path(
        str(output_path / "synth_output.jsonl"),
        use_lora=use_lora,
        max_steps=max_steps,
        output_dir=str(output_path / "model_output"),
    )

    cli_utils.CONSOLE.print("\n[bold cyan]Step 4/5: Generating configurations...[/bold cyan]")

    # Save all configs
    synth_path = output_path / "synth_config.yaml"
    judge_path = output_path / "judge_config.yaml"
    train_path = output_path / "train_config.yaml"

    synth_config.to_yaml(str(synth_path))
    judge_config.to_yaml(str(judge_path))
    train_config.to_yaml(str(train_path))

    cli_utils.CONSOLE.print(f"\n[green]Configs saved:[/green]")
    cli_utils.CONSOLE.print(f"  Synth:  {synth_path}")
    cli_utils.CONSOLE.print(f"  Judge:  {judge_path}")
    cli_utils.CONSOLE.print(f"  Train:  {train_path}")

    # Display annotated configs for all three
    if verbose:
        cli_utils.CONSOLE.print("\n[bold]Pipeline Configuration Summary:[/bold]")

    _display_annotated_synth_config(synth_config, synth_path, schema, goal, verbose=verbose)
    if verbose:
        cli_utils.CONSOLE.print("")
    _display_annotated_judge_config(judge_config, judge_path, schema, judge_type, verbose=verbose)
    if verbose:
        cli_utils.CONSOLE.print("")
    _display_annotated_train_config(
        train_config, train_path, "meta-llama/Llama-3.2-1B-Instruct", use_lora, verbose=verbose
    )

    return [
        f"oumi synth -c {synth_path}",
        f"oumi judge dataset -c {judge_path} --input {output_path / 'synth_output.jsonl'}",
        f"oumi train -c {train_path}",
    ]


def generate(
    ctx: typer.Context,
    data: Annotated[
        str,
        typer.Option(
            "--data",
            "-d",
            help="Path to your data file.",
        ),
    ],
    goal: Annotated[
        str,
        typer.Option(
            "--goal",
            "-g",
            help="Goal: synth, judge, train, or pipeline.",
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output path for generated config(s).",
        ),
    ] = "./oumi_config.yaml",
    synth_goal: Annotated[
        Optional[str],
        typer.Option(
            "--synth-goal",
            help="Synthesis goal: qa, conversation, augmentation, instruction.",
        ),
    ] = None,
    judge_type: Annotated[
        Optional[str],
        typer.Option(
            "--judge-type",
            help="Judge type: generic, compliance, relevance, safety, groundedness.",
        ),
    ] = None,
    num_samples: Annotated[
        int,
        typer.Option(
            "--num-samples",
            "-n",
            help="Number of samples to generate (for synth).",
        ),
    ] = 100,
    base_model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Base model for training.",
        ),
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    use_lora: Annotated[
        bool,
        typer.Option(
            "--lora/--no-lora",
            help="Use LoRA for training.",
        ),
    ] = True,
):
    """Generate Oumi config from your data automatically.

    This command analyzes your data and generates configuration files
    without interactive prompts.

    Examples:
        # Generate synth config
        oumi onboard generate --data ./data.csv --goal synth -o ./synth.yaml

        # Generate judge config
        oumi onboard generate --data ./data.json --goal judge --judge-type compliance

        # Generate full pipeline configs
        oumi onboard generate --data ./data.csv --goal pipeline -o ./configs/
    """
    # Delayed imports
    from oumi.onboarding import DataAnalyzer
    from oumi.onboarding.config_builder import (
        JudgeConfigBuilder,
        SynthConfigBuilder,
        TrainConfigBuilder,
    )

    # Validate goal
    if goal not in GOAL_CHOICES:
        cli_utils.CONSOLE.print(
            f"[red]Invalid goal: {goal}. Choose from: {GOAL_CHOICES}[/red]"
        )
        raise typer.Exit(1)

    # Analyze data
    data_path = Path(data)
    if not data_path.exists():
        cli_utils.CONSOLE.print(f"[red]Error: File not found: {data}[/red]")
        raise typer.Exit(1)

    with cli_utils.CONSOLE.status("[green]Analyzing data...[/green]", spinner="dots"):
        analyzer = DataAnalyzer()
        schema = analyzer.analyze(data_path)

    cli_utils.CONSOLE.print(
        f"[green]Analyzed: {schema.row_count} rows, {len(schema.columns)} columns[/green]"
    )

    output_path = Path(output)

    if goal == "synth":
        # Use provided or inferred synth goal
        sg = synth_goal or analyzer.suggest_goal(schema)
        builder = SynthConfigBuilder()
        config = builder.from_schema(schema, goal=sg, num_samples=num_samples)

        if output_path.suffix != ".yaml":
            output_path = output_path / "synth_config.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        config.to_yaml(str(output_path))
        cli_utils.CONSOLE.print(f"[green]Synth config saved to: {output_path}[/green]")

    elif goal == "judge":
        jt = judge_type or "generic"
        builder = JudgeConfigBuilder()
        config = builder.from_schema(schema, judge_type=jt)

        if output_path.suffix != ".yaml":
            output_path = output_path / "judge_config.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        config.to_yaml(str(output_path))
        cli_utils.CONSOLE.print(f"[green]Judge config saved to: {output_path}[/green]")

    elif goal == "train":
        builder = TrainConfigBuilder()
        config = builder.from_data_path(
            data,
            base_model=base_model,
            use_lora=use_lora,
        )

        if output_path.suffix != ".yaml":
            output_path = output_path / "train_config.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        config.to_yaml(str(output_path))
        cli_utils.CONSOLE.print(f"[green]Train config saved to: {output_path}[/green]")

    elif goal == "pipeline":
        # Create directory for multiple configs
        if output_path.suffix == ".yaml":
            output_path = output_path.parent / "configs"
        output_path.mkdir(parents=True, exist_ok=True)

        sg = synth_goal or analyzer.suggest_goal(schema)
        jt = judge_type or "generic"

        # Generate all configs
        synth_builder = SynthConfigBuilder()
        synth_config = synth_builder.from_schema(
            schema,
            goal=sg,
            num_samples=num_samples,
            output_path=str(output_path / "synth_output.jsonl"),
        )

        judge_builder = JudgeConfigBuilder()
        judge_config = judge_builder.from_schema(schema, judge_type=jt)

        train_builder = TrainConfigBuilder()
        train_config = train_builder.from_data_path(
            str(output_path / "synth_output.jsonl"),
            base_model=base_model,
            use_lora=use_lora,
            output_dir=str(output_path / "model_output"),
        )

        # Save configs
        synth_config.to_yaml(str(output_path / "synth_config.yaml"))
        judge_config.to_yaml(str(output_path / "judge_config.yaml"))
        train_config.to_yaml(str(output_path / "train_config.yaml"))

        cli_utils.CONSOLE.print(f"[green]Pipeline configs saved to: {output_path}/[/green]")
        cli_utils.CONSOLE.print("  - synth_config.yaml")
        cli_utils.CONSOLE.print("  - judge_config.yaml")
        cli_utils.CONSOLE.print("  - train_config.yaml")


def templates(
    ctx: typer.Context,
    config_type: Annotated[
        Optional[str],
        typer.Option(
            "--type",
            "-t",
            help="Filter by config type: synth, judge, train.",
        ),
    ] = None,
):
    """List available configuration templates.

    Example:
        oumi onboard templates --type synth
    """
    table = Table(title="Available Templates", show_edge=False)
    table.add_column("Template", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Description", style="green")

    # Built-in templates
    templates_list = [
        ("qa_generation", "synth", "Generate Q&A pairs from context"),
        ("conversation_augmentation", "synth", "Augment conversation data"),
        ("data_augmentation", "synth", "Create variations of existing data"),
        ("instruction_following", "synth", "Generate instruction-following data"),
        ("compliance_judge", "judge", "Evaluate compliance with guidelines"),
        ("relevance_judge", "judge", "Evaluate answer relevance"),
        ("safety_judge", "judge", "Evaluate content safety"),
        ("groundedness_judge", "judge", "Evaluate factual accuracy"),
        ("lora_sft", "train", "LoRA fine-tuning configuration"),
        ("full_sft", "train", "Full fine-tuning configuration"),
    ]

    for name, ttype, desc in templates_list:
        if config_type is None or ttype == config_type:
            table.add_row(name, ttype, desc)

    cli_utils.CONSOLE.print(table)
    cli_utils.CONSOLE.print(
        "\n[dim]Use templates with: oumi onboard generate --template <name>[/dim]"
    )


def analyze(
    ctx: typer.Context,
    data: Annotated[
        str,
        typer.Option(
            "--data",
            "-d",
            help="Path to your data file.",
        ),
    ],
):
    """Analyze a data file and show suggested configurations.

    Example:
        oumi onboard analyze --data ./my_data.csv
    """
    from oumi.onboarding import DataAnalyzer, FieldMapper

    data_path = Path(data)
    if not data_path.exists():
        cli_utils.CONSOLE.print(f"[red]Error: File not found: {data}[/red]")
        raise typer.Exit(1)

    with cli_utils.CONSOLE.status("[green]Analyzing...[/green]", spinner="dots"):
        analyzer = DataAnalyzer()
        schema = analyzer.analyze(data_path)
        mapper = FieldMapper()
        mappings = mapper.suggest_mappings(schema)

    # Display schema
    _display_schema_info(schema)

    # Display suggested mappings
    if mappings:
        mapping_table = Table(title="Suggested Field Mappings", show_edge=False)
        mapping_table.add_column("Your Column", style="cyan")
        mapping_table.add_column("Oumi Placeholder", style="green")
        mapping_table.add_column("Confidence", style="yellow")

        for m in mappings:
            conf_str = f"{m.confidence:.0%}"
            mapping_table.add_row(m.customer_column, f"{{{m.oumi_placeholder}}}", conf_str)

        cli_utils.CONSOLE.print(mapping_table)

    # Display suggested goal
    suggested_goal = analyzer.suggest_goal(schema)
    cli_utils.CONSOLE.print(f"\n[bold]Suggested synthesis goal:[/bold] [green]{suggested_goal}[/green]")

    cli_utils.CONSOLE.print(
        f"\n[dim]To generate configs: oumi onboard generate --data {data} --goal synth[/dim]"
    )
