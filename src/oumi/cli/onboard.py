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
# These describe what each file contributes to training data generation
FILE_ROLE_GUIDANCE = {
    "primary": {
        "title": "TRAINING DATA",
        "description": "Your main dataset for training/synthesis",
        "help": [
            "Contains the inputs you want to generate outputs for",
            "Usually your largest dataset with user questions or requests",
            "Examples: customer_questions.csv, support_tickets.json",
        ],
    },
    "reference": {
        "title": "KNOWLEDGE BASE",
        "description": "Background information for grounding answers",
        "help": [
            "Facts, documents, or data the model should reference",
            "Helps generate accurate, grounded responses",
            "Examples: product_catalog.csv, company_policies.pdf, FAQ.docx",
        ],
    },
    "rules": {
        "title": "SYSTEM INSTRUCTIONS",
        "description": "Guidelines for how the model should behave",
        "help": [
            "Tone, style, constraints, and response format",
            "Becomes part of the system prompt",
            "Examples: style_guide.md, response_rules.txt, brand_voice.docx",
        ],
    },
    "examples": {
        "title": "EXAMPLE OUTPUTS",
        "description": "Sample input-output pairs showing ideal responses",
        "help": [
            "Shows the model what good answers look like",
            "Used for few-shot prompting and quality benchmarks",
            "Examples: approved_responses.csv, gold_standard.json",
        ],
    },
}

# Column role descriptions for granular assignment
# These describe what each column contributes to training examples
COLUMN_ROLE_GUIDANCE = {
    "context": {
        "title": "SOURCE CONTENT",
        "description": "Text to generate questions/answers from",
        "help": "The main content column (e.g., document text, article body)",
    },
    "question": {
        "title": "USER INPUT",
        "description": "Questions or requests from users",
        "help": "What users ask the model (defines input distribution)",
    },
    "answer": {
        "title": "MODEL OUTPUT",
        "description": "Ideal responses to learn from",
        "help": "What the model should output (defines output quality)",
    },
    "reference_values": {
        "title": "VALID OPTIONS",
        "description": "Allowed values for constrained outputs",
        "help": "List of valid choices (e.g., categories, product names)",
    },
    "label": {
        "title": "QUALITY LABEL",
        "description": "Quality scores or classifications",
        "help": "Labels like 'good', 'bad', 'approved', 'rejected'",
    },
    "metadata": {
        "title": "EXTRA CONTEXT",
        "description": "Additional info to include in prompts",
        "help": "Supporting fields like date, category, customer_id",
    },
}


# ============================================================================
# SIMPLIFIED WIZARD FLOW: Minimum viable dataclasses (Occam's Razor)
# ============================================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    """Minimal task specification - just what's essential."""

    name: str = ""
    """Short name/title for the task."""

    description: str = ""
    """What the model should do (free-form description from user)."""

    example_input: str = ""
    """Example input for the task."""

    system_prompt: str = ""
    """The complete system prompt (auto-generated, user can edit)."""


@dataclass
class InputSpec:
    """Minimal input specification - detected from data."""

    format: str = "single_turn"
    """Detected format: single_turn, multi_turn, document, structured."""

    samples: list = field(default_factory=list)
    """Sample inputs from the data (auto-detected)."""

    source_column: str = ""
    """Column to use for inputs."""


@dataclass
class OutputSpec:
    """Minimal output specification - just quality criteria."""

    criteria: list = field(default_factory=list)
    """What makes a good response (user-provided list)."""

    example: str = ""
    """One good example output (optional)."""


# Input format types (simplified - just for detection)
INPUT_FORMATS = {
    "single_turn": "Single question/answer",
    "multi_turn": "Multi-turn conversation",
    "document": "Document/text processing",
    "structured": "Structured data (JSON/records)",
}

# Backward compatibility aliases for legacy code
TaskDefinition = TaskSpec
SystemPromptSpec = TaskSpec  # System prompt is now part of TaskSpec
InputDistributionSpec = InputSpec
OutputQualitySpec = OutputSpec

# Legacy INPUT_FORMAT_TYPES for backward compatibility
INPUT_FORMAT_TYPES = {
    "single_turn": {"name": "Single-turn Q&A", "description": "One question, one answer"},
    "multi_turn": {"name": "Multi-turn conversation", "description": "Back-and-forth dialogue"},
    "document": {"name": "Document/Text", "description": "Long-form text processing"},
    "structured": {"name": "Structured data", "description": "JSON or tabular data"},
    "instruction": {"name": "Instruction-following", "description": "Task instructions with input"},
}


@dataclass
class WizardState:
    """Simplified state object for the wizard."""

    task: TaskSpec = field(default_factory=TaskSpec)
    inputs: InputSpec = field(default_factory=InputSpec)
    outputs: OutputSpec = field(default_factory=OutputSpec)

    # File analysis results
    files: list = field(default_factory=list)
    schemas: dict = field(default_factory=dict)
    primary_schema: Any = None

    # LLM analyzer (not serialized)
    llm_analyzer: Any = None
    domain_analysis: Any = None

    # Track completed steps for caching
    completed_steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize state to a dictionary for caching."""
        # Serialize files with analysis results and content hash
        files_data = []
        for f in self.files:
            file_path = f.get("path")
            file_data = {
                "path": str(file_path) if file_path else "",
                "name": f.get("name", ""),
                "extension": f.get("extension", ""),
                # File analysis results from LLM
                "suggested_purpose": f.get("suggested_purpose", ""),
                "suggested_role": f.get("suggested_role", ""),
                "role_reason": f.get("role_reason", ""),
                # Content hash for cache invalidation
                "content_hash": f.get("content_hash", ""),
            }
            files_data.append(file_data)

        # Serialize domain analysis if available
        domain_data = None
        if self.domain_analysis:
            domain_data = {
                "domain": self.domain_analysis.domain,
                "description": self.domain_analysis.description,
                "terminology": self.domain_analysis.terminology,
                "quality_signals": self.domain_analysis.quality_signals,
                "common_issues": self.domain_analysis.common_issues,
                "suggested_persona": self.domain_analysis.suggested_persona,
                "data_purpose": self.domain_analysis.data_purpose,
            }

        return {
            "task": {
                "description": self.task.description,
                "system_prompt": self.task.system_prompt,
            },
            "inputs": {
                "format": self.inputs.format,
                "samples": self.inputs.samples,
                "source_column": self.inputs.source_column,
            },
            "outputs": {
                "criteria": self.outputs.criteria,
                "example": self.outputs.example,
            },
            "files": files_data,
            "domain_analysis": domain_data,
            "completed_steps": self.completed_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WizardState":
        """Deserialize state from a dictionary."""
        state = cls()

        # Restore task
        task_data = data.get("task", {})
        state.task = TaskSpec(
            description=task_data.get("description", ""),
            system_prompt=task_data.get("system_prompt", ""),
        )

        # Restore inputs
        inputs_data = data.get("inputs", {})
        state.inputs = InputSpec(
            format=inputs_data.get("format", "single_turn"),
            samples=inputs_data.get("samples", []),
            source_column=inputs_data.get("source_column", ""),
        )

        # Restore outputs
        outputs_data = data.get("outputs", {})
        state.outputs = OutputSpec(
            criteria=outputs_data.get("criteria", []),
            example=outputs_data.get("example", ""),
        )

        # Restore files with analysis results
        state.files = [
            {
                "path": Path(f.get("path", "")) if f.get("path") else None,
                "name": f.get("name", ""),
                "extension": f.get("extension", ""),
                # Restore file analysis results
                "suggested_purpose": f.get("suggested_purpose", ""),
                "suggested_role": f.get("suggested_role", ""),
                "role_reason": f.get("role_reason", ""),
                "content_hash": f.get("content_hash", ""),
            }
            for f in data.get("files", [])
        ]

        # Restore domain analysis if available
        domain_data = data.get("domain_analysis")
        if domain_data:
            from oumi.onboarding.llm_analyzer import DomainAnalysis

            state.domain_analysis = DomainAnalysis(
                domain=domain_data.get("domain", "unknown"),
                description=domain_data.get("description", ""),
                terminology=domain_data.get("terminology", []),
                quality_signals=domain_data.get("quality_signals", []),
                common_issues=domain_data.get("common_issues", []),
                suggested_persona=domain_data.get("suggested_persona", ""),
                data_purpose=domain_data.get("data_purpose", ""),
            )

        # Restore completed steps
        state.completed_steps = data.get("completed_steps", [])

        return state


# Cache file name
WIZARD_CACHE_FILE = ".wizard_cache.yaml"


def _compute_file_hash(file_path: Path) -> str:
    """Compute a hash of the file content for cache invalidation.

    Args:
        file_path: Path to the file.

    Returns:
        SHA256 hash of the file content as hex string.
    """
    import hashlib

    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        # If we can't read the file, return empty string (will force re-analysis)
        return ""


def _get_cache_path(output_dir: Path) -> Path:
    """Get the path to the wizard cache file."""
    return output_dir / WIZARD_CACHE_FILE


def _save_wizard_cache(state: WizardState, output_dir: Path, step_name: str) -> None:
    """Save wizard state to cache file after completing a step.

    Args:
        state: Current wizard state.
        output_dir: Output directory for cache file.
        step_name: Name of the step just completed.
    """
    import yaml

    # Add step to completed list if not already there
    if step_name not in state.completed_steps:
        state.completed_steps.append(step_name)

    cache_path = _get_cache_path(output_dir)
    cache_data = state.to_dict()

    # Add metadata
    cache_data["_metadata"] = {
        "last_updated": str(Path()),  # Will be set by file system
        "cache_version": "1.0",
        "description": "Oumi wizard state cache. You can edit this file to modify wizard settings.",
    }

    with open(cache_path, "w") as f:
        yaml.dump(cache_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    cli_utils.CONSOLE.print(f"[dim]Cache saved: {cache_path}[/dim]")


def _load_wizard_cache(output_dir: Path) -> Optional[WizardState]:
    """Load wizard state from cache file if it exists.

    Args:
        output_dir: Output directory containing cache file.

    Returns:
        WizardState if cache exists and is valid, None otherwise.
    """
    import yaml

    cache_path = _get_cache_path(output_dir)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cache_data = yaml.safe_load(f)

        if not cache_data or not isinstance(cache_data, dict):
            return None

        # Remove metadata before deserializing
        cache_data.pop("_metadata", None)

        return WizardState.from_dict(cache_data)
    except Exception as e:
        cli_utils.CONSOLE.print(f"[yellow]Warning: Could not load cache: {e}[/yellow]")
        return None


def _display_cache_summary(state: WizardState, cache_path: Path) -> None:
    """Display a summary of the cached wizard state.

    Args:
        state: Loaded wizard state.
        cache_path: Path to the cache file.
    """
    completed = state.completed_steps

    step_status = []
    all_steps = [
        ("task", "Task"),
        ("inputs", "Inputs"),
        ("outputs", "Outputs"),
        ("generate", "Generate"),
    ]

    for step_id, step_name in all_steps:
        if step_id in completed:
            step_status.append(f"  [green]✓[/green] {step_name}")
        else:
            step_status.append(f"  [dim]○[/dim] {step_name}")

    # Build summary with new field names
    task_preview = state.task.description[:50] + "..." if state.task.description else "[not set]"
    prompt_preview = state.task.system_prompt[:50] + "..." if state.task.system_prompt else "[not set]"
    input_preview = INPUT_FORMATS.get(state.inputs.format, state.inputs.format) if state.inputs.format else "[not set]"
    criteria_preview = ", ".join(state.outputs.criteria[:3]) if state.outputs.criteria else "[not set]"

    summary_parts = [
        f"[bold]Task:[/bold] {task_preview}",
        f"[bold]System Prompt:[/bold] {prompt_preview}",
        f"[bold]Input Format:[/bold] {input_preview}",
        f"[bold]Quality Criteria:[/bold] {criteria_preview}",
        "",
        "[bold]Steps Completed:[/bold]",
        *step_status,
    ]

    cli_utils.CONSOLE.print(
        Panel(
            "\n".join(summary_parts),
            title=f"[cyan]Cached State: {cache_path}[/cyan]",
            border_style="cyan",
        )
    )


def _prompt_cache_action(cache_path: Path) -> str:
    """Prompt user for what to do with existing cache.

    Args:
        cache_path: Path to the cache file.

    Returns:
        One of: "resume", "edit", "restart"
    """
    cli_utils.CONSOLE.print(
        "\n[bold]Options:[/bold]\n"
        "  [cyan][1][/cyan] Resume from where you left off\n"
        "  [cyan][2][/cyan] Edit the cache file first (opens in editor)\n"
        "  [cyan][3][/cyan] Start fresh (delete cache)\n"
    )

    choice = Prompt.ask("Your choice", choices=["1", "2", "3"], default="1")

    if choice == "1":
        return "resume"
    elif choice == "2":
        return "edit"
    else:
        return "restart"


def _open_cache_for_editing(cache_path: Path) -> None:
    """Open the cache file in the user's default editor.

    Args:
        cache_path: Path to the cache file.
    """
    import os
    import subprocess

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))

    cli_utils.CONSOLE.print(f"\n[dim]Opening {cache_path} in {editor}...[/dim]")
    cli_utils.CONSOLE.print("[dim]Save and close the editor when done.[/dim]\n")

    try:
        subprocess.run([editor, str(cache_path)], check=True)
        cli_utils.CONSOLE.print("[green]Cache file updated.[/green]")
    except FileNotFoundError:
        cli_utils.CONSOLE.print(
            f"[yellow]Could not open editor '{editor}'.[/yellow]\n"
            f"[dim]Please edit the file manually: {cache_path}[/dim]\n"
        )
        Prompt.ask("Press Enter when done editing")
    except subprocess.CalledProcessError:
        cli_utils.CONSOLE.print("[yellow]Editor closed without saving changes.[/yellow]")


# Tabular file extensions that support column selection
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}


# ============================================================================
# SIMPLIFIED WIZARD FLOW: Step functions (Occam's Razor)
# ============================================================================


def _wizard_step_task(state: WizardState, verbose: bool = False) -> WizardState:
    """Step 1: Define task and generate system prompt.

    Single question: "What do you want your model to do?"
    Then auto-generates system prompt for confirmation.

    Args:
        state: Current wizard state.
        verbose: Show detailed output.

    Returns:
        Updated state with task description and system prompt.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 1/4: Define Your Task ━━━[/bold cyan]\n"
    )

    # Try to infer from files first
    suggested_desc = ""
    if state.llm_analyzer and state.files:
        with cli_utils.CONSOLE.status("[dim]Analyzing your files...[/dim]", spinner="dots"):
            analysis = _analyze_task_from_files(state.files, state.llm_analyzer)
            suggested_desc = analysis.get("task_description", "")

        if suggested_desc:
            cli_utils.CONSOLE.print(
                Panel(
                    suggested_desc,
                    title="[green]Detected Task[/green]",
                    border_style="green",
                )
            )
            if Confirm.ask("Is this correct?", default=True):
                state.task.description = suggested_desc
            else:
                state.task.description = Prompt.ask(
                    "What should your model do?",
                    default=suggested_desc,
                )
        else:
            state.task.description = Prompt.ask("What should your model do?")
    else:
        state.task.description = Prompt.ask("What should your model do?")

    # Auto-generate system prompt
    if state.llm_analyzer:
        with cli_utils.CONSOLE.status("[dim]Generating system prompt...[/dim]", spinner="dots"):
            state.task.system_prompt = _generate_system_prompt(state, state.llm_analyzer)

        cli_utils.CONSOLE.print(
            Panel(
                state.task.system_prompt,
                title="[green]Generated System Prompt[/green]",
                border_style="green",
            )
        )

        if not Confirm.ask("Accept this prompt?", default=True):
            feedback = Prompt.ask("What should change?")
            with cli_utils.CONSOLE.status("[dim]Updating...[/dim]", spinner="dots"):
                state.task.system_prompt = _refine_system_prompt(
                    state.task.system_prompt, feedback, state.llm_analyzer
                )
            cli_utils.CONSOLE.print(
                Panel(state.task.system_prompt, title="[green]Updated[/green]", border_style="green")
            )
    else:
        state.task.system_prompt = Prompt.ask(
            "System prompt for the model",
            default=f"You are a helpful assistant. {state.task.description}",
        )

    cli_utils.CONSOLE.print("[green]✓ Task defined[/green]")
    return state


def _wizard_step_inputs(state: WizardState, verbose: bool = False) -> WizardState:
    """Step 2: Define input distribution.

    Auto-detects format and samples from data, user just confirms.

    Args:
        state: Current wizard state.
        verbose: Show detailed output.

    Returns:
        Updated state with input spec.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 2/4: Input Distribution ━━━[/bold cyan]\n"
    )

    # Auto-detect from schema
    if state.primary_schema:
        # Find likely input column
        input_col = None
        for col in state.primary_schema.columns or []:
            name_lower = col.name.lower()
            if any(kw in name_lower for kw in ["question", "input", "query", "prompt", "text", "content"]):
                input_col = col.name
                break

        if not input_col and state.primary_schema.columns:
            input_col = state.primary_schema.columns[0].name

        state.inputs.source_column = input_col or ""

        # Get samples
        if state.primary_schema.sample_rows and input_col:
            samples = [
                str(row.get(input_col, ""))[:200]
                for row in state.primary_schema.sample_rows[:3]
                if row.get(input_col)
            ]
            state.inputs.samples = samples

    # Detect format
    if state.llm_analyzer and state.inputs.samples:
        with cli_utils.CONSOLE.status("[dim]Detecting input format...[/dim]", spinner="dots"):
            state.inputs.format = _detect_input_format(state, state.llm_analyzer)
    else:
        state.inputs.format = "single_turn"

    # Show what we found
    format_name = INPUT_FORMATS.get(state.inputs.format, state.inputs.format)
    cli_utils.CONSOLE.print(f"[bold]Format:[/bold] {format_name}")
    if state.inputs.source_column:
        cli_utils.CONSOLE.print(f"[bold]Source column:[/bold] {state.inputs.source_column}")

    if state.inputs.samples:
        cli_utils.CONSOLE.print("\n[bold]Sample inputs:[/bold]")
        for i, sample in enumerate(state.inputs.samples[:3], 1):
            cli_utils.CONSOLE.print(f"  {i}. {sample[:100]}{'...' if len(sample) > 100 else ''}")

    if not Confirm.ask("\nIs this correct?", default=True):
        # Let user specify column
        if state.primary_schema and state.primary_schema.columns:
            cols = [c.name for c in state.primary_schema.columns]
            cli_utils.CONSOLE.print(f"[dim]Available columns: {', '.join(cols)}[/dim]")
        state.inputs.source_column = Prompt.ask("Input column name", default=state.inputs.source_column)

    cli_utils.CONSOLE.print("[green]✓ Inputs configured[/green]")
    return state


def _wizard_step_outputs(state: WizardState, verbose: bool = False) -> WizardState:
    """Step 3: Define output quality criteria.

    Single question: "What makes a good response?"

    Args:
        state: Current wizard state.
        verbose: Show detailed output.

    Returns:
        Updated state with output criteria.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 3/4: Output Quality ━━━[/bold cyan]\n"
    )

    # Try to infer criteria
    suggested_criteria = []
    if state.llm_analyzer:
        with cli_utils.CONSOLE.status("[dim]Suggesting quality criteria...[/dim]", spinner="dots"):
            suggested_criteria = _suggest_quality_criteria(state, state.llm_analyzer)

    if suggested_criteria:
        cli_utils.CONSOLE.print("[bold]Suggested criteria:[/bold]")
        for i, c in enumerate(suggested_criteria, 1):
            cli_utils.CONSOLE.print(f"  {i}. {c}")

        if Confirm.ask("\nAccept these criteria?", default=True):
            state.outputs.criteria = suggested_criteria
        else:
            criteria_input = Prompt.ask(
                "What makes a good response? (comma-separated)",
                default=", ".join(suggested_criteria[:3]),
            )
            state.outputs.criteria = [c.strip() for c in criteria_input.split(",") if c.strip()]
    else:
        criteria_input = Prompt.ask(
            "What makes a good response? (comma-separated)",
            default="accurate, helpful, clear",
        )
        state.outputs.criteria = [c.strip() for c in criteria_input.split(",") if c.strip()]

    # Optional: get one example
    if Confirm.ask("Add an example of a good response?", default=False):
        state.outputs.example = Prompt.ask("Example good response")

    cli_utils.CONSOLE.print("[green]✓ Quality criteria defined[/green]")
    return state


def _wizard_step_generate(
    state: WizardState, output_path: Path, verbose: bool = False
) -> tuple[str, list[str]]:
    """Step 4: Generate synthesis config and judges.

    Auto-generates both based on previous steps.

    Args:
        state: Current wizard state.
        output_path: Directory to save configs.
        verbose: Show detailed output.

    Returns:
        Tuple of (synthesis_config_path, list of judge_config_paths).
    """
    from oumi.onboarding import SynthConfigBuilder, JudgeConfigBuilder

    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 4/4: Generate Configs ━━━[/bold cyan]\n"
    )

    # Generate synthesis config
    cli_utils.CONSOLE.print("[dim]Generating synthesis config...[/dim]")
    synth_builder = SynthConfigBuilder()

    if state.primary_schema:
        synth_config = synth_builder.from_schema(
            state.primary_schema,
            synth_type="qa",
            output_path=str(output_path / "synth_output.jsonl"),
        )
    else:
        synth_config = synth_builder.from_template(
            "qa_generation",
            output_path=str(output_path / "synth_output.jsonl"),
        )

    # Update with task info
    if state.task.system_prompt:
        synth_config.generation.system_prompt = state.task.system_prompt

    synth_path = output_path / "synth_config.yaml"
    synth_config.to_yaml(str(synth_path))
    cli_utils.CONSOLE.print(f"[green]✓ Synthesis config:[/green] {synth_path}")

    # Generate judges from criteria
    judge_paths = []
    if state.outputs.criteria:
        cli_utils.CONSOLE.print("[dim]Generating judge configs...[/dim]")
        judge_builder = JudgeConfigBuilder()

        for criterion in state.outputs.criteria[:3]:  # Max 3 judges
            judge_name = criterion.lower().replace(" ", "_")[:20]
            judge_config = judge_builder.from_custom_criteria(
                schema=state.primary_schema,
                judge_name=judge_name,
                criteria=f"Evaluate whether the response is {criterion}",
                description=f"Judge for: {criterion}",
            )
            judge_path = output_path / f"judge_{judge_name}.yaml"
            judge_config.to_yaml(str(judge_path))
            judge_paths.append(str(judge_path))
            cli_utils.CONSOLE.print(f"[green]✓ Judge config:[/green] {judge_path}")

    # Summary
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Task:[/bold] {state.task.description[:100]}...\n"
            f"[bold]Input format:[/bold] {INPUT_FORMATS.get(state.inputs.format, state.inputs.format)}\n"
            f"[bold]Quality criteria:[/bold] {', '.join(state.outputs.criteria)}\n\n"
            f"[bold]Generated:[/bold]\n"
            f"  • Synthesis config: {synth_path}\n"
            f"  • Judge configs: {len(judge_paths)}",
            title="[green]Setup Complete[/green]",
            border_style="green",
        )
    )

    return str(synth_path), judge_paths


# ============================================================================
# Helper functions for wizard steps
# ============================================================================


def _analyze_task_from_files(files: list[dict], llm_analyzer) -> dict:
    """Analyze files to suggest what task the user is trying to accomplish.

    Args:
        files: List of file info dicts with analysis.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with task analysis.
    """
    import json

    # Build file summaries
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


def _generate_system_prompt(state: WizardState, llm_analyzer) -> str:
    """Generate a system prompt based on task description.

    Args:
        state: Wizard state with task description.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Generated system prompt string.
    """
    import json

    # Gather context
    context_parts = [f"Task: {state.task.description}"]

    if state.domain_analysis:
        context_parts.append(f"Domain: {state.domain_analysis.domain}")

    # Add sample from files
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


def _refine_system_prompt(current: str, feedback: str, llm_analyzer) -> str:
    """Refine a system prompt based on feedback.

    Args:
        current: Current system prompt.
        feedback: User feedback on what to change.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Refined system prompt.
    """
    prompt = f"""Refine this system prompt based on the feedback.

Current prompt:
{current}

Feedback: {feedback}

Return ONLY the refined prompt text."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return current


def _detect_input_format(samples: list[str], llm_analyzer) -> str:
    """Detect input format from samples.

    Args:
        samples: List of sample inputs.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Format type: single_turn, multi_turn, document, or structured.
    """
    samples_text = "\n".join(f"- {s[:200]}" for s in samples[:3])

    prompt = f"""Classify the format of these inputs:

{samples_text}

Choose ONE format:
- single_turn: Single questions or requests
- multi_turn: Conversation with multiple turns
- document: Long text/documents to process
- structured: JSON or structured data

Return ONLY the format name."""

    try:
        result = llm_analyzer._invoke(prompt).strip().lower()
        if result in ["single_turn", "multi_turn", "document", "structured"]:
            return result
    except Exception:
        pass
    return "single_turn"


def _suggest_quality_criteria(state: WizardState, llm_analyzer) -> list[str]:
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


def _wizard_step_build_system_prompt_legacy(state: WizardState, verbose: bool = False) -> WizardState:
    """LEGACY - NOT USED. Kept for reference only.

    Args:
        state: Current wizard state with task definition.
        verbose: Show detailed output.

    Returns:
        Updated state with system prompt spec.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 2/6: Build System Prompt ━━━[/bold cyan]\n"
    )

    if verbose:
        cli_utils.CONSOLE.print(
            "[dim]The system prompt defines your model's persona, capabilities,\n"
            "and constraints. We'll use your files to suggest one.[/dim]\n"
        )

    if state.llm_analyzer:
        # Generate initial system prompt from file content
        with cli_utils.CONSOLE.status(
            "[dim]Generating system prompt from your files...[/dim]", spinner="dots"
        ):
            prompt_spec = _generate_system_prompt_from_files(state, state.llm_analyzer)

        state.system_prompt = prompt_spec

        # Display generated prompt
        cli_utils.CONSOLE.print(
            Panel(
                state.system_prompt.full_prompt,
                title="[green]Generated System Prompt[/green]",
                border_style="green",
            )
        )

        # Iterative refinement
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            cli_utils.CONSOLE.print(
                "\n[bold]Options:[/bold]\n"
                "  [cyan][1][/cyan] Accept this prompt\n"
                "  [cyan][2][/cyan] Make it more formal/professional\n"
                "  [cyan][3][/cyan] Make it more conversational/friendly\n"
                "  [cyan][4][/cyan] Add specific constraints or capabilities\n"
                "  [cyan][5][/cyan] Custom feedback\n"
            )

            choice = Prompt.ask("Your choice", choices=["1", "2", "3", "4", "5"], default="1")

            if choice == "1":
                break

            feedback = ""
            if choice == "2":
                feedback = "Make this more formal and professional in tone."
            elif choice == "3":
                feedback = "Make this more conversational and approachable."
            elif choice == "4":
                addition = Prompt.ask("What should be added?")
                feedback = f"Add this to the prompt: {addition}"
            elif choice == "5":
                feedback = Prompt.ask("Describe what changes you'd like")

            # Refine the prompt
            with cli_utils.CONSOLE.status(
                "[dim]Refining prompt...[/dim]", spinner="dots"
            ):
                refined = _refine_system_prompt(
                    state.system_prompt.full_prompt, feedback, state.llm_analyzer
                )
                state.system_prompt.full_prompt = refined

            cli_utils.CONSOLE.print(
                Panel(
                    state.system_prompt.full_prompt,
                    title=f"[green]Revised System Prompt (v{iteration + 2})[/green]",
                    border_style="green",
                )
            )
            iteration += 1
    else:
        # Manual system prompt creation
        cli_utils.CONSOLE.print(
            "[bold]Let's build your system prompt:[/bold]\n"
        )

        persona = Prompt.ask(
            "Describe the AI's role/persona",
            default=f"You are an expert assistant for {state.task.name}",
        )
        tone = Prompt.ask(
            "What tone should it use?",
            default="professional and helpful",
        )
        constraints = Prompt.ask(
            "Any constraints? (e.g., 'never make up information')",
            default="Be accurate and helpful",
        )

        state.system_prompt.persona = persona
        state.system_prompt.tone = tone
        state.system_prompt.constraints = [constraints]
        state.system_prompt.full_prompt = f"{persona} Use a {tone} tone. {constraints}"

    cli_utils.CONSOLE.print(
        "\n[green]✓ System prompt defined[/green]"
    )

    return state


def _generate_system_prompt_from_files(state: WizardState, llm_analyzer) -> SystemPromptSpec:
    """Generate a system prompt based on file content.

    Args:
        state: Wizard state with file analysis.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        SystemPromptSpec with generated prompt.
    """
    import json

    # Gather context from files
    context_parts = []

    # Add task info
    context_parts.append(f"Task: {state.task.name}")
    context_parts.append(f"Description: {state.task.description}")

    # Add domain info
    if state.domain_analysis:
        context_parts.append(f"Domain: {state.domain_analysis.domain}")
        if state.domain_analysis.terminology:
            context_parts.append(f"Key terms: {', '.join(state.domain_analysis.terminology[:10])}")

    # Add content from rules/guidelines files
    for f in state.files:
        if f.get("extension") in {".docx", ".doc", ".txt", ".md"}:
            if f.get("schema") and f["schema"].raw_text:
                text_preview = f["schema"].raw_text[:1500]
                context_parts.append(f"\nFrom {f['name']}:\n{text_preview}")

    # Add sample data
    if state.primary_schema and state.primary_schema.sample_rows:
        sample = json.dumps(state.primary_schema.sample_rows[0], indent=2)[:500]
        context_parts.append(f"\nSample data:\n{sample}")

    prompt = f"""Create a system prompt for an AI model based on this context:

{chr(10).join(context_parts)}

The system prompt should:
1. Define a clear expert persona for the "{state.task.name}" task
2. Set expectations for output quality
3. Include relevant domain terminology
4. Be 3-5 sentences

Return JSON:
{{
    "full_prompt": "The complete system prompt text",
    "persona": "Brief description of the AI's role",
    "tone": "Communication style",
    "capabilities": ["what the AI can do"],
    "constraints": ["what the AI should not do"]
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        return SystemPromptSpec(
            full_prompt=result.get("full_prompt", "You are a helpful assistant."),
            persona=result.get("persona", ""),
            tone=result.get("tone", "professional"),
            capabilities=result.get("capabilities", []),
            constraints=result.get("constraints", []),
        )
    except Exception as e:
        return SystemPromptSpec(
            full_prompt=f"You are an expert assistant for {state.task.name}. {state.task.description}",
            persona=f"Expert in {state.task.name}",
            tone="professional",
        )


def _refine_system_prompt(current_prompt: str, feedback: str, llm_analyzer) -> str:
    """Refine a system prompt based on user feedback.

    Args:
        current_prompt: The current system prompt text.
        feedback: User's feedback for improvement.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Refined system prompt text.
    """
    prompt = f"""Revise this system prompt based on the feedback.

CURRENT PROMPT:
{current_prompt}

FEEDBACK: {feedback}

Return ONLY the revised system prompt text, no JSON or formatting."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return current_prompt


def _wizard_step_define_input_distribution(state: WizardState, verbose: bool = False) -> WizardState:
    """Step 3: Define user inputs distribution using file content.

    Analyzes existing data to understand input patterns and helps
    user define what types of inputs to generate.

    Args:
        state: Current wizard state.
        verbose: Show detailed output.

    Returns:
        Updated state with input distribution spec.
    """
    import json

    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 3/6: Define User Inputs Distribution ━━━[/bold cyan]\n"
    )

    if verbose:
        cli_utils.CONSOLE.print(
            "[dim]Define what kinds of inputs users will send.\n"
            "This determines the diversity of your training data.[/dim]\n"
        )

    # =========================================================================
    # STEP 3a: Determine input format type
    # =========================================================================

    # Try to detect format from data
    detected_format = "single_turn"
    if state.llm_analyzer:
        with cli_utils.CONSOLE.status(
            "[dim]Detecting input format from your data...[/dim]", spinner="dots"
        ):
            detected_format = _detect_input_format(state, state.llm_analyzer)

    # Display format options
    cli_utils.CONSOLE.print("[bold]What format is your input data?[/bold]\n")

    format_choices = list(INPUT_FORMAT_TYPES.keys())
    for i, fmt_key in enumerate(format_choices, 1):
        fmt = INPUT_FORMAT_TYPES[fmt_key]
        is_detected = fmt_key == detected_format
        marker = " [green](detected)[/green]" if is_detected else ""
        cli_utils.CONSOLE.print(
            f"  [cyan][{i}][/cyan] {fmt['name']}{marker}\n"
            f"      [dim]{fmt['description']}[/dim]"
        )

    # Find default choice
    default_idx = format_choices.index(detected_format) + 1 if detected_format in format_choices else 1
    format_choice = Prompt.ask(
        "\nSelect input format",
        choices=[str(i) for i in range(1, len(format_choices) + 1)],
        default=str(default_idx),
    )
    state.input_distribution.input_format = format_choices[int(format_choice) - 1]

    # Format-specific options
    if state.input_distribution.input_format == "multi_turn":
        state.input_distribution.num_turns = IntPrompt.ask(
            "Average number of conversation turns",
            default=3,
        )
    elif state.input_distribution.input_format == "document":
        doc_types = ["article", "report", "code", "email", "transcript", "other"]
        cli_utils.CONSOLE.print("\n[bold]Document type:[/bold]")
        for i, dt in enumerate(doc_types, 1):
            cli_utils.CONSOLE.print(f"  [{i}] {dt}")
        doc_choice = Prompt.ask(
            "Select",
            choices=[str(i) for i in range(1, len(doc_types) + 1)],
            default="1",
        )
        state.input_distribution.document_type = doc_types[int(doc_choice) - 1]

    cli_utils.CONSOLE.print(
        f"\n[green]✓ Input format:[/green] {INPUT_FORMAT_TYPES[state.input_distribution.input_format]['name']}"
    )

    # =========================================================================
    # STEP 3b: Analyze input patterns based on format
    # =========================================================================

    # Find input-like columns from data
    input_columns = []
    if state.primary_schema and state.primary_schema.columns:
        for col in state.primary_schema.columns:
            col_name_lower = col.name.lower()
            if any(kw in col_name_lower for kw in ["question", "query", "input", "request", "prompt", "message", "text", "content", "document"]):
                input_columns.append(col)
            elif col.is_text and (col.avg_length or 0) > 20:
                input_columns.append(col)

    if state.llm_analyzer:
        # Analyze input distribution from files with format context
        with cli_utils.CONSOLE.status(
            "[dim]Analyzing input patterns from your data...[/dim]", spinner="dots"
        ):
            input_analysis = _analyze_input_distribution(state, state.llm_analyzer)

        input_types = input_analysis.get("input_types", [])
        sample_inputs = input_analysis.get("sample_inputs", [])
        source_column = input_analysis.get("source_column", "")
        variation_strategies = input_analysis.get("variation_strategies", [])

        # Ensure lists contain strings
        if not isinstance(input_types, list):
            input_types = [str(input_types)] if input_types else []
        input_types = [str(t) for t in input_types]

        if not isinstance(sample_inputs, list):
            sample_inputs = [str(sample_inputs)] if sample_inputs else []
        sample_inputs = [str(s) for s in sample_inputs]

        if not isinstance(variation_strategies, list):
            variation_strategies = [str(variation_strategies)] if variation_strategies else []
        variation_strategies = [str(v) for v in variation_strategies]

        source_column = str(source_column) if source_column else ""

        # Display analysis
        cli_utils.CONSOLE.print(
            Panel(
                f"[bold]Input Types Found:[/bold]\n"
                + "\n".join(f"  • {t}" for t in input_types[:5])
                + f"\n\n[bold]Sample Inputs:[/bold]\n"
                + "\n".join(f"  \"{s[:100]}{'...' if len(s) > 100 else ''}\"" for s in sample_inputs[:3])
                + (f"\n\n[bold]Source Column:[/bold] {source_column}" if source_column else "")
                + f"\n\n[bold]Variation Strategies:[/bold]\n"
                + "\n".join(f"  • {v}" for v in variation_strategies[:4]),
                title="[green]Input Distribution Analysis[/green]",
                border_style="green",
            )
        )

        # Let user confirm or modify
        use_analysis = Confirm.ask("\nUse this input distribution?", default=True)

        if use_analysis:
            state.input_distribution.input_types = input_types
            state.input_distribution.sample_inputs = sample_inputs
            state.input_distribution.source_column = source_column
            state.input_distribution.variation_strategies = variation_strategies
        else:
            # Manual definition
            state.input_distribution = _manual_input_distribution(state, input_columns)

        # Generate input template
        with cli_utils.CONSOLE.status(
            "[dim]Creating input generation template...[/dim]", spinner="dots"
        ):
            template = _generate_input_template(state, state.llm_analyzer)
            state.input_distribution.input_template = template

        cli_utils.CONSOLE.print(
            Panel(
                state.input_distribution.input_template,
                title="[yellow]Input Generation Template[/yellow]",
                border_style="yellow",
            )
        )

        # Refinement loop for input template
        if Confirm.ask("\nWould you like to refine this template?", default=False):
            feedback = Prompt.ask("What should be different?")
            refined = _refine_input_template(
                state.input_distribution.input_template, feedback, state.llm_analyzer
            )
            state.input_distribution.input_template = refined
            cli_utils.CONSOLE.print(
                Panel(
                    refined,
                    title="[yellow]Revised Input Template[/yellow]",
                    border_style="yellow",
                )
            )
    else:
        # Manual input distribution
        state.input_distribution = _manual_input_distribution(state, input_columns)

    cli_utils.CONSOLE.print(
        "\n[green]✓ Input distribution defined[/green]"
    )

    return state


def _detect_input_format(state: WizardState, llm_analyzer) -> str:
    """Detect the input format from the data.

    Args:
        state: Wizard state with file analysis.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        One of: single_turn, multi_turn, document, structured, instruction
    """
    import json

    # Build context
    context_parts = []
    context_parts.append(f"Task: {state.task.name}")
    context_parts.append(f"Description: {state.task.description}")
    context_parts.append(f"Example input: {state.task.example_input}")

    # Add column info
    if state.primary_schema and state.primary_schema.columns:
        cols_info = []
        for col in state.primary_schema.columns[:10]:
            col_info = f"{col.name} ({col.dtype})"
            if col.avg_length:
                col_info += f" [avg_len={col.avg_length}]"
            cols_info.append(col_info)
        context_parts.append(f"Columns: {', '.join(cols_info)}")

    # Add sample rows
    if state.primary_schema and state.primary_schema.sample_rows:
        samples = json.dumps(state.primary_schema.sample_rows[:2], indent=2)[:800]
        context_parts.append(f"Sample data:\n{samples}")

    prompt = f"""Analyze this data to determine the input format type.

{chr(10).join(context_parts)}

Determine which format best describes the INPUT data:

1. single_turn - Simple question/answer pairs, one input one output
2. multi_turn - Conversation with multiple back-and-forth exchanges
3. document - Long-form text like articles, reports, code, emails
4. structured - JSON objects, database records, structured data
5. instruction - Task instructions with separate input data

Look for clues like:
- Column names (messages, conversation, turns → multi_turn)
- Text length (very long → document)
- JSON/dict-like content → structured
- Short questions → single_turn
- "instruction" + "input" columns → instruction

Return JSON:
{{
    "format": "single_turn|multi_turn|document|structured|instruction",
    "confidence": "high|medium|low",
    "reasoning": "brief explanation"
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        fmt = result.get("format", "single_turn")
        if fmt in INPUT_FORMAT_TYPES:
            return fmt
        return "single_turn"
    except Exception:
        # Fallback: heuristic detection
        if state.primary_schema and state.primary_schema.columns:
            col_names = [c.name.lower() for c in state.primary_schema.columns]

            # Check for conversation indicators
            if any(kw in " ".join(col_names) for kw in ["message", "conversation", "turn", "dialogue"]):
                return "multi_turn"

            # Check for document indicators
            if any(kw in " ".join(col_names) for kw in ["document", "article", "content", "body", "text"]):
                # Check average length
                for col in state.primary_schema.columns:
                    if col.is_text and col.avg_length and col.avg_length > 500:
                        return "document"

            # Check for instruction indicators
            if any(kw in " ".join(col_names) for kw in ["instruction", "task"]):
                return "instruction"

            # Check for structured data
            if state.primary_schema.sample_rows:
                sample = state.primary_schema.sample_rows[0]
                if any(isinstance(v, (dict, list)) for v in sample.values()):
                    return "structured"

        return "single_turn"


def _analyze_input_distribution(state: WizardState, llm_analyzer) -> dict:
    """Analyze files to understand input distribution.

    Args:
        state: Wizard state with file analysis.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with input distribution analysis.
    """
    import json

    input_format = state.input_distribution.input_format

    # Build context
    context_parts = []
    context_parts.append(f"Task: {state.task.name}")
    context_parts.append(f"Description: {state.task.description}")
    context_parts.append(f"Example input: {state.task.example_input}")
    context_parts.append(f"Input format: {input_format} - {INPUT_FORMAT_TYPES.get(input_format, {}).get('description', '')}")

    # Add column info
    if state.primary_schema and state.primary_schema.columns:
        cols_info = []
        for col in state.primary_schema.columns[:10]:
            col_info = f"{col.name} ({col.dtype})"
            if col.sample_values:
                col_info += f" e.g., '{str(col.sample_values[0])[:50]}'"
            cols_info.append(col_info)
        context_parts.append(f"Columns: {', '.join(cols_info)}")

    # Add sample rows
    if state.primary_schema and state.primary_schema.sample_rows:
        samples = json.dumps(state.primary_schema.sample_rows[:3], indent=2)[:1000]
        context_parts.append(f"Sample data:\n{samples}")

    # Format-specific analysis prompt
    format_guidance = {
        "single_turn": "Focus on the types of questions/requests users will ask.",
        "multi_turn": "Identify conversation patterns, typical turn sequences, and dialogue styles.",
        "document": "Identify document types, typical lengths, and content patterns.",
        "structured": "Identify the schema structure, required fields, and data patterns.",
        "instruction": "Identify instruction types and input data patterns.",
    }

    prompt = f"""Analyze this data to understand the input distribution for ML training.

{chr(10).join(context_parts)}

{format_guidance.get(input_format, '')}

Based on the data:
1. What types of inputs will users send?
2. What are representative examples from the data?
3. Which column contains user inputs (if any)?
4. How should we vary inputs for diversity?

Return JSON:
{{
    "input_types": ["type1", "type2", "type3"],
    "sample_inputs": ["example input 1", "example input 2", "example input 3"],
    "source_column": "column_name or empty string if none",
    "variation_strategies": ["strategy1", "strategy2"]
}}

Return ONLY the JSON object."""

    try:
        return llm_analyzer._invoke_json(prompt)
    except Exception:
        return {
            "input_types": ["General questions", "Specific requests"],
            "sample_inputs": [state.task.example_input],
            "source_column": "",
            "variation_strategies": ["Rephrase", "Change topic"],
        }


def _manual_input_distribution(state: WizardState, input_columns: list) -> InputDistributionSpec:
    """Manually define input distribution.

    Args:
        state: Wizard state.
        input_columns: Candidate input columns from data.

    Returns:
        InputDistributionSpec with manual definition.
    """
    spec = InputDistributionSpec()

    # Select source column if available
    if input_columns:
        cli_utils.CONSOLE.print("\n[bold]Potential input columns found:[/bold]")
        for i, col in enumerate(input_columns[:5], 1):
            sample = col.sample_values[0] if col.sample_values else ""
            cli_utils.CONSOLE.print(f"  [{i}] {col.name}: \"{str(sample)[:50]}...\"")

        if Confirm.ask("\nUse one of these as input source?", default=True):
            col_choice = Prompt.ask(
                "Select column",
                choices=[str(i) for i in range(1, len(input_columns[:5]) + 1)],
                default="1",
            )
            spec.source_column = input_columns[int(col_choice) - 1].name

    # Define input types
    cli_utils.CONSOLE.print("\n[bold]What types of inputs will users send?[/bold]")
    input_types = []
    for i in range(1, 4):
        inp_type = Prompt.ask(f"Input type {i} (or 'done')", default="done" if i > 1 else "")
        if inp_type.lower() == "done":
            break
        input_types.append(inp_type)
    spec.input_types = input_types or ["General questions"]

    # Add sample inputs
    spec.sample_inputs = [state.task.example_input] if state.task.example_input else []

    return spec


def _generate_input_template(state: WizardState, llm_analyzer) -> str:
    """Generate a template for creating diverse inputs.

    Args:
        state: Wizard state.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Input generation template string.
    """
    import json

    columns = []
    if state.primary_schema and state.primary_schema.columns:
        columns = [c.name for c in state.primary_schema.columns]

    input_format = state.input_distribution.input_format
    format_info = INPUT_FORMAT_TYPES.get(input_format, {})

    # Format-specific instructions
    format_instructions = {
        "single_turn": """Create a template for generating single questions/requests.
The template should produce one-off queries that users might ask.""",

        "multi_turn": f"""Create a template for generating multi-turn conversations with {state.input_distribution.num_turns} turns.
The template should produce a dialogue with multiple exchanges.
Use format:
User: <message>
Assistant: <response>
User: <follow-up>
...""",

        "document": f"""Create a template for generating {state.input_distribution.document_type or 'document'} inputs.
The template should produce long-form text content to be processed.
Documents should be realistic and varied in structure.""",

        "structured": """Create a template for generating structured/JSON inputs.
The template should produce valid JSON objects or structured records.
Use format: {"field1": "<value>", "field2": "<value>", ...}""",

        "instruction": """Create a template for generating instruction+input pairs.
Format should be:
Instruction: <what to do>
Input: <data to process>""",
    }

    prompt = f"""Create a template for generating diverse inputs for this task.

Task: {state.task.name}
Description: {state.task.description}
Input format: {format_info.get('name', input_format)} - {format_info.get('description', '')}
Input types: {state.input_distribution.input_types}
Sample inputs: {state.input_distribution.sample_inputs[:2]}
Available columns: {columns}

{format_instructions.get(input_format, '')}

Use {{column_name}} placeholders where data from columns should be inserted.

The template should:
1. Generate diverse variations matching the {input_format} format
2. Cover different scenarios and edge cases
3. Be realistic and natural

Return ONLY the template text, no JSON."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        # Fallback templates based on format
        fallbacks = {
            "single_turn": "Generate a question about {context} that a user might ask.",
            "multi_turn": "Generate a conversation about {context} with 3 turns.\n\nUser: <question>\nAssistant: <response>\nUser: <follow-up>",
            "document": "Generate a {document_type} about {context}.",
            "structured": 'Generate a JSON object with fields from {context}:\n{{"field1": "value", "field2": "value"}}',
            "instruction": "Instruction: Process the following {context}\nInput: {input_data}",
        }
        return fallbacks.get(input_format, "Generate an input based on {context}.")


def _refine_input_template(current: str, feedback: str, llm_analyzer) -> str:
    """Refine input template based on feedback.

    Args:
        current: Current template.
        feedback: User feedback.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Refined template.
    """
    prompt = f"""Revise this input generation template based on feedback.

CURRENT TEMPLATE:
{current}

FEEDBACK: {feedback}

Return ONLY the revised template text."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return current


def _wizard_step_define_output_quality(state: WizardState, verbose: bool = False) -> WizardState:
    """Step 4: Define model output quality criteria and formats.

    Guides the user to specify what good outputs look like,
    using examples from their data.

    Args:
        state: Current wizard state.
        verbose: Show detailed output.

    Returns:
        Updated state with output quality spec.
    """
    import json

    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 4/6: Define Output Quality & Format ━━━[/bold cyan]\n"
    )

    if verbose:
        cli_utils.CONSOLE.print(
            "[dim]Define what high-quality outputs look like.\n"
            "This guides both generation and evaluation.[/dim]\n"
        )

    # Find output-like columns
    output_columns = []
    if state.primary_schema and state.primary_schema.columns:
        for col in state.primary_schema.columns:
            col_name_lower = col.name.lower()
            if any(kw in col_name_lower for kw in ["answer", "response", "output", "reply", "result"]):
                output_columns.append(col)
            elif col.is_text and (col.avg_length or 0) > 100:
                output_columns.append(col)

    if state.llm_analyzer:
        # Analyze output quality from files
        with cli_utils.CONSOLE.status(
            "[dim]Analyzing output patterns from your data...[/dim]", spinner="dots"
        ):
            output_analysis = _analyze_output_quality(state, state.llm_analyzer)

        format_type = output_analysis.get("format_type", "text")
        format_template = output_analysis.get("format_template", "")
        quality_criteria = output_analysis.get("quality_criteria", [])
        good_examples = output_analysis.get("good_examples", [])
        output_prefix = output_analysis.get("output_prefix", "Answer:")

        # Ensure quality_criteria and good_examples are lists of strings
        if not isinstance(quality_criteria, list):
            quality_criteria = [str(quality_criteria)] if quality_criteria else []
        quality_criteria = [str(c) for c in quality_criteria]

        if not isinstance(good_examples, list):
            good_examples = [str(good_examples)] if good_examples else []
        good_examples = [str(e) for e in good_examples]

        # Format example output for display
        example_output_str = ""
        if good_examples and len(good_examples) > 0:
            example_text = str(good_examples[0])[:200]
            example_output_str = f"\n\n[bold]Example Output:[/bold]\n  {example_text}..."

        # Display analysis
        cli_utils.CONSOLE.print(
            Panel(
                f"[bold]Output Format:[/bold] {format_type}\n\n"
                f"[bold]Quality Criteria:[/bold]\n"
                + "\n".join(f"  • {c}" for c in quality_criteria[:5])
                + f"\n\n[bold]Output Prefix:[/bold] {output_prefix}"
                + example_output_str,
                title="[green]Output Quality Analysis[/green]",
                border_style="green",
            )
        )

        # Let user confirm or modify
        use_analysis = Confirm.ask("\nUse these quality criteria?", default=True)

        if use_analysis:
            state.output_quality.format_type = format_type
            state.output_quality.format_template = format_template
            state.output_quality.quality_criteria = quality_criteria
            state.output_quality.good_examples = good_examples
            state.output_quality.output_prefix = output_prefix
        else:
            # Manual definition
            state.output_quality = _manual_output_quality(state, output_columns)

        # Generate output template
        with cli_utils.CONSOLE.status(
            "[dim]Creating output generation template...[/dim]", spinner="dots"
        ):
            template = _generate_output_template(state, state.llm_analyzer)
            state.output_quality.format_template = template

        cli_utils.CONSOLE.print(
            Panel(
                state.output_quality.format_template,
                title="[magenta]Output Generation Template[/magenta]",
                border_style="magenta",
            )
        )

        # Refinement loop
        if Confirm.ask("\nWould you like to refine this template?", default=False):
            feedback = Prompt.ask("What should be different?")
            refined = _refine_output_template(
                state.output_quality.format_template, feedback, state.llm_analyzer
            )
            state.output_quality.format_template = refined
            cli_utils.CONSOLE.print(
                Panel(
                    refined,
                    title="[magenta]Revised Output Template[/magenta]",
                    border_style="magenta",
                )
            )
    else:
        state.output_quality = _manual_output_quality(state, output_columns)

    cli_utils.CONSOLE.print(
        "\n[green]✓ Output quality criteria defined[/green]"
    )

    return state


def _analyze_output_quality(state: WizardState, llm_analyzer) -> dict:
    """Analyze files to understand output quality patterns.

    Args:
        state: Wizard state.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with output quality analysis.
    """
    import json

    context_parts = []
    context_parts.append(f"Task: {state.task.name}")
    context_parts.append(f"Description: {state.task.description}")
    context_parts.append(f"Example output: {state.task.example_output}")

    # Add sample data
    if state.primary_schema and state.primary_schema.sample_rows:
        samples = json.dumps(state.primary_schema.sample_rows[:2], indent=2)[:800]
        context_parts.append(f"Sample data:\n{samples}")

    # Add domain analysis
    if state.domain_analysis:
        context_parts.append(f"Quality signals: {state.domain_analysis.quality_signals}")
        context_parts.append(f"Common issues: {state.domain_analysis.common_issues}")

    prompt = f"""Analyze this to define output quality criteria for ML training.

{chr(10).join(context_parts)}

Define:
1. What format should outputs be in?
2. What makes a high-quality output?
3. What prefix should outputs have (e.g., "Answer:")?
4. Example of good output

Return JSON:
{{
    "format_type": "text|json|structured|list",
    "format_template": "template showing output structure",
    "quality_criteria": ["criterion1", "criterion2", "criterion3"],
    "good_examples": ["example good output"],
    "output_prefix": "prefix like 'Answer:' or empty"
}}

Return ONLY the JSON object."""

    try:
        return llm_analyzer._invoke_json(prompt)
    except Exception:
        return {
            "format_type": "text",
            "format_template": "Provide a clear, helpful response.",
            "quality_criteria": ["Accurate", "Clear", "Complete"],
            "good_examples": [state.task.example_output],
            "output_prefix": "Answer:",
        }


def _manual_output_quality(state: WizardState, output_columns: list) -> OutputQualitySpec:
    """Manually define output quality criteria.

    Args:
        state: Wizard state.
        output_columns: Candidate output columns.

    Returns:
        OutputQualitySpec with manual definition.
    """
    spec = OutputQualitySpec()

    # Format type
    cli_utils.CONSOLE.print(
        "\n[bold]Output Format:[/bold]\n"
        "  [1] Text - Free-form text response\n"
        "  [2] JSON - Structured JSON output\n"
        "  [3] List - Bulleted or numbered list\n"
    )
    format_choice = Prompt.ask("Select format", choices=["1", "2", "3"], default="1")
    format_types = {"1": "text", "2": "json", "3": "list"}
    spec.format_type = format_types[format_choice]

    # Output prefix
    spec.output_prefix = Prompt.ask(
        "Output prefix (e.g., 'Answer:')",
        default="Answer:" if spec.format_type == "text" else "",
    )

    # Quality criteria
    cli_utils.CONSOLE.print("\n[bold]Quality Criteria (what makes a good output?):[/bold]")
    criteria = []
    default_criteria = ["Accurate", "Clear", "Complete"]
    for i, default in enumerate(default_criteria, 1):
        criterion = Prompt.ask(f"Criterion {i}", default=default)
        criteria.append(criterion)
    spec.quality_criteria = criteria

    # Good example
    spec.good_examples = [state.task.example_output] if state.task.example_output else []

    return spec


def _generate_output_template(state: WizardState, llm_analyzer) -> str:
    """Generate a template for creating outputs.

    Args:
        state: Wizard state.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Output generation template string.
    """
    prompt = f"""Create an instruction template for generating high-quality outputs.

Task: {state.task.name}
System prompt: {state.system_prompt.full_prompt[:200]}...
Output format: {state.output_quality.format_type}
Quality criteria: {state.output_quality.quality_criteria}
Output prefix: {state.output_quality.output_prefix}

Create an instruction that tells the model how to generate good outputs.
It should reference {{question}} for the user's input.

The template should:
1. Guide the model to produce {state.output_quality.format_type} format
2. Ensure the quality criteria are met
3. Use the output prefix {state.output_quality.output_prefix}

Return ONLY the template text, no JSON."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return f"Provide a helpful response to: {{question}}\n\nFormat as: {state.output_quality.output_prefix} <your response>"


def _refine_output_template(current: str, feedback: str, llm_analyzer) -> str:
    """Refine output template based on feedback."""
    prompt = f"""Revise this output template based on feedback.

CURRENT:
{current}

FEEDBACK: {feedback}

Return ONLY the revised template text."""

    try:
        return llm_analyzer._invoke(prompt).strip()
    except Exception:
        return current


def _wizard_step_generate_synthesis_config(state: WizardState, output_path: Path, verbose: bool = False) -> str:
    """Step 5: Generate synthesis config for (input, output) pairs.

    Creates a synthesis configuration that generates training data
    matching the problem definition and covering the input distribution.

    Args:
        state: Wizard state with all specifications.
        output_path: Output directory.
        verbose: Show detailed output.

    Returns:
        Path to the generated config file.
    """
    from oumi.onboarding.config_builder import SynthConfigBuilder

    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 5/6: Generate Synthesis Config ━━━[/bold cyan]\n"
    )

    if verbose:
        cli_utils.CONSOLE.print(
            "[dim]Creating a synthesis config that generates (input, output) pairs\n"
            "matching your task definition and covering the input distribution.[/dim]\n"
        )

    # Ask for number of samples
    cli_utils.CONSOLE.print(
        "[dim]Tip: Start small (10-50) to verify quality, then scale up.[/dim]"
    )
    num_samples = IntPrompt.ask("Number of samples to generate", default=100)

    # Build the config using the wizard state
    builder = SynthConfigBuilder()

    # Build attribute map from source column if available
    attribute_map = None
    if state.input_distribution.source_column:
        attribute_map = {state.input_distribution.source_column: "context"}

    # Create config with custom prompts from wizard state
    config = builder.from_schema_with_custom_prompts(
        state.primary_schema,
        goal="qa",  # Default to Q&A, the most common
        num_samples=num_samples,
        output_path=str(output_path / "synth_output.jsonl"),
        system_prompt=state.system_prompt.full_prompt,
        question_template=state.input_distribution.input_template,
        answer_template=state.output_quality.format_template,
        postprocessing={
            "cut_prefix": state.output_quality.output_prefix,
            "strip_whitespace": True,
        },
        attribute_map=attribute_map,
    )

    # Save config
    config_path = output_path / "synth_config.yaml"
    config.to_yaml(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]✓ Synthesis config saved: {config_path}[/green]")

    # Display summary
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Samples to generate:[/bold] {num_samples}\n"
            f"[bold]Output file:[/bold] {output_path / 'synth_output.jsonl'}\n"
            f"[bold]Model:[/bold] {config.inference_config.model.model_name}\n\n"
            f"[bold]System Prompt:[/bold]\n{state.system_prompt.full_prompt[:200]}...\n\n"
            f"[bold]Input Template:[/bold]\n{state.input_distribution.input_template[:150]}...\n\n"
            f"[bold]Output Template:[/bold]\n{state.output_quality.format_template[:150]}...",
            title="[green]Synthesis Configuration[/green]",
            border_style="green",
        )
    )

    return str(config_path)


def _wizard_step_define_judges(state: WizardState, output_path: Path, verbose: bool = False) -> str:
    """Step 6: Define judges based on quality criteria.

    Creates judge configurations to evaluate generated data
    based on the output quality criteria defined earlier.
    Suggests custom judges tailored to the specific criteria.

    Args:
        state: Wizard state with output quality spec.
        output_path: Output directory.
        verbose: Show detailed output.

    Returns:
        Path to the generated judge config file.
    """
    from oumi.onboarding.config_builder import JudgeConfigBuilder

    cli_utils.CONSOLE.print(
        "\n[bold cyan]━━━ Step 6/6: Define Judges ━━━[/bold cyan]\n"
    )

    if verbose:
        cli_utils.CONSOLE.print(
            "[dim]Create judges to automatically evaluate the quality of generated data.\n"
            "We'll suggest judges based on your quality criteria from Step 4.[/dim]\n"
        )

    # Show quality criteria from earlier
    cli_utils.CONSOLE.print(
        Panel(
            "[bold]Your Quality Criteria:[/bold]\n"
            + "\n".join(f"  • {c}" for c in state.output_quality.quality_criteria),
            title="From Step 4",
            border_style="cyan",
        )
    )

    # Use LLM to suggest custom judges based on criteria
    suggested_judges = []
    if state.llm_analyzer:
        with cli_utils.CONSOLE.status(
            "[dim]Analyzing criteria to suggest judges...[/dim]", spinner="dots"
        ):
            suggested_judges = _suggest_judges_from_criteria(state, state.llm_analyzer)

    if suggested_judges:
        cli_utils.CONSOLE.print(
            "\n[bold]Suggested Judges Based on Your Criteria:[/bold]\n"
        )
        for i, judge in enumerate(suggested_judges, 1):
            name = judge.get("name", f"Judge {i}")
            description = judge.get("description", "")
            criteria = judge.get("criteria", "")
            cli_utils.CONSOLE.print(
                f"  [cyan][{i}][/cyan] [bold]{name}[/bold]\n"
                f"      [dim]{description}[/dim]\n"
                f"      [dim]Evaluates: {criteria}[/dim]\n"
            )

        # Let user select which judges to use
        cli_utils.CONSOLE.print(
            f"  [cyan][{len(suggested_judges) + 1}][/cyan] [bold]Add custom judge[/bold]\n"
            f"      [dim]Define your own evaluation criteria[/dim]\n"
        )

        selected = Prompt.ask(
            "\nSelect judges (comma-separated, e.g., '1,2' or 'all')",
            default="all",
        )

        if selected.lower() == "all":
            selected_judges = suggested_judges
        else:
            selected_indices = [int(x.strip()) - 1 for x in selected.split(",") if x.strip().isdigit()]
            selected_judges = []
            for idx in selected_indices:
                if 0 <= idx < len(suggested_judges):
                    selected_judges.append(suggested_judges[idx])
                elif idx == len(suggested_judges):
                    # Custom judge
                    custom = _prompt_custom_judge()
                    if custom:
                        selected_judges.append(custom)
    else:
        # No LLM or suggestion failed - manual judge definition
        cli_utils.CONSOLE.print(
            "\n[bold]Define your judges:[/bold]\n"
            "[dim]Judges evaluate output quality based on specific criteria.[/dim]\n"
        )
        selected_judges = []

        # Suggest based on keywords in criteria
        criteria_text = " ".join(state.output_quality.quality_criteria).lower()

        default_judges = []
        if any(kw in criteria_text for kw in ["accurate", "factual", "correct", "true"]):
            default_judges.append({
                "name": "Accuracy Judge",
                "description": "Evaluates factual correctness",
                "criteria": "Is the response factually accurate and correct?",
                "type": "accuracy",
            })
        if any(kw in criteria_text for kw in ["relevant", "address", "answer", "helpful"]):
            default_judges.append({
                "name": "Relevance Judge",
                "description": "Evaluates if response addresses the query",
                "criteria": "Does the response directly address the user's question?",
                "type": "relevance",
            })
        if any(kw in criteria_text for kw in ["clear", "readable", "understand", "coherent"]):
            default_judges.append({
                "name": "Clarity Judge",
                "description": "Evaluates clarity and readability",
                "criteria": "Is the response clear, well-organized, and easy to understand?",
                "type": "clarity",
            })
        if any(kw in criteria_text for kw in ["complete", "thorough", "comprehensive"]):
            default_judges.append({
                "name": "Completeness Judge",
                "description": "Evaluates completeness of response",
                "criteria": "Does the response fully address all aspects of the query?",
                "type": "completeness",
            })
        if any(kw in criteria_text for kw in ["safe", "appropriate", "harmful", "toxic"]):
            default_judges.append({
                "name": "Safety Judge",
                "description": "Evaluates content safety",
                "criteria": "Is the response safe, appropriate, and free of harmful content?",
                "type": "safety",
            })
        if any(kw in criteria_text for kw in ["format", "structure", "json", "style"]):
            default_judges.append({
                "name": "Format Judge",
                "description": "Evaluates format compliance",
                "criteria": "Does the response follow the expected format and structure?",
                "type": "format",
            })

        # If no specific judges found, add a generic one
        if not default_judges:
            default_judges.append({
                "name": "Quality Judge",
                "description": "Overall quality assessment",
                "criteria": "; ".join(state.output_quality.quality_criteria),
                "type": "generic",
            })

        # Show default judges and let user modify
        for i, judge in enumerate(default_judges, 1):
            cli_utils.CONSOLE.print(
                f"  [cyan][{i}][/cyan] {judge['name']}: {judge['description']}"
            )

        use_defaults = Confirm.ask("\nUse these judges?", default=True)
        if use_defaults:
            selected_judges = default_judges
        else:
            # Let user define custom judges
            selected_judges = []
            while True:
                custom = _prompt_custom_judge()
                if custom:
                    selected_judges.append(custom)
                if not Confirm.ask("Add another judge?", default=False):
                    break

    # Ensure at least one judge
    if not selected_judges:
        selected_judges = [{
            "name": "Quality Judge",
            "description": "Overall quality assessment",
            "criteria": "; ".join(state.output_quality.quality_criteria),
            "type": "generic",
        }]

    # Build and save judge configs
    builder = JudgeConfigBuilder()
    config_paths = []

    for i, judge in enumerate(selected_judges):
        judge_name = judge.get("name", f"judge_{i+1}").lower().replace(" ", "_")
        judge_criteria = judge.get("criteria", "")
        judge_type = judge.get("type", "custom")

        # Create config
        config = builder.from_custom_criteria(
            schema=state.primary_schema,
            judge_name=judge_name,
            criteria=judge_criteria,
            description=judge.get("description", ""),
        )

        # Save config
        config_filename = f"judge_{judge_name}.yaml" if len(selected_judges) > 1 else "judge_config.yaml"
        config_path = output_path / config_filename
        config.to_yaml(str(config_path))
        config_paths.append(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]✓ {len(selected_judges)} judge config(s) saved[/green]")

    # Display summary
    judges_summary = "\n".join(
        f"  • [bold]{j.get('name', 'Judge')}[/bold]: {j.get('criteria', '')[:60]}..."
        for j in selected_judges
    )
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Judges Configured:[/bold] {len(selected_judges)}\n\n"
            f"{judges_summary}\n\n"
            f"[bold]Config Files:[/bold]\n"
            + "\n".join(f"  • {p}" for p in config_paths),
            title="[green]Judge Configuration[/green]",
            border_style="green",
        )
    )

    return config_paths[0] if config_paths else str(output_path / "judge_config.yaml")


def _suggest_judges_from_criteria(state: WizardState, llm_analyzer) -> list:
    """Use LLM to suggest custom judges based on quality criteria.

    Args:
        state: Wizard state with quality criteria.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        List of suggested judge dicts with name, description, criteria.
    """
    criteria_list = state.output_quality.quality_criteria
    task_name = state.task.name
    task_desc = state.task.description

    prompt = f"""Based on these quality criteria for a "{task_name}" task, suggest specific judges to evaluate outputs.

Task: {task_name}
Description: {task_desc}

Quality Criteria:
{chr(10).join(f"- {c}" for c in criteria_list)}

For each criterion (or group of related criteria), suggest a specialized judge.
Each judge should have a clear, specific evaluation focus.

Examples of good judges:
- "Domain Accuracy Judge" - checks domain-specific facts are correct
- "Tone Consistency Judge" - ensures response matches expected tone
- "Completeness Judge" - verifies all parts of query are addressed
- "Format Compliance Judge" - checks output follows required structure

Return JSON array:
[
    {{
        "name": "Descriptive Judge Name",
        "description": "What this judge evaluates",
        "criteria": "Specific evaluation criteria as a question or checklist",
        "type": "category (accuracy/relevance/safety/format/tone/completeness/custom)"
    }},
    ...
]

Suggest 2-4 judges that together cover all the quality criteria.
Return ONLY the JSON array."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []


def _prompt_custom_judge() -> Optional[dict]:
    """Prompt user to define a custom judge.

    Returns:
        Dict with judge definition or None if cancelled.
    """
    cli_utils.CONSOLE.print("\n[bold]Define Custom Judge:[/bold]")

    name = Prompt.ask("Judge name", default="Custom Judge")
    if not name:
        return None

    description = Prompt.ask("What does this judge evaluate?", default="Overall quality")
    criteria = Prompt.ask(
        "Evaluation criteria (question or checklist)",
        default="Is the response high quality?",
    )

    return {
        "name": name,
        "description": description,
        "criteria": criteria,
        "type": "custom",
    }



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


def _identify_task_from_files(files: list[dict], llm_analyzer) -> dict:
    """Use AI to identify what task the user is trying to accomplish.

    Args:
        files: List of file info dicts with analysis results.
        llm_analyzer: LLMAnalyzer instance.

    Returns:
        Dict with task suggestions: {
            "primary_task": str,
            "task_description": str,
            "alternatives": [{"task": str, "description": str}, ...],
            "reasoning": str
        }
    """
    # Build context from analyzed files
    file_summaries = []
    for f in files:
        summary = f"- {f['name']} ({f['extension']})"
        if f.get("suggested_purpose"):
            summary += f": {f['suggested_purpose']}"
        if f.get("schema") and f["schema"].columns:
            cols = [c.name for c in f["schema"].columns[:10]]
            summary += f" [columns: {', '.join(cols)}]"
        file_summaries.append(summary)

    prompt = f"""Based on these files, identify what ML task the user is likely trying to accomplish.

FILES:
{chr(10).join(file_summaries)}

Common ML data tasks (with examples):

1. **Q&A System** - Model answers questions using knowledge base
   Example: "What is the return policy?" → Model looks up docs and answers
   Input: question | Output: answer grounded in documents

2. **Customer Support Bot** - Handle support tickets and inquiries
   Example: "My order hasn't arrived" → Empathetic response + next steps
   Input: customer message | Output: helpful response following guidelines

3. **Content Classification** - Categorize text into predefined labels
   Example: Email → "urgent" / "billing" / "general inquiry"
   Input: text | Output: category label

4. **Data Extraction** - Pull structured fields from unstructured text
   Example: Invoice PDF → {{vendor, amount, date, line_items}}
   Input: document | Output: structured JSON

5. **Content Generation** - Create content matching style/format
   Example: Product specs → Marketing description
   Input: seed data | Output: generated content

6. **Conversation Agent** - Multi-turn dialogue with memory
   Example: Back-and-forth troubleshooting session
   Input: conversation history | Output: next response

7. **Compliance Checker** - Verify content follows rules
   Example: Response text → Does it follow brand guidelines? (yes/no + issues)
   Input: content to check | Output: pass/fail + reasoning

8. **Summarization** - Condense long content
   Example: 10-page report → 3-bullet executive summary
   Input: long text | Output: concise summary

Analyze the files and suggest the most likely task and 2-3 alternatives.

Return a JSON object:
{{
    "primary_task": "Short task name",
    "task_description": "What the model will do (1-2 sentences)",
    "example_input": "A concrete example of what users will send",
    "example_output": "What the model should respond with",
    "alternatives": [
        {{"task": "Alternative name", "description": "Why this might fit", "example": "Brief example"}},
        {{"task": "Another alternative", "description": "Why this might fit", "example": "Brief example"}}
    ],
    "reasoning": "Why you chose this task based on file contents (cite specific files/columns)"
}}

Return ONLY the JSON object."""

    try:
        result = llm_analyzer._invoke_json(prompt)
        return result
    except Exception as e:
        return {
            "primary_task": "Custom Task",
            "task_description": "Unable to determine - please describe your task",
            "alternatives": [],
            "reasoning": f"Analysis failed: {str(e)[:100]}"
        }


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
                "[bold]How your columns map to training data[/bold]\n\n"
                "[dim]The AI analyzed your columns and suggested how each\n"
                "contributes to generating training examples.[/dim]",
                border_style="cyan",
            )
        )
        # Full table with all details
        table = Table(title="Column Mappings", show_edge=False, expand=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("File", style="green", no_wrap=True)
        table.add_column("Column", style="yellow", no_wrap=True)
        table.add_column("Type", style="dim", no_wrap=True)
        table.add_column("Maps To", style="magenta", no_wrap=True)
        table.add_column("Why?", style="dim")

        for i, col_data in enumerate(all_columns, 1):
            role_key = col_data.get("suggested_role", "").lower()
            role_info = COLUMN_ROLE_GUIDANCE.get(role_key, {})
            role_title = role_info.get("title", role_key.upper())
            role_reason = col_data.get("role_reason", "")
            table.add_row(
                str(i),
                col_data["file"]["name"],
                col_data["column"],
                col_data["col_info"].dtype,
                role_title,
                role_reason,
            )
        cli_utils.CONSOLE.print(table)
    else:
        # Compact display
        cli_utils.CONSOLE.print("\n[bold]Column mappings:[/bold]")
        for i, col_data in enumerate(all_columns, 1):
            role_key = col_data.get("suggested_role", "").lower()
            role_info = COLUMN_ROLE_GUIDANCE.get(role_key, {})
            role_title = role_info.get("title", role_key.upper())
            cli_utils.CONSOLE.print(
                f"  [{i}] [yellow]{col_data['column']}[/yellow] → [magenta]{role_title}[/magenta]"
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

    This wizard helps you build training data through 4 simple steps:
    1. Task - What should your model do?
    2. Inputs - What data will it receive?
    3. Outputs - What makes a good response?
    4. Generate - Create configs for synthesis and evaluation

    Examples:
        # Single file
        oumi onboard wizard --data ./my_data.csv

        # With AI analysis (recommended)
        oumi onboard wizard --data ./data/ --llm

        # Show detailed output
        oumi onboard wizard --data ./data/ --llm --verbose
    """
    # Delayed imports
    from oumi.onboarding import DataAnalyzer

    # Initialize wizard state
    state = WizardState()

    # Welcome message
    if verbose:
        cli_utils.CONSOLE.print(
            Panel(
                "[bold green]Welcome to the Oumi Onboarding Wizard![/bold green]\n\n"
                "4 simple steps to build training data:\n\n"
                "  [cyan]1.[/cyan] Task    - What should your model do?\n"
                "  [cyan]2.[/cyan] Inputs  - What data will it receive?\n"
                "  [cyan]3.[/cyan] Outputs - What makes a good response?\n"
                "  [cyan]4.[/cyan] Generate - Create synthesis & judge configs\n\n"
                "[dim]The wizard suggests options based on your data.[/dim]",
                title="Oumi Onboard",
                border_style="green",
            )
        )
    else:
        cli_utils.CONSOLE.print(
            "[bold green]Oumi Onboarding Wizard[/bold green]\n"
            "[dim]4 steps: Task → Inputs → Outputs → Generate[/dim]\n"
        )

    # Check if input is a directory or file
    data_path = Path(data)
    if not data_path.exists():
        cli_utils.CONSOLE.print(f"[red]Error: Path not found: {data}[/red]")
        raise typer.Exit(1)

    analyzer = DataAnalyzer()

    # Initialize LLM analyzer if requested
    if use_llm:
        valid_engines = ["ANTHROPIC", "OPENAI", "DEEPSEEK", "TOGETHER"]
        engine_upper = engine.upper()
        if engine_upper not in valid_engines:
            cli_utils.CONSOLE.print(
                f"[yellow]Warning: Unknown engine '{engine}'. "
                f"Valid options: {', '.join(valid_engines)}. Using ANTHROPIC.[/yellow]"
            )
            engine_upper = "ANTHROPIC"

        try:
            from oumi.onboarding.llm_analyzer import LLMAnalyzer

            state.llm_analyzer = LLMAnalyzer(engine=engine_upper, model=model)
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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # CACHE: Check for existing wizard state BEFORE analyzing files
    # =========================================================================

    cache_path = _get_cache_path(output_path)
    cached_state = _load_wizard_cache(output_path)
    use_cached_analysis = False

    if cached_state:
        cli_utils.CONSOLE.print(
            "\n[bold cyan]Found existing wizard cache![/bold cyan]"
        )
        _display_cache_summary(cached_state, cache_path)

        action = _prompt_cache_action(cache_path)

        if action == "edit":
            _open_cache_for_editing(cache_path)
            # Reload after editing
            cached_state = _load_wizard_cache(output_path)
            if not cached_state:
                cli_utils.CONSOLE.print("[yellow]Could not reload cache. Starting fresh.[/yellow]")

        if action == "restart":
            cli_utils.CONSOLE.print("[dim]Starting fresh...[/dim]")
            cache_path.unlink(missing_ok=True)
            cached_state = None

        if cached_state and action in ("resume", "edit"):
            # Restore state from cache
            state.task = cached_state.task
            state.inputs = cached_state.inputs
            state.outputs = cached_state.outputs
            state.completed_steps = cached_state.completed_steps
            use_cached_analysis = True
            cli_utils.CONSOLE.print(
                f"[green]Resuming from step {len(state.completed_steps) + 1}...[/green]\n"
            )

    # Scan and analyze files
    cli_utils.CONSOLE.print("[dim]Scanning files...[/dim]")

    if data_path.is_dir():
        files = _detect_files_in_directory(data_path)
        if not files:
            cli_utils.CONSOLE.print(
                f"[red]Error: No supported files found in {data_path}[/red]\n"
                f"[dim]Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}[/dim]"
            )
            raise typer.Exit(1)
        cli_utils.CONSOLE.print(f"[green]Found {len(files)} file(s)[/green]")
    else:
        files = [{
            "path": data_path,
            "name": data_path.name,
            "extension": data_path.suffix.lower(),
        }]

    # Compute content hashes for all files
    for f in files:
        f["content_hash"] = _compute_file_hash(f["path"])

    # Build lookup of cached file analysis by path and hash
    cached_file_analysis = {}
    if cached_state and cached_state.files:
        for cf in cached_state.files:
            if cf.get("path") and cf.get("content_hash"):
                cached_file_analysis[str(cf["path"])] = cf

    # Analyze files and store in state, using cache when content unchanged
    files_needing_llm_analysis = []
    with cli_utils.CONSOLE.status("[dim]Analyzing files...[/dim]", spinner="dots"):
        for f in files:
            file_path_str = str(f["path"])
            cached_info = cached_file_analysis.get(file_path_str)

            # Check if we can use cached analysis (same content hash)
            if (
                use_cached_analysis
                and cached_info
                and cached_info.get("content_hash") == f["content_hash"]
                and cached_info.get("suggested_purpose")
            ):
                # Use cached file analysis
                f["suggested_purpose"] = cached_info.get("suggested_purpose", "")
                f["suggested_role"] = cached_info.get("suggested_role", "")
                f["role_reason"] = cached_info.get("role_reason", "")
                if verbose:
                    cli_utils.CONSOLE.print(
                        f"[dim]Using cached analysis for {f['name']}[/dim]"
                    )
            else:
                # Need fresh analysis
                files_needing_llm_analysis.append(f)

            # Always analyze schema (it's fast and needed for the wizard)
            try:
                schema = analyzer.analyze(f["path"])
                f["schema"] = schema
                state.schemas[str(f["path"])] = schema
            except Exception as e:
                f["schema"] = None
                if verbose:
                    cli_utils.CONSOLE.print(f"[yellow]Warning: Could not analyze {f['name']}: {e}[/yellow]")

    # Run LLM analysis only for files that need it
    if use_llm and state.llm_analyzer and files_needing_llm_analysis:
        if verbose:
            cli_utils.CONSOLE.print(
                f"[dim]Running LLM analysis for {len(files_needing_llm_analysis)} file(s)...[/dim]"
            )
        files_needing_llm_analysis = _analyze_file_purposes(
            files_needing_llm_analysis, analyzer, state.llm_analyzer
        )
        # Update the original files list with analysis results
        analyzed_by_path = {str(f["path"]): f for f in files_needing_llm_analysis}
        for f in files:
            if str(f["path"]) in analyzed_by_path:
                analyzed = analyzed_by_path[str(f["path"])]
                f["suggested_purpose"] = analyzed.get("suggested_purpose", "")
                f["suggested_role"] = analyzed.get("suggested_role", "")
                f["role_reason"] = analyzed.get("role_reason", "")
    elif use_llm and state.llm_analyzer and not files_needing_llm_analysis:
        cli_utils.CONSOLE.print("[dim]All file analyses loaded from cache[/dim]")

    state.files = files

    # Set primary schema (use first tabular file or first file)
    for f in files:
        if f.get("schema") and f["schema"].columns:
            state.primary_schema = f["schema"]
            break
    if not state.primary_schema and files and files[0].get("schema"):
        state.primary_schema = files[0]["schema"]

    # Run domain analysis if LLM enabled
    # Use cached domain analysis if available and no files needed re-analysis
    if use_llm and state.llm_analyzer and state.primary_schema:
        if (
            use_cached_analysis
            and cached_state
            and cached_state.domain_analysis
            and not files_needing_llm_analysis
        ):
            # Use cached domain analysis
            state.domain_analysis = cached_state.domain_analysis
            if verbose:
                cli_utils.CONSOLE.print("[dim]Using cached domain analysis[/dim]")
        else:
            # Run fresh domain analysis
            try:
                with cli_utils.CONSOLE.status(
                    "[dim]Analyzing domain...[/dim]", spinner="dots"
                ):
                    state.domain_analysis = state.llm_analyzer.analyze(state.primary_schema)
            except Exception as e:
                if verbose:
                    cli_utils.CONSOLE.print(f"[yellow]Warning: Domain analysis failed: {e}[/yellow]")

    # =========================================================================
    # SIMPLIFIED WIZARD FLOW: 4-step process with caching
    # =========================================================================

    # Step 1: Define Task (includes system prompt)
    if "task" not in state.completed_steps:
        state = _wizard_step_task(state, verbose=verbose)
        _save_wizard_cache(state, output_path, "task")
    else:
        desc_preview = state.task.description[:50] + "..." if state.task.description else "defined"
        cli_utils.CONSOLE.print(
            f"\n[dim]Step 1/4: Task[/dim] [green]✓[/green] {desc_preview}"
        )

    # Step 2: Define Inputs
    if "inputs" not in state.completed_steps:
        state = _wizard_step_inputs(state, verbose=verbose)
        _save_wizard_cache(state, output_path, "inputs")
    else:
        input_preview = INPUT_FORMATS.get(state.inputs.format, state.inputs.format)
        cli_utils.CONSOLE.print(
            f"\n[dim]Step 2/4: Inputs[/dim] [green]✓[/green] {input_preview}"
        )

    # Step 3: Define Output Quality
    if "outputs" not in state.completed_steps:
        state = _wizard_step_outputs(state, verbose=verbose)
        _save_wizard_cache(state, output_path, "outputs")
    else:
        quality_preview = ", ".join(state.outputs.criteria[:2]) or "defined"
        cli_utils.CONSOLE.print(
            f"\n[dim]Step 3/4: Outputs[/dim] [green]✓[/green] {quality_preview}"
        )

    # Step 4: Generate Configs (synthesis + judges)
    if "generate" not in state.completed_steps:
        synth_config_path, judge_config_paths = _wizard_step_generate(state, output_path, verbose=verbose)
        _save_wizard_cache(state, output_path, "generate")
    else:
        synth_config_path = str(output_path / "synth_config.yaml")
        judge_config_paths = []
        cli_utils.CONSOLE.print(
            f"\n[dim]Step 4/4: Generate[/dim] [green]✓[/green] configs generated"
        )

    # =========================================================================
    # Summary and next steps
    # =========================================================================

    cli_utils.CONSOLE.print(
        "\n[bold green]━━━ Complete! ━━━[/bold green]\n"
    )

    # Build command list
    commands = [f"oumi synth -c {synth_config_path}"]
    if judge_config_paths:
        judge_path = judge_config_paths[0] if judge_config_paths else str(output_path / "judge_config.yaml")
        commands.append(f"oumi judge dataset -c {judge_path} --input {output_path / 'synth_output.jsonl'}")

    cli_utils.CONSOLE.print(
        Panel(
            "\n".join(f"[bold white]{i+1}. {cmd}[/bold white]" for i, cmd in enumerate(commands)),
            title="[green]Run these commands in order[/green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    cli_utils.CONSOLE.print(f"\n[dim]Configs saved to: {output_path}[/dim]")

    # Show prerequisites
    cli_utils.CONSOLE.print(
        Panel(
            "[yellow]API Key:[/yellow] Set ANTHROPIC_API_KEY or OPENAI_API_KEY env var\n\n"
            "[dim]To use a local model, edit the config and change:\n"
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
    if verbose:
        cli_utils.CONSOLE.print(
            Panel(
                f"[bold]Recommended: [green]{suggested_goal}[/green][/bold] "
                f"(based on your data structure)\n\n"
                "[bold white][1] qa[/bold white] - Question-Answer Pairs\n"
                "    [dim]Input: Questions about your content\n"
                "    Output: Accurate answers grounded in source material\n"
                "    Format: {question, answer, context}[/dim]\n\n"
                "[bold white][2] conversation[/bold white] - Multi-Turn Dialogues\n"
                "    [dim]Input: Conversation starters or scenarios\n"
                "    Output: Natural back-and-forth exchanges\n"
                "    Format: {messages: [{role, content}, ...]}[/dim]\n\n"
                "[bold white][3] augmentation[/bold white] - Variations of Existing Data\n"
                "    [dim]Input: Your existing examples\n"
                "    Output: Paraphrased versions preserving meaning\n"
                "    Format: Same as input[/dim]\n\n"
                "[bold white][4] instruction[/bold white] - Instruction-Following Data\n"
                "    [dim]Input: Task instructions or commands\n"
                "    Output: Correct task completions\n"
                "    Format: {instruction, input, output}[/dim]",
                title="What type of training data do you want to generate?",
                border_style="blue",
            )
        )
    else:
        cli_utils.CONSOLE.print(
            f"[bold]Data type to generate[/bold] [dim](recommended: {suggested_goal})[/dim]\n"
            "  [white][1][/white] qa           - Question-answer pairs\n"
            "  [white][2][/white] conversation - Multi-turn dialogues\n"
            "  [white][3][/white] augmentation - Variations of existing data\n"
            "  [white][4][/white] instruction  - Instruction-following examples"
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
                "[bold]How would you like to define your task?[/bold]\n\n"
                "[cyan][1][/cyan] [bold]Quick[/bold] - AI auto-generates based on your data\n"
                "    [dim]System prompt, inputs, and outputs are inferred automatically.\n"
                "    Good for: Fast iteration, exploring what's possible.[/dim]\n\n"
                "[cyan][2][/cyan] [bold]Interactive[/bold] - Define each component step by step\n"
                "    [dim]You'll specify: system prompt → input distribution → output quality.\n"
                "    Good for: Production use, high-quality customized data.[/dim]",
                title="Task Definition Mode",
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
                "\n[bold magenta]Interactive Task Definition[/bold magenta]\n"
                "[dim]We'll define your task in 3 steps:[/dim]\n"
                "  [cyan]1.[/cyan] System Prompt - How should your model behave?\n"
                "  [cyan]2.[/cyan] Input Distribution - What questions/requests will users make?\n"
                "  [cyan]3.[/cyan] Output Quality - What does a good answer look like?\n"
            )

            # Step 1: Build system prompt interactively
            cli_utils.CONSOLE.print(
                "\n[bold cyan]─── Step 1: System Prompt ───[/bold cyan]\n"
                "[dim]Define your model's persona, capabilities, and constraints.[/dim]"
            )
            system_prompt = _iterative_system_prompt_builder(
                schema, domain, llm_analyzer, file_roles
            )

            # Step 2: Build question template interactively
            cli_utils.CONSOLE.print(
                "\n[bold cyan]─── Step 2: Input Distribution ───[/bold cyan]\n"
                "[dim]Define the types of questions or requests users will make.[/dim]"
            )
            question_template = _iterative_question_template_builder(
                schema, domain, system_prompt, llm_analyzer
            )

            # Step 3: Build answer template interactively
            cli_utils.CONSOLE.print(
                "\n[bold cyan]─── Step 3: Output Quality ───[/bold cyan]\n"
                "[dim]Define what a high-quality answer looks like.[/dim]"
            )
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

    cli_utils.CONSOLE.print("\n[bold cyan]Step 3/5: Defining output quality criteria...[/bold cyan]")

    if verbose:
        cli_utils.CONSOLE.print(
            "[dim]A judge evaluates: \"What does a good answer look like?\"\n"
            "It scores your data so you can filter low-quality examples.[/dim]\n"
        )

    if verbose:
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
                title="What quality criteria should we evaluate?",
                border_style="blue",
            )
        )
    else:
        cli_utils.CONSOLE.print(
            "[bold]Quality criteria to evaluate:[/bold]\n"
            "  [white][1][/white] generic      - Overall quality, coherence, helpfulness\n"
            "  [white][2][/white] compliance   - Follows rules and guidelines\n"
            "  [white][3][/white] relevance    - Addresses the question asked\n"
            "  [white][4][/white] safety       - Free from harmful content\n"
            "  [white][5][/white] groundedness - Claims supported by context"
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
