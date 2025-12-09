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

"""Simplified wizard step functions for the onboard wizard.

Each step has ONE user prompt requiring explicit confirmation.
"""

from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

import oumi.cli.cli_utils as cli_utils

from .dataclasses import INPUT_FORMATS, TASK_TYPES, WizardState
from .helpers import (
    analyze_task_from_files,
    detect_all_elements,
    detect_input_source,
    extract_use_case_from_documents,
    fallback_input_detection,
    generate_system_prompt,
    infer_task_type,
    suggest_quality_criteria,
)


# =============================================================================
# Phase 0 & 1: Detection and Confirmation Steps
# =============================================================================


def wizard_step_detect(state: WizardState) -> WizardState:
    """Phase 0: Detect what elements the customer has provided.

    This is a silent phase - no user interaction, just analysis.

    Args:
        state: Current wizard state with files and schema.

    Returns:
        Updated state with detection results populated.
    """
    with cli_utils.CONSOLE.status(
        "[dim]Analyzing your files to detect provided elements...[/dim]",
        spinner="dots",
    ):
        state.detection = detect_all_elements(
            files=state.files,
            primary_schema=state.primary_schema,
            llm_analyzer=state.llm_analyzer,
            domain_analysis=state.domain_analysis,
        )

    return state


def wizard_step_confirm_detection(
    state: WizardState, auto_accept: bool = False
) -> WizardState:
    """Phase 1: Show detection results and get user confirmation.

    Displays a summary of what was detected and allows user to confirm
    or correct the detection results.

    Args:
        state: Current wizard state with detection results.
        auto_accept: If True, automatically accept detection results.

    Returns:
        Updated state with confirmed detection results.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Detection Summary ---[/bold cyan]\n"
    )

    detection = state.detection

    # Build detection summary table
    table = Table(title="Detected Elements", show_header=True, header_style="bold")
    table.add_column("Element", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Task definition
    if detection.has_task_definition:
        task_preview = (detection.task_definition or "")[:50]
        if len(detection.task_definition or "") > 50:
            task_preview += "..."
        table.add_row("Task Definition", "[green]Found[/green]", task_preview)
    else:
        table.add_row("Task Definition", "[yellow]Not found[/yellow]", "Will generate suggestions")

    # System prompt
    if detection.system_prompt:
        prompt_preview = detection.system_prompt[:50]
        if len(detection.system_prompt) > 50:
            prompt_preview += "..."
        table.add_row("System Prompt", "[green]Found[/green]", prompt_preview)
    else:
        table.add_row("System Prompt", "[yellow]Not found[/yellow]", "Will generate")

    # User prompt template
    if detection.has_user_prompt_template:
        template_preview = (detection.user_prompt_template or "")[:50]
        vars_str = f"Variables: {', '.join(detection.template_variables[:3])}"
        table.add_row("Prompt Template", "[green]Found[/green]", vars_str)
    else:
        table.add_row("Prompt Template", "[dim]None[/dim]", "")

    # Labeled examples
    if detection.has_labeled_examples:
        count = len(detection.labeled_examples)
        table.add_row(
            "Labeled Examples",
            "[green]Found[/green]",
            f"{count} input-output pairs",
        )
    else:
        table.add_row("Labeled Examples", "[dim]None[/dim]", "")

    # Unlabeled prompts
    if detection.has_unlabeled_prompts:
        count = len(detection.unlabeled_prompts)
        table.add_row(
            "Unlabeled Prompts",
            "[green]Found[/green]",
            f"{count} prompts (will label via teacher)",
        )
    else:
        table.add_row("Unlabeled Prompts", "[dim]None[/dim]", "")

    # Evaluation criteria
    if detection.has_eval_criteria:
        criteria_str = ", ".join(detection.eval_criteria[:3])
        table.add_row("Eval Criteria", "[green]Found[/green]", criteria_str)
    else:
        table.add_row("Eval Criteria", "[yellow]Not found[/yellow]", "Will generate suggestions")

    # Seed data
    if detection.has_seed_data:
        cols_str = ", ".join(detection.seed_columns[:3])
        table.add_row("Seed Data", "[green]Found[/green]", f"Columns: {cols_str}")
    else:
        table.add_row("Seed Data", "[dim]None[/dim]", "")

    cli_utils.CONSOLE.print(table)
    cli_utils.CONSOLE.print()

    # Determine generation mode
    if detection.has_labeled_examples:
        mode = "[bold green]Augmentation Mode[/bold green]"
        mode_desc = "Using your labeled examples as seeds to generate diverse training data"
    elif detection.has_unlabeled_prompts:
        mode = "[bold blue]Teacher Labeling Mode[/bold blue]"
        mode_desc = "Generating outputs for your prompts using a teacher model"
    else:
        mode = "[bold cyan]Synthesis Mode[/bold cyan]"
        mode_desc = "Generating both inputs and outputs from scratch"

    cli_utils.CONSOLE.print(f"[bold]Generation Mode:[/bold] {mode}")
    cli_utils.CONSOLE.print(f"[dim]{mode_desc}[/dim]\n")

    # Ask for confirmation
    if not auto_accept:
        if not Confirm.ask("Proceed with these detection results?", default=True):
            cli_utils.CONSOLE.print(
                "[yellow]You can manually specify elements in the following steps.[/yellow]\n"
            )
            # Clear detections so wizard falls back to generation
            state.detection.has_task_definition = False
            state.detection.has_labeled_examples = False
            state.detection.has_unlabeled_prompts = False
            state.detection.has_eval_criteria = False

    cli_utils.CONSOLE.print("[green]v Detection confirmed[/green]")
    return state


def wizard_step_template(
    state: WizardState, auto_accept: bool = False
) -> WizardState:
    """Step for clarifying user prompt template mappings.

    Only runs if a user prompt template was detected.
    Shows the template and asks user to confirm variable-to-column mappings.

    Args:
        state: Current wizard state with detected template.
        auto_accept: If True, automatically accept suggested mappings.

    Returns:
        Updated state with confirmed template mappings.
    """
    if not state.detection.has_user_prompt_template:
        return state

    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Template Configuration ---[/bold cyan]\n"
    )

    # Display the template
    template = state.detection.user_prompt_template or ""
    cli_utils.CONSOLE.print("[bold]Detected User Prompt Template:[/bold]")
    cli_utils.CONSOLE.print(
        Panel(template, border_style="cyan")
    )

    # Show detected variables
    variables = state.detection.template_variables
    if not variables:
        cli_utils.CONSOLE.print("[yellow]No variables detected in template.[/yellow]")
        return state

    cli_utils.CONSOLE.print(f"\n[bold]Template Variables:[/bold] {', '.join(variables)}")

    # Get available columns
    available_columns = []
    if state.primary_schema and state.primary_schema.columns:
        available_columns = [col.name for col in state.primary_schema.columns]

    # Show suggested mappings and get confirmation
    cli_utils.CONSOLE.print("\n[bold]Variable Mappings:[/bold]")

    mapping = state.detection.template_mapping.copy()

    for var in variables:
        suggested = mapping.get(var, "")

        # If no suggestion, try to find a matching column
        if not suggested and available_columns:
            # Simple matching: look for columns with similar names
            for col in available_columns:
                if var.lower() in col.lower() or col.lower() in var.lower():
                    suggested = col
                    break

        if auto_accept and suggested:
            mapping[var] = suggested
            cli_utils.CONSOLE.print(f"  {{{var}}} -> [green]{suggested}[/green]")
        else:
            # Show available columns
            if available_columns:
                cols_str = ", ".join(available_columns[:10])
                cli_utils.CONSOLE.print(f"\n  Available columns: {cols_str}")

            col_choice = Prompt.ask(
                f"  Map {{{var}}} to column",
                default=suggested if suggested else "",
            )
            mapping[var] = col_choice
            cli_utils.CONSOLE.print(f"  {{{var}}} -> [green]{col_choice}[/green]")

    state.detection.template_mapping = mapping
    cli_utils.CONSOLE.print("\n[green]v Template mappings configured[/green]")
    return state


# =============================================================================
# Phase 2: Conditional Wizard Steps (existing steps, now with conditional logic)
# =============================================================================


def wizard_step_task(state: WizardState, auto_accept: bool = False) -> WizardState:
    """Step 1: Define task and generate system prompt.

    Uses detection results if available, otherwise falls back to extraction
    or generation of task suggestions.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with task description, system prompt, and task type.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 1/4: Define Your Task ---[/bold cyan]\n"
    )

    # Check if detection already found task definition
    if state.detection.has_task_definition and state.detection.task_definition:
        cli_utils.CONSOLE.print(
            "[bold green]Using detected task definition from your documents.[/bold green]\n"
        )

        # Build summary of detected elements
        detected_parts = []
        detected_parts.append(
            f"[bold]Task:[/bold] {state.detection.task_definition}"
        )
        if state.detection.system_prompt:
            prompt_preview = state.detection.system_prompt[:300]
            if len(state.detection.system_prompt) > 300:
                prompt_preview += "..."
            detected_parts.append(f"[bold]System Prompt:[/bold]\n{prompt_preview}")

        cli_utils.CONSOLE.print(
            Panel(
                "\n\n".join(detected_parts),
                title="[green]Detected Task[/green]",
                border_style="green",
            )
        )

        # Ask for confirmation
        if Confirm.ask("Use this detected task?", default=auto_accept or True):
            state.task.description = state.detection.task_definition
            state.task.system_prompt = (
                state.detection.system_prompt
                or generate_system_prompt(state, state.llm_analyzer)
            )

            # Infer task type
            with cli_utils.CONSOLE.status(
                "[dim]Detecting task type...[/dim]", spinner="dots"
            ):
                task_type, _ = infer_task_type(
                    state.task.description,
                    state.task.system_prompt,
                    state.llm_analyzer,
                )
                state.task.task_type = task_type

            cli_utils.CONSOLE.print("[green]v Task defined from detection[/green]")
            return state

    # Fall back to original extraction logic if detection didn't find task
    extracted_use_case = None
    if not state.detection.has_task_definition:
        with cli_utils.CONSOLE.status(
            "[dim]Analyzing your files for use case specification...[/dim]", spinner="dots"
        ):
            extracted_use_case = extract_use_case_from_documents(
                state.files, state.llm_analyzer
            )

    if extracted_use_case and extracted_use_case.has_explicit_use_case:
        # Customer provided explicit use case - show what we extracted
        cli_utils.CONSOLE.print(
            "[bold green]Found explicit use case in your documents![/bold green]\n"
        )

        # Build summary of extracted elements
        extracted_parts = []
        if extracted_use_case.task_description:
            extracted_parts.append(
                f"[bold]Task:[/bold] {extracted_use_case.task_description}"
            )
        if extracted_use_case.system_prompt:
            # Truncate for display
            prompt_preview = extracted_use_case.system_prompt[:300]
            if len(extracted_use_case.system_prompt) > 300:
                prompt_preview += "..."
            extracted_parts.append(f"[bold]System Prompt:[/bold]\n{prompt_preview}")
        if extracted_use_case.output_schema:
            extracted_parts.append(
                f"[bold]Output Schema:[/bold] {extracted_use_case.output_schema[:200]}"
            )
        if extracted_use_case.input_fields:
            extracted_parts.append(
                f"[bold]Input Fields:[/bold] {', '.join(extracted_use_case.input_fields)}"
            )
        if extracted_use_case.output_fields:
            extracted_parts.append(
                f"[bold]Output Fields:[/bold] {', '.join(extracted_use_case.output_fields)}"
            )

        cli_utils.CONSOLE.print(
            Panel(
                "\n\n".join(extracted_parts),
                title="[green]Extracted Use Case[/green]",
                border_style="green",
            )
        )

        # Ask if user wants to use extracted use case
        if Confirm.ask("Use this extracted use case?", default=auto_accept or True):
            # Use extracted values
            state.task.description = (
                extracted_use_case.task_description
                or "Process input according to the provided specification"
            )
            state.task.system_prompt = (
                extracted_use_case.system_prompt
                or generate_system_prompt(state, state.llm_analyzer)
            )

            # Store additional extracted info for later use
            state.extracted_use_case = extracted_use_case

            # Infer task type
            with cli_utils.CONSOLE.status(
                "[dim]Detecting task type...[/dim]", spinner="dots"
            ):
                task_type, _ = infer_task_type(
                    state.task.description,
                    state.task.system_prompt,
                    state.llm_analyzer,
                )
                state.task.task_type = task_type

            cli_utils.CONSOLE.print("[green]v Task defined from your documentation[/green]")
            return state

    # Fall back to generating task suggestions
    with cli_utils.CONSOLE.status("[dim]Generating task suggestions...[/dim]", spinner="dots"):
        analysis = analyze_task_from_files(
            state.files, state.llm_analyzer, state.domain_analysis
        )
        task_options = analysis.get("task_options", [])

    # Display task options
    cli_utils.CONSOLE.print("[bold]Select a task for your model:[/bold]\n")
    for i, task in enumerate(task_options, 1):
        cli_utils.CONSOLE.print(f"  [cyan][{i}][/cyan] {task}")
    cli_utils.CONSOLE.print(f"  [cyan][{len(task_options) + 1}][/cyan] Enter custom task\n")

    # Get user selection
    valid_choices = [str(i) for i in range(1, len(task_options) + 2)]
    choice = Prompt.ask(
        "Your choice",
        choices=valid_choices,
        default="1" if auto_accept else None,
    )

    choice_idx = int(choice) - 1
    if choice_idx < len(task_options):
        state.task.description = task_options[choice_idx]
    else:
        state.task.description = Prompt.ask("What should your model do?")

    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Task:[/bold] {state.task.description}",
            title="[green]Selected Task[/green]",
            border_style="green",
        )
    )

    cli_utils.CONSOLE.print("[green]v Task description confirmed[/green]\n")

    # Now generate system prompt based on confirmed task
    with cli_utils.CONSOLE.status("[dim]Generating system prompt...[/dim]", spinner="dots"):
        state.task.system_prompt = generate_system_prompt(state, state.llm_analyzer)

    # Infer task type
    with cli_utils.CONSOLE.status("[dim]Detecting task type...[/dim]", spinner="dots"):
        task_type, _ = infer_task_type(
            state.task.description, state.task.system_prompt, state.llm_analyzer
        )
        state.task.task_type = task_type

    # Display system prompt and task type
    type_info = TASK_TYPES.get(state.task.task_type, {})
    summary = (
        f"[bold]Type:[/bold] {type_info.get('name', state.task.task_type)}\n\n"
        f"[bold]System Prompt:[/bold]\n{state.task.system_prompt}"
    )
    cli_utils.CONSOLE.print(
        Panel(summary, title="[green]Generated System Prompt[/green]", border_style="green")
    )

    # Confirm system prompt
    while not Confirm.ask("Accept this system prompt?", default=auto_accept):
        state.task.system_prompt = Prompt.ask(
            "Enter your system prompt",
            default=state.task.system_prompt,
        )
        # Re-infer task type based on updated prompt
        with cli_utils.CONSOLE.status("[dim]Updating task type...[/dim]", spinner="dots"):
            task_type, _ = infer_task_type(
                state.task.description, state.task.system_prompt, state.llm_analyzer
            )
            state.task.task_type = task_type

        type_info = TASK_TYPES.get(state.task.task_type, {})
        summary = (
            f"[bold]Type:[/bold] {type_info.get('name', state.task.task_type)}\n\n"
            f"[bold]System Prompt:[/bold]\n{state.task.system_prompt}"
        )
        cli_utils.CONSOLE.print(
            Panel(summary, title="[green]Updated System Prompt[/green]", border_style="green")
        )

    cli_utils.CONSOLE.print("[green]v Task defined[/green]")
    return state


def wizard_step_inputs(state: WizardState, auto_accept: bool = False) -> WizardState:
    """Step 2: Define input distribution.

    Requires explicit user confirmation before proceeding.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with input spec.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 2/4: Input Distribution ---[/bold cyan]\n"
    )

    # Detect input source
    if state.primary_schema:
        with cli_utils.CONSOLE.status("[dim]Analyzing data for best input columns...[/dim]", spinner="dots"):
            detection = detect_input_source(state, state.llm_analyzer)
    else:
        detection = fallback_input_detection(state)

    state.inputs.source_column = detection["source_column"]
    state.inputs.format = detection["format"]
    state.inputs.samples = detection["samples"]

    # Display summary
    format_name = INPUT_FORMATS.get(state.inputs.format, state.inputs.format)
    summary_parts = [f"[bold]Format:[/bold] {format_name}"]
    if state.inputs.source_column:
        summary_parts.append(f"[bold]Source column:[/bold] {state.inputs.source_column}")
    if state.inputs.samples:
        summary_parts.append("\n[bold]Sample inputs:[/bold]")
        for i, sample in enumerate(state.inputs.samples[:3], 1):
            truncated = sample[:100] + "..." if len(sample) > 100 else sample
            summary_parts.append(f"  {i}. {truncated}")

    cli_utils.CONSOLE.print(
        Panel("\n".join(summary_parts), title="[green]Input Configuration[/green]", border_style="green")
    )

    # Require explicit confirmation
    while not Confirm.ask("Accept this input configuration?", default=auto_accept):
        # Show available columns if schema exists
        if state.primary_schema and state.primary_schema.columns:
            cli_utils.CONSOLE.print("\n[bold]Available columns:[/bold]")
            for col in state.primary_schema.columns:
                cli_utils.CONSOLE.print(f"  - {col.name} ({col.dtype})")

        state.inputs.source_column = Prompt.ask(
            "\nInput column name",
            default=state.inputs.source_column,
        )

        # Update samples for new column
        if state.inputs.source_column and state.primary_schema and state.primary_schema.sample_rows:
            state.inputs.samples = [
                str(row.get(state.inputs.source_column, ""))[:200]
                for row in state.primary_schema.sample_rows[:3]
                if row.get(state.inputs.source_column)
            ]

        # Show updated summary
        format_name = INPUT_FORMATS.get(state.inputs.format, state.inputs.format)
        summary_parts = [f"[bold]Format:[/bold] {format_name}"]
        if state.inputs.source_column:
            summary_parts.append(f"[bold]Source column:[/bold] {state.inputs.source_column}")
        if state.inputs.samples:
            summary_parts.append("\n[bold]Sample inputs:[/bold]")
            for i, sample in enumerate(state.inputs.samples[:3], 1):
                truncated = sample[:100] + "..." if len(sample) > 100 else sample
                summary_parts.append(f"  {i}. {truncated}")

        cli_utils.CONSOLE.print(
            Panel("\n".join(summary_parts), title="[green]Updated Input Configuration[/green]", border_style="green")
        )

    cli_utils.CONSOLE.print("[green]v Inputs configured[/green]")
    return state


def wizard_step_outputs(state: WizardState, auto_accept: bool = False) -> WizardState:
    """Step 3: Define output quality criteria.

    Merges extracted criteria from documents with generated suggestions,
    allowing user to select from both sources.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with output criteria.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 3/4: Output Quality ---[/bold cyan]\n"
    )

    extracted_criteria = []
    generated_criteria = []

    # Check for extracted criteria from detection
    if state.detection.has_eval_criteria and state.detection.eval_criteria:
        extracted_criteria = state.detection.eval_criteria[:5]
        cli_utils.CONSOLE.print(
            f"[green]Found {len(extracted_criteria)} criteria in your documents[/green]\n"
        )

    # Generate additional suggestions
    with cli_utils.CONSOLE.status("[dim]Generating additional quality criteria...[/dim]", spinner="dots"):
        generated_criteria = suggest_quality_criteria(state, state.llm_analyzer)

    # Merge criteria, removing duplicates (case-insensitive)
    merged_criteria = []
    criteria_sources = {}
    seen_lower = set()

    # Add extracted criteria first (they take priority)
    for c in extracted_criteria:
        c_lower = c.lower().strip()
        if c_lower not in seen_lower:
            merged_criteria.append(c)
            criteria_sources[c] = "extracted"
            seen_lower.add(c_lower)

    # Add generated criteria that aren't duplicates
    for c in generated_criteria:
        c_lower = c.lower().strip()
        if c_lower not in seen_lower:
            merged_criteria.append(c)
            criteria_sources[c] = "generated"
            seen_lower.add(c_lower)

    state.outputs.criteria = merged_criteria[:7]  # Limit to 7 criteria
    state.outputs.criteria_sources = criteria_sources

    # Display summary with sources
    cli_utils.CONSOLE.print("[bold]Quality criteria:[/bold]")
    for i, c in enumerate(state.outputs.criteria, 1):
        source = criteria_sources.get(c, "")
        if source == "extracted":
            cli_utils.CONSOLE.print(f"  {i}. {c} [green](from docs)[/green]")
        else:
            cli_utils.CONSOLE.print(f"  {i}. {c} [dim](generated)[/dim]")

    # Require explicit confirmation
    while not Confirm.ask("\nAccept these criteria?", default=auto_accept):
        criteria_input = Prompt.ask(
            "What makes a good response? (comma-separated)",
            default=", ".join(state.outputs.criteria[:3]),
        )
        state.outputs.criteria = [c.strip() for c in criteria_input.split(",") if c.strip()]
        # Mark user-edited criteria as "user"
        state.outputs.criteria_sources = {c: "user" for c in state.outputs.criteria}

        # Show updated criteria
        cli_utils.CONSOLE.print("\n[bold]Updated quality criteria:[/bold]")
        for i, c in enumerate(state.outputs.criteria, 1):
            cli_utils.CONSOLE.print(f"  {i}. {c}")

    cli_utils.CONSOLE.print("[green]v Quality criteria defined[/green]")
    return state


def wizard_step_generate(
    state: WizardState, output_path: Path, num_samples: int = 10
) -> tuple[str, list[str]]:
    """Step 4: Generate synthesis config and judges.

    No user prompts - fully automatic.

    Args:
        state: Current wizard state.
        output_path: Directory to save configs.
        num_samples: Number of samples to generate.

    Returns:
        Tuple of (synthesis_config_path, list of judge_config_paths).
    """
    from oumi.onboarding import JudgeConfigBuilder, SynthConfigBuilder

    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 4/4: Generate Configs ---[/bold cyan]\n"
    )

    # Generate synthesis config
    cli_utils.CONSOLE.print("[dim]Generating synthesis config...[/dim]")
    synth_builder = SynthConfigBuilder()

    # Build synthesis config with task-specific prompts
    synth_config = synth_builder.build(
        schema=state.primary_schema,
        goal="qa",
        task_type=state.task.task_type if state.task else "generation",
        task_description=state.task.description if state.task else "",
        system_prompt=state.task.system_prompt if state.task else "",
        domain=state.domain_analysis,
        num_samples=num_samples,
        output_path=str(output_path / "synth_output.jsonl"),
    )

    synth_path = output_path / "synth_config.yaml"
    synth_config.to_yaml(str(synth_path), exclude_defaults=True)
    cli_utils.CONSOLE.print(f"[green]v Synthesis config:[/green] {synth_path}")

    # Generate judges from criteria
    judge_paths = []
    if state.outputs.criteria:
        cli_utils.CONSOLE.print("[dim]Generating judge configs...[/dim]")
        judge_builder = JudgeConfigBuilder()

        for criterion in state.outputs.criteria[:3]:
            judge_name = criterion.lower().replace(" ", "_")[:20]
            judge_config = judge_builder.build(
                schema=state.primary_schema,
                judge_name=judge_name,
                criteria=f"Evaluate whether the response is {criterion}",
                task_type=state.task.task_type if state.task else "generation",
                task_description=state.task.description if state.task else "",
                domain=state.domain_analysis,
            )
            judge_path = output_path / f"judge_{judge_name}.yaml"
            judge_config.to_yaml(str(judge_path), exclude_defaults=True)
            judge_paths.append(str(judge_path))
            cli_utils.CONSOLE.print(f"[green]v Judge config:[/green] {judge_path}")

    # Summary
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Task:[/bold] {state.task.description[:100]}...\n"
            f"[bold]Input format:[/bold] {INPUT_FORMATS.get(state.inputs.format, state.inputs.format)}\n"
            f"[bold]Quality criteria:[/bold] {', '.join(state.outputs.criteria)}\n\n"
            f"[bold]Generated:[/bold]\n"
            f"  - Synthesis config: {synth_path}\n"
            f"  - Judge configs: {len(judge_paths)}",
            title="[green]Setup Complete[/green]",
            border_style="green",
        )
    )

    return str(synth_path), judge_paths
