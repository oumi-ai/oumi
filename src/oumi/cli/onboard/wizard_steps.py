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

import re
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

import oumi.cli.cli_utils as cli_utils

from .dataclasses import INPUT_FORMATS, TASK_TYPES, WizardState
from .helpers import (
    analyze_task_from_files,
    clarify_template_variables,
    detect_all_elements,
    detect_input_source,
    extract_use_case_from_documents,
    fallback_input_detection,
    generate_system_prompt,
    infer_task_type,
    merge_criteria,
    suggest_quality_criteria,
)


# =============================================================================
# Shared flow display helper
# =============================================================================


def _print_flow(current_step: str, state: WizardState, template_applicable: bool = True):
    """Render a richer flow diagram with conditional branches."""

    def fmt(label: str, key: str, applicable: bool = True) -> str:
        if not applicable:
            return f"[dim]{label} (skip)[/dim]"
        if key == current_step:
            return f"[bold cyan]{label}[/bold cyan]"
        if key in (state.completed_steps or []):
            return f"[green]{label}[/green]"
        return label

    has_template = state.detection.has_user_prompt_template if state.detection else False
    template_label = fmt("Template", "template", applicable=has_template and template_applicable)

    # Determine generation mode preview
    if state.detection and state.detection.has_labeled_examples:
        mode = "augmentation (labeled examples)"
    elif state.detection and state.detection.has_unlabeled_prompts:
        mode = "teacher_labeling (unlabeled prompts)"
    else:
        mode = "synthesis (from scratch)"

    detection_flag = fmt("Detection", "detection")
    confirm_flag = fmt("Confirm", "confirm")
    task_flag = fmt("Task", "task")
    criteria_flag = fmt("Criteria", "outputs")  # Display as "Criteria", cache key is "outputs"
    generate_flag = fmt("Generate", "generate")

    # Build streamlined 5-step flow diagram
    line1 = f"{detection_flag} ──> {confirm_flag} ──> {task_flag}"
    # Show template note if detected (no longer a separate step)
    if has_template:
        line2 = f"{' ' * 26}(template: auto-inferred)"
    else:
        line2 = ""
    line3 = f"{criteria_flag} ──> {generate_flag}"
    line4 = f"Mode: {mode}"

    # Detection summary hints
    det = state.detection or None
    det_bits = []
    if det:
        det_bits.append(f"task={bool(det.has_task_definition)}")
        det_bits.append(f"template={bool(det.has_user_prompt_template)}")
        det_bits.append(f"labeled={len(det.labeled_examples) if det.labeled_examples else 0}")
        det_bits.append(f"prompts={len(det.unlabeled_prompts) if det.unlabeled_prompts else 0}")
        det_bits.append(f"eval={bool(det.has_eval_criteria)}")
    line5 = f"Detection: {', '.join(det_bits)}" if det_bits else "Detection: pending"

    cli_utils.CONSOLE.print("\n[dim]Flow:[/dim]")
    cli_utils.CONSOLE.print(line1)
    if line2:
        cli_utils.CONSOLE.print(f"[dim]{line2}[/dim]")
    cli_utils.CONSOLE.print(line3)
    cli_utils.CONSOLE.print(line4)
    cli_utils.CONSOLE.print(line5 + "\n")


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
    _print_flow("detection", state, template_applicable=True)
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
    _print_flow("confirm", state, template_applicable=state.detection.has_user_prompt_template)
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
        table.add_row("Task Definition", "[green]Found[/green]", detection.task_definition or "")
    else:
        table.add_row("Task Definition", "[yellow]Not found[/yellow]", "Will generate suggestions")

    # System prompt
    if detection.system_prompt:
        table.add_row("System Prompt", "[green]Found[/green]", detection.system_prompt)
    else:
        table.add_row("System Prompt", "[yellow]Not found[/yellow]", "Will generate")

    # User prompt template
    if detection.has_user_prompt_template:
        vars_str = f"Variables: {', '.join(detection.template_variables)}" if detection.template_variables else ""
        table.add_row("Prompt Template", "[green]Found[/green]", vars_str or (detection.user_prompt_template or ""))
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
        criteria_str = ", ".join(detection.eval_criteria)
        table.add_row("Eval Criteria", "[green]Found[/green]", criteria_str)
    else:
        table.add_row("Eval Criteria", "[yellow]Not found[/yellow]", "Will generate suggestions")

    # Seed data
    if detection.has_seed_data:
        cols_str = ", ".join(detection.seed_columns)
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
            state.detection.has_user_prompt_template = False
            state.detection.user_prompt_template = None
            state.detection.template_variables = []
            state.detection.template_mapping = {}
            state.detection.has_seed_data = False

    cli_utils.CONSOLE.print("[green]v Detection confirmed[/green]")
    return state


# =============================================================================
# Phase 2: Task Definition Step (includes template handling)
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
    _print_flow("task", state, template_applicable=state.detection.has_user_prompt_template)
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Task Definition ---[/bold cyan]\n"
    )

    # Check if detection already found task definition
    if state.detection.has_task_definition and state.detection.task_definition:
        cli_utils.CONSOLE.print(
            "[bold green]Using detected task definition from your documents.[/bold green]\n"
        )

        # Build summary of detected elements (including template if present)
        detected_parts = []
        detected_parts.append(
            f"[bold]Task:[/bold] {state.detection.task_definition}"
        )
        if state.detection.system_prompt:
            detected_parts.append(f"[bold]System Prompt:[/bold]\n{state.detection.system_prompt}")

        # Show template inline if detected
        if state.detection.has_user_prompt_template and state.detection.user_prompt_template:
            detected_parts.append(f"[bold]User Prompt Template:[/bold]\n{state.detection.user_prompt_template}")
            if state.detection.template_variables:
                vars_str = ", ".join(state.detection.template_variables)
                detected_parts.append(f"[bold]Template Variables:[/bold] {vars_str}")

        cli_utils.CONSOLE.print(
            Panel(
                "\n\n".join(detected_parts),
                title="[green]Detected Task & Template[/green]",
                border_style="green",
            )
        )

        # Single confirmation for task + template
        if Confirm.ask("Use this detected task" + (" and template" if state.detection.has_user_prompt_template else "") + "?", default=auto_accept or True):
            state.task.description = state.detection.task_definition
            state.task.system_prompt = (
                state.detection.system_prompt
                or generate_system_prompt(state, state.llm_analyzer)
            )

            # Auto-infer template mappings if template exists
            if state.detection.has_user_prompt_template and state.detection.template_variables:
                if not state.detection.template_mapping:
                    cli_utils.CONSOLE.print("[dim]Inferring template variable mappings...[/dim]")
                    from .helpers import clarify_template_variables
                    state.detection.template_mapping = clarify_template_variables(
                        state.detection.user_prompt_template,
                        state.detection.template_variables,
                        state.primary_schema,
                        state.llm_analyzer
                    )

                # Show inferred mappings
                if state.detection.template_mapping:
                    cli_utils.CONSOLE.print("\n[bold]Template Variable Mappings:[/bold]")
                    for var, col in state.detection.template_mapping.items():
                        cli_utils.CONSOLE.print(f"  {{{var}}} -> [green]{col or 'n/a'}[/green]")
                    cli_utils.CONSOLE.print("[dim](Mappings are auto-inferred; edit cache file later if needed)[/dim]")

                if not any(state.detection.template_mapping.values()):
                    cli_utils.CONSOLE.print(
                        "\n[yellow]No column mappings found. Template placeholders remain user-provided at runtime.[/yellow]"
                    )

            # Infer task type
            with cli_utils.CONSOLE.status(
                "[dim]Detecting task type...[/dim]", spinner="dots"
            ):
                task_type, _ = infer_task_type(
                    state.task.description,
                    state.task.system_prompt,
                    state.llm_analyzer,
                    state.domain_analysis,
                )
                state.task.task_type = task_type

            cli_utils.CONSOLE.print("[green]v Task" + (" and template" if state.detection.has_user_prompt_template else "") + " defined from detection[/green]")
            return state

    # Fall back to generating task suggestions
    # Note: Detection already ran extract_use_case_from_documents(), so no need to re-run
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
            state.task.description, state.task.system_prompt, state.llm_analyzer, state.domain_analysis
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
                state.task.description, state.task.system_prompt, state.llm_analyzer, state.domain_analysis
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


# =============================================================================
# Phase 3: Quality Criteria Step
# =============================================================================


def wizard_step_outputs(state: WizardState, auto_accept: bool = False) -> WizardState:
    """Step 3: Define quality criteria for evaluation.

    Merges extracted criteria from documents with generated suggestions,
    allowing user to select from both sources.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with quality criteria.
    """
    _print_flow("outputs", state, template_applicable=state.detection.has_user_prompt_template)
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Quality Criteria ---[/bold cyan]\n"
    )

    extracted_criteria = []
    extracted_descriptions = {}
    generated_criteria = []
    generated_descriptions = {}

    # Check for extracted criteria from detection
    if state.detection.has_eval_criteria and state.detection.eval_criteria:
        extracted_criteria = state.detection.eval_criteria
        cli_utils.CONSOLE.print(
            f"[green]Found {len(extracted_criteria)} criteria in your documents[/green]\n"
        )

    # Smart generation: only generate if extracted criteria insufficient
    if len(extracted_criteria) >= 5 and state.detection.eval_confidence > 0.7:
        # Have sufficient high-quality criteria, skip generation
        cli_utils.CONSOLE.print("[dim]Using extracted criteria (sufficient quality)[/dim]\n")
        merged_criteria = extracted_criteria[:7]
        criteria_sources = {c: "extracted" for c in merged_criteria}
        criteria_descriptions = extracted_descriptions
    else:
        # Generate to supplement
        with cli_utils.CONSOLE.status("[dim]Generating additional quality criteria...[/dim]", spinner="dots"):
            generated_criteria, generated_descriptions = suggest_quality_criteria(state, state.llm_analyzer)

        merged_criteria, criteria_sources, criteria_descriptions = merge_criteria(
            extracted_criteria, generated_criteria,
            extracted_descriptions, generated_descriptions
        )

    state.outputs.criteria = merged_criteria[:7]  # Limit to 7 criteria
    state.outputs.criteria_sources = {
        c: source for c, source in criteria_sources.items() if c in state.outputs.criteria
    }
    state.outputs.criteria_descriptions = {
        c: desc for c, desc in criteria_descriptions.items() if c in state.outputs.criteria
    }

    # Display summary with sources and descriptions
    cli_utils.CONSOLE.print("[bold]Quality criteria:[/bold]\n")
    for i, c in enumerate(state.outputs.criteria, 1):
        source = criteria_sources.get(c, "")
        desc = criteria_descriptions.get(c, "")

        if source == "extracted":
            source_label = "[green](from docs)[/green]"
        else:
            source_label = "[dim](generated)[/dim]"

        cli_utils.CONSOLE.print(f"  {i}. [bold]{c}[/bold] {source_label}")
        if desc:
            # Indent and wrap description
            cli_utils.CONSOLE.print(f"     [dim]{desc}[/dim]")
        cli_utils.CONSOLE.print()  # Empty line between criteria

    # Require explicit confirmation
    while not Confirm.ask("\nAccept these criteria?", default=auto_accept):
        criteria_input = Prompt.ask(
            "What makes a good response? (comma-separated)",
            default=", ".join(state.outputs.criteria),
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

    _print_flow("generate", state, template_applicable=state.detection.has_user_prompt_template)
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Generate Configs ---[/bold cyan]\n"
    )

    detection = state.detection

    # Decide generation mode based on detection
    if detection.has_labeled_examples:
        generation_mode = "augmentation"
        mode_desc = "Augmenting labeled examples"
    elif detection.has_unlabeled_prompts:
        generation_mode = "teacher_labeling"
        mode_desc = "Generating outputs for your prompts"
    else:
        generation_mode = "synthesis"
        mode_desc = "Full synthesis from scratch"

    cli_utils.CONSOLE.print(f"[bold]Mode:[/bold] {generation_mode} ({mode_desc})")

    # Determine input configuration from detection (replaces wizard_step_inputs)
    if detection.has_labeled_examples:
        # Use labeled examples
        state.inputs.format = "single_turn"
        state.inputs.source_column = detection.input_column or "detected_examples"
        if detection.labeled_examples:
            state.inputs.samples = [
                str(ex.get("input", ""))
                for ex in detection.labeled_examples[:3]
                if ex.get("input")
            ]
    elif detection.has_unlabeled_prompts:
        # Use unlabeled prompts
        state.inputs.format = "single_turn"
        state.inputs.source_column = detection.prompt_column or "prompt"
        if detection.unlabeled_prompts:
            state.inputs.samples = [
                str(p) for p in detection.unlabeled_prompts[:3] if p
            ]
    else:
        # Use schema-based detection or fallback
        state.inputs.format = "single_turn"
        state.inputs.source_column = detection.input_column or detection.prompt_column
        if not state.inputs.source_column and state.primary_schema:
            # Fallback: use first text column
            for col in state.primary_schema.columns:
                if col.is_text:
                    state.inputs.source_column = col.name
                    break

    cli_utils.CONSOLE.print("[dim]Generating synthesis config...[/dim]")
    synth_builder = SynthConfigBuilder()

    seed_data = _extract_seed_data(state)

    synth_config = synth_builder.build(
        schema=state.primary_schema,
        goal="qa",
        task_type=state.task.task_type if state.task else "generation",
        task_description=state.task.description if state.task else "",
        system_prompt=state.task.system_prompt if state.task else "",
        domain=state.domain_analysis,
        num_samples=num_samples,
        output_path=str(output_path / "synth_output.jsonl"),
        generation_mode=generation_mode,
        labeled_examples=detection.labeled_examples
        if detection.has_labeled_examples
        else None,
        unlabeled_prompts=detection.unlabeled_prompts
        if detection.has_unlabeled_prompts
        else None,
        seed_data=seed_data or None,
        user_prompt_template=detection.user_prompt_template,
        template_mapping=detection.template_mapping,
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
            # Sanitize criterion name for use as filename
            judge_name = re.sub(r'[^a-z0-9_-]+', '_', criterion.lower())[:20].strip('_')
            if not judge_name:  # Fallback if sanitization results in empty string
                judge_name = f"criterion_{len(judge_paths)}"

            # Use full description if available, otherwise generate a simple one
            if criterion in state.outputs.criteria_descriptions:
                criteria_text = state.outputs.criteria_descriptions[criterion]
            else:
                criteria_text = f"Evaluate whether the response is {criterion}"

            judge_config = judge_builder.build(
                schema=state.primary_schema,
                judge_name=judge_name,
                criteria=criteria_text,
                task_type=state.task.task_type if state.task else "generation",
                task_description=state.task.description if state.task else "",
                domain=state.domain_analysis,
                llm_analyzer=state.llm_analyzer,
            )
            judge_path = output_path / f"judge_{judge_name}.yaml"
            judge_config.to_yaml(str(judge_path), exclude_defaults=True)
            judge_paths.append(str(judge_path))
            cli_utils.CONSOLE.print(f"[green]v Judge config:[/green] {judge_path}")

    # Summary
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Task:[/bold] {state.task.description}\n"
            f"[bold]Input format:[/bold] {INPUT_FORMATS.get(state.inputs.format, state.inputs.format)}\n"
            f"[bold]Generation mode:[/bold] {generation_mode}\n"
            f"[bold]Quality criteria:[/bold] {', '.join(state.outputs.criteria)}\n\n"
            f"[bold]Generated:[/bold]\n"
            f"  - Synthesis config: {synth_path}\n"
            f"  - Judge configs: {len(judge_paths)}",
            title="[green]Setup Complete[/green]",
            border_style="green",
        )
    )

    return str(synth_path), judge_paths


def _extract_seed_data(state: WizardState) -> dict[str, list[str]]:
    """Extract seed column samples for diversity."""
    seed_data: dict[str, list[str]] = {}

    if (
        not state.detection.seed_columns
        or not state.primary_schema
        or not getattr(state.primary_schema, "sample_rows", None)
    ):
        return seed_data

    for col_name in state.detection.seed_columns:
        values: list[str] = []
        for row in state.primary_schema.sample_rows[:20]:
            if col_name in row and row[col_name] is not None:
                val = str(row[col_name])
                if val and val not in values:
                    values.append(val[:100])
            if len(values) >= 8:
                break
        if values:
            seed_data[col_name] = values

    return seed_data
