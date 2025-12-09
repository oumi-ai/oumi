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

import oumi.cli.cli_utils as cli_utils

from .dataclasses import INPUT_FORMATS, TASK_TYPES, WizardState
from .helpers import (
    analyze_task_from_files,
    detect_input_source,
    extract_use_case_from_documents,
    fallback_input_detection,
    generate_system_prompt,
    infer_task_type,
    suggest_quality_criteria,
)


def wizard_step_task(state: WizardState, auto_accept: bool = False) -> WizardState:
    """Step 1: Define task and generate system prompt.

    First attempts to extract explicit use case from customer documents.
    Falls back to generating task suggestions if no explicit use case found.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with task description, system prompt, and task type.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 1/4: Define Your Task ---[/bold cyan]\n"
    )

    # First, try to extract explicit use case from customer documents
    extracted_use_case = None
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

    Requires explicit user confirmation before proceeding.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with output criteria.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 3/4: Output Quality ---[/bold cyan]\n"
    )

    # Suggest quality criteria
    with cli_utils.CONSOLE.status("[dim]Suggesting quality criteria...[/dim]", spinner="dots"):
        suggested_criteria = suggest_quality_criteria(state, state.llm_analyzer)

    state.outputs.criteria = suggested_criteria

    # Display summary
    cli_utils.CONSOLE.print("[bold]Quality criteria:[/bold]")
    for i, c in enumerate(state.outputs.criteria, 1):
        cli_utils.CONSOLE.print(f"  {i}. {c}")

    # Require explicit confirmation
    while not Confirm.ask("\nAccept these criteria?", default=auto_accept):
        criteria_input = Prompt.ask(
            "What makes a good response? (comma-separated)",
            default=", ".join(state.outputs.criteria[:3]),
        )
        state.outputs.criteria = [c.strip() for c in criteria_input.split(",") if c.strip()]

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
