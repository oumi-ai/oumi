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
    fallback_input_detection,
    generate_system_prompt,
    infer_task_type,
    suggest_quality_criteria,
)


def wizard_step_task(state: WizardState, auto_accept: bool = False) -> WizardState:
    """Step 1: Define task and generate system prompt.

    Requires explicit user confirmation before proceeding.

    Args:
        state: Current wizard state.
        auto_accept: If True, default to accepting suggestions.

    Returns:
        Updated state with task description, system prompt, and task type.
    """
    cli_utils.CONSOLE.print(
        "\n[bold cyan]--- Step 1/4: Define Your Task ---[/bold cyan]\n"
    )

    # Analyze files to suggest task
    with cli_utils.CONSOLE.status("[dim]Analyzing your files...[/dim]", spinner="dots"):
        analysis = analyze_task_from_files(state.files, state.llm_analyzer)
        state.task.description = analysis.get("task_description", "")

    # Display task description and ask for confirmation
    cli_utils.CONSOLE.print(
        Panel(
            f"[bold]Task:[/bold] {state.task.description}",
            title="[green]Suggested Task[/green]",
            border_style="green",
        )
    )

    # Confirm task description first
    while not Confirm.ask("Accept this task description?", default=auto_accept):
        state.task.description = Prompt.ask(
            "What should your model do?",
            default=state.task.description,
        )
        cli_utils.CONSOLE.print(
            Panel(
                f"[bold]Task:[/bold] {state.task.description}",
                title="[green]Updated Task[/green]",
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

    if state.primary_schema:
        synth_config = synth_builder.from_schema(
            state.primary_schema,
            goal="qa",
            num_samples=num_samples,
            output_path=str(output_path / "synth_output.jsonl"),
            system_prompt=state.task.system_prompt,
            task_description=state.task.description,
            task_type=state.task.task_type,
        )
    else:
        synth_config = synth_builder.from_template(
            "qa_generation",
            num_samples=num_samples,
            output_path=str(output_path / "synth_output.jsonl"),
        )

    synth_path = output_path / "synth_config.yaml"
    synth_config.to_yaml(str(synth_path))
    cli_utils.CONSOLE.print(f"[green]v Synthesis config:[/green] {synth_path}")

    # Generate judges from criteria
    judge_paths = []
    if state.outputs.criteria:
        cli_utils.CONSOLE.print("[dim]Generating judge configs...[/dim]")
        judge_builder = JudgeConfigBuilder()

        for criterion in state.outputs.criteria[:3]:
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
