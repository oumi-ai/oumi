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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

import oumi.cli.cli_utils as cli_utils

from .cache import (
    clear_cache,
    compute_file_hash,
    display_cache_summary,
    get_cache_path,
    list_cache_files,
    load_wizard_cache,
    open_cache_for_editing,
    prompt_cache_action,
    save_wizard_cache,
)
from .dataclasses import (
    GOAL_CHOICES,
    INPUT_FORMATS,
    SUPPORTED_EXTENSIONS,
    WizardState,
)
from .helpers import analyze_file_purposes, detect_files_in_directory
from .wizard_steps import (
    wizard_step_confirm_detection,
    wizard_step_detect,
    wizard_step_generate,
    wizard_step_outputs,
    wizard_step_task,
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
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            "-e",
            help="LLM inference engine (ANTHROPIC, OPENAI, DEEPSEEK, TOGETHER).",
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
    num_samples: Annotated[
        int,
        typer.Option(
            "--num-samples",
            "-n",
            help="Number of samples to generate in synthesis config.",
        ),
    ] = 10,
    auto_accept: Annotated[
        bool,
        typer.Option(
            "--auto-accept/--no-auto-accept",
            "-y",
            help="Auto-accept suggestions (press Enter to continue).",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging (show LLM inputs/outputs).",
        ),
    ] = False,
):
    """Interactive wizard to guide you through Oumi setup.

    This wizard helps you build training data through a streamlined detection-first flow:
    1. Detection - Auto-analyze files for task, template, examples, evals
    2. Confirmation - Review and confirm detected elements
    3. Task Definition - Define task and system prompt (template auto-inferred if detected)
    4. Quality Criteria - Define evaluation criteria (merges extracted + generated)
    5. Generate - Create configs with appropriate mode (augmentation/teacher_labeling/synthesis)

    LLM analysis is required for intelligent suggestions.

    Examples:
        # Basic usage (uses ANTHROPIC by default)
        oumi onboard wizard --data ./my_data.csv

        # Use OpenAI
        oumi onboard wizard --data ./data/ --engine OPENAI

        # Specify model
        oumi onboard wizard --data ./data/ --model claude-sonnet-4-20250514

        # Enable verbose logging to see LLM inputs/outputs
        oumi onboard wizard --data ./my_data.csv --verbose
    """
    from oumi.onboarding import DataAnalyzer
    from oumi.onboarding.llm_analyzer import LLMAnalyzer

    state = WizardState()

    cli_utils.CONSOLE.print(
        "[bold green]Oumi Onboarding Wizard[/bold green]\n"
        "[dim]Flow: Detect -> Confirm -> Task -> Quality Criteria -> Generate[/dim]\n"
    )

    data_path = Path(data)
    if not data_path.exists():
        cli_utils.CONSOLE.print(f"[red]Error: Path not found: {data}[/red]")
        raise typer.Exit(1)

    valid_engines = ["ANTHROPIC", "OPENAI", "DEEPSEEK", "TOGETHER"]
    engine_upper = engine.upper()
    if engine_upper not in valid_engines:
        cli_utils.CONSOLE.print(
            f"[yellow]Warning: Unknown engine '{engine}'. "
            f"Valid options: {', '.join(valid_engines)}. Using ANTHROPIC.[/yellow]"
        )
        engine_upper = "ANTHROPIC"

    try:
        state.llm_analyzer = LLMAnalyzer(engine=engine_upper, model=model, verbose=verbose)
        cli_utils.CONSOLE.print(
            f"[dim]Using {engine_upper} engine"
            f"{f' with model {model}' if model else ''}[/dim]\n"
        )
    except Exception as e:
        cli_utils.CONSOLE.print(
            f"[red]Error: Could not initialize LLM analyzer: {e}[/red]\n"
            "[dim]Make sure you have the required API key set (e.g., ANTHROPIC_API_KEY).[/dim]"
        )
        raise typer.Exit(1)

    analyzer = DataAnalyzer()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = get_cache_path(output_path, model, engine_upper)
    cached_state = load_wizard_cache(output_path, model, engine_upper)
    use_cached_analysis = False

    if cached_state:
        cli_utils.CONSOLE.print("\n[bold cyan]Found existing wizard cache![/bold cyan]")
        display_cache_summary(cached_state, cache_path)

        action = prompt_cache_action(cache_path)

        if action == "edit":
            open_cache_for_editing(cache_path)
            cached_state = load_wizard_cache(output_path, model, engine_upper)
            if not cached_state:
                cli_utils.CONSOLE.print("[yellow]Could not reload cache. Starting fresh.[/yellow]")

        if action == "restart":
            cli_utils.CONSOLE.print("[dim]Starting fresh...[/dim]")
            cache_path.unlink(missing_ok=True)
            cached_state = None

        if cached_state and action in ("resume", "edit"):
            state.task = cached_state.task
            state.inputs = cached_state.inputs
            state.outputs = cached_state.outputs
            state.detection = cached_state.detection
            state.files = cached_state.files
            state.primary_schema = cached_state.primary_schema
            state.domain_analysis = cached_state.domain_analysis
            state.completed_steps = cached_state.completed_steps
            use_cached_analysis = True
            cli_utils.CONSOLE.print(
                f"[green]Resuming from step {len(state.completed_steps) + 1}...[/green]\n"
            )

    cli_utils.CONSOLE.print("[dim]Scanning files...[/dim]")

    if data_path.is_dir():
        files = detect_files_in_directory(data_path)
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

    for f in files:
        f["content_hash"] = compute_file_hash(f["path"])

    cached_file_analysis = {}
    if cached_state and cached_state.files:
        for cf in cached_state.files:
            if cf.get("path") and cf.get("content_hash"):
                cached_file_analysis[str(cf["path"])] = cf

    files_needing_llm_analysis = []
    cache_hits = 0
    total_files = len(files)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]({task.completed}/{task.total})[/dim]"),
        console=cli_utils.CONSOLE,
        transient=True,
    ) as progress:
        scan_task = progress.add_task("[dim]Scanning files...[/dim]", total=total_files)
        for idx, f in enumerate(files, 1):
            file_path_str = str(f["path"])
            cached_info = cached_file_analysis.get(file_path_str)
            used_cache = False

            if (
                use_cached_analysis
                and cached_info
                and cached_info.get("content_hash") == f["content_hash"]
                and cached_info.get("suggested_purpose")
            ):
                f["suggested_purpose"] = cached_info.get("suggested_purpose", "")
                f["suggested_role"] = cached_info.get("suggested_role", "")
                f["role_reason"] = cached_info.get("role_reason", "")
                used_cache = True
            else:
                files_needing_llm_analysis.append(f)

            # Check for cached schema
            if (
                use_cached_analysis
                and cached_info
                and cached_info.get("content_hash") == f["content_hash"]
                and cached_info.get("schema_cache")
            ):
                f["schema"] = cached_info.get("schema_cache")
                used_cache = True
            else:
                try:
                    schema = analyzer.analyze(f["path"])
                    f["schema"] = schema
                except Exception:
                    f["schema"] = None

            if used_cache:
                cache_hits += 1
                status = "[green](cached)[/green]"
            else:
                status = "[yellow](new)[/yellow]"

            progress.update(
                scan_task,
                advance=1,
                description=f"[dim]Analyzing {f['name']}[/dim] {status}",
            )

    if cache_hits > 0:
        cli_utils.CONSOLE.print(f"[dim]Used cache for {cache_hits}/{total_files} files[/dim]")

    if files_needing_llm_analysis:
        files_needing_llm_analysis = analyze_file_purposes(
            files_needing_llm_analysis, analyzer, state.llm_analyzer
        )
        analyzed_by_path = {str(f["path"]): f for f in files_needing_llm_analysis}
        for f in files:
            if str(f["path"]) in analyzed_by_path:
                analyzed = analyzed_by_path[str(f["path"])]
                f["suggested_purpose"] = analyzed.get("suggested_purpose", "")
                f["suggested_role"] = analyzed.get("suggested_role", "")
                f["role_reason"] = analyzed.get("role_reason", "")

    state.files = files

    for f in files:
        if f.get("schema") and f["schema"].columns:
            state.primary_schema = f["schema"]
            break
    if not state.primary_schema and files and files[0].get("schema"):
        state.primary_schema = files[0]["schema"]

    # Analyze domain FIRST (provides context to detection)
    # Domain analysis helps detection identify task types, input sources, and quality criteria
    if state.primary_schema:
        if use_cached_analysis and cached_state and cached_state.domain_analysis:
            state.domain_analysis = cached_state.domain_analysis
        else:
            try:
                with cli_utils.CONSOLE.status("[dim]Analyzing domain...[/dim]", spinner="dots"):
                    state.domain_analysis = state.llm_analyzer.analyze(state.primary_schema)
            except Exception:
                pass

    # Phase 0: Detection (silent, can now use domain context)
    if "detection" not in state.completed_steps:
        state = wizard_step_detect(state)
        save_wizard_cache(state, output_path, "detection", model, engine_upper)
    else:
        cli_utils.CONSOLE.print(f"\n[dim]Detection[/dim] [green]v[/green]")

    # Phase 1: Confirmation
    if "confirm" not in state.completed_steps:
        state = wizard_step_confirm_detection(state, auto_accept=auto_accept)
        save_wizard_cache(state, output_path, "confirm", model, engine_upper)
    else:
        cli_utils.CONSOLE.print(f"\n[dim]Detection summary[/dim] [green]v[/green]")

    # Phase 2: Task Definition (includes template if detected)
    if "task" not in state.completed_steps:
        state = wizard_step_task(state, auto_accept=auto_accept)
        save_wizard_cache(state, output_path, "task", model, engine_upper)
    else:
        desc_preview = state.task.description or "defined"
        cli_utils.CONSOLE.print(f"\n[dim]Task[/dim] [green]v[/green] {desc_preview}")

    # Template and Inputs steps removed - now handled automatically
    # Template: auto-inferred during task step
    # Inputs: determined during generate step based on detection results

    # Phase 3: Quality Criteria (formerly "outputs")
    if "outputs" not in state.completed_steps:
        state = wizard_step_outputs(state, auto_accept=auto_accept)
        save_wizard_cache(state, output_path, "outputs", model, engine_upper)
    else:
        quality_preview = ", ".join(state.outputs.criteria) or "defined"
        cli_utils.CONSOLE.print(f"\n[dim]Outputs[/dim] [green]v[/green] {quality_preview}")

    if "generate" not in state.completed_steps:
        synth_config_path, judge_config_paths = wizard_step_generate(
            state, output_path, num_samples=num_samples
        )
        save_wizard_cache(state, output_path, "generate", model, engine_upper)
    else:
        synth_config_path = str(output_path / "synth_config.yaml")
        judge_config_paths = []
        cli_utils.CONSOLE.print(f"\n[dim]Generate[/dim] [green]v[/green] configs generated")

    cli_utils.CONSOLE.print("\n[bold green]--- Complete! ---[/bold green]\n")

    commands = [f"oumi synth -c {synth_config_path}"]
    if judge_config_paths:
        judge_path = judge_config_paths[0]
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
    ] = 10,
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
    from oumi.onboarding import DataAnalyzer
    from oumi.onboarding.config_builder import (
        JudgeConfigBuilder,
        SynthConfigBuilder,
        TrainConfigBuilder,
    )

    if goal not in GOAL_CHOICES:
        cli_utils.CONSOLE.print(
            f"[red]Invalid goal: {goal}. Choose from: {GOAL_CHOICES}[/red]"
        )
        raise typer.Exit(1)

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
        sg = synth_goal or analyzer.suggest_goal(schema)
        builder = SynthConfigBuilder()
        config = builder.build(
            schema=schema,
            goal=sg,
            num_samples=num_samples,
            task_description=f"Generate {sg} data from the provided dataset",
            system_prompt="You are a helpful AI assistant.",
        )

        if output_path.suffix != ".yaml":
            output_path = output_path / "synth_config.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        config.to_yaml(str(output_path), exclude_defaults=True)
        cli_utils.CONSOLE.print(f"[green]Synth config saved to: {output_path}[/green]")

    elif goal == "judge":
        jt = judge_type or "generic"
        builder = JudgeConfigBuilder()
        config = builder.build(
            schema=schema,
            judge_name=jt,
            criteria=f"Evaluate the quality of the response for {jt} criteria",
            task_type="generation",
            task_description="Evaluate model outputs",
        )

        if output_path.suffix != ".yaml":
            output_path = output_path / "judge_config.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        config.to_yaml(str(output_path), exclude_defaults=True)
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

        config.to_yaml(str(output_path), exclude_defaults=True)
        cli_utils.CONSOLE.print(f"[green]Train config saved to: {output_path}[/green]")

    elif goal == "pipeline":
        if output_path.suffix == ".yaml":
            output_path = output_path.parent / "configs"
        output_path.mkdir(parents=True, exist_ok=True)

        sg = synth_goal or analyzer.suggest_goal(schema)
        jt = judge_type or "generic"

        synth_builder = SynthConfigBuilder()
        synth_config = synth_builder.build(
            schema=schema,
            goal=sg,
            num_samples=num_samples,
            output_path=str(output_path / "synth_output.jsonl"),
            task_description=f"Generate {sg} data from the provided dataset",
            system_prompt="You are a helpful AI assistant.",
        )

        judge_builder = JudgeConfigBuilder()
        judge_config = judge_builder.build(
            schema=schema,
            judge_name=jt,
            criteria=f"Evaluate the quality of the response for {jt} criteria",
            task_type="generation",
            task_description="Evaluate model outputs",
        )

        train_builder = TrainConfigBuilder()
        train_config = train_builder.from_data_path(
            str(output_path / "synth_output.jsonl"),
            base_model=base_model,
            use_lora=use_lora,
            output_dir=str(output_path / "model_output"),
        )

        synth_config.to_yaml(str(output_path / "synth_config.yaml"), exclude_defaults=True)
        judge_config.to_yaml(str(output_path / "judge_config.yaml"), exclude_defaults=True)
        train_config.to_yaml(str(output_path / "train_config.yaml"), exclude_defaults=True)

        cli_utils.CONSOLE.print(f"[green]Pipeline configs saved to: {output_path}/[/green]")
        cli_utils.CONSOLE.print("  - synth_config.yaml")
        cli_utils.CONSOLE.print("  - judge_config.yaml")
        cli_utils.CONSOLE.print("  - train_config.yaml")


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

    _display_schema_info(schema)

    if mappings:
        mapping_table = Table(title="Suggested Field Mappings", show_edge=False)
        mapping_table.add_column("Your Column", style="cyan")
        mapping_table.add_column("Oumi Placeholder", style="green")
        mapping_table.add_column("Confidence", style="yellow")

        for m in mappings:
            conf_str = f"{m.confidence:.0%}"
            mapping_table.add_row(m.customer_column, f"{{{m.oumi_placeholder}}}", conf_str)

        cli_utils.CONSOLE.print(mapping_table)

    suggested_goal = analyzer.suggest_goal(schema)
    cli_utils.CONSOLE.print(f"\n[bold]Suggested synthesis goal:[/bold] [green]{suggested_goal}[/green]")

    cli_utils.CONSOLE.print(
        f"\n[dim]To generate configs: oumi onboard generate --data {data} --goal synth[/dim]"
    )


def _display_schema_info(schema):
    """Display analyzed schema information."""
    table = Table(title=f"Data Analysis: {schema.source_path}", show_edge=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Format", schema.detected_format)
    table.add_row("Rows", str(schema.row_count))
    table.add_row("Columns", str(len(schema.columns)))

    if schema.conversation_columns:
        table.add_row("Conversation cols", ", ".join(schema.conversation_columns))
    if schema.text_columns:
        table.add_row("Text cols", ", ".join(schema.text_columns))
    if schema.categorical_columns:
        table.add_row("Categorical cols", ", ".join(schema.categorical_columns))

    cli_utils.CONSOLE.print(table)


def cache_clear(
    ctx: typer.Context,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory containing wizard cache files.",
        ),
    ] = "./oumi_configs",
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Only clear cache for this specific model. Requires --engine.",
        ),
    ] = None,
    engine: Annotated[
        Optional[str],
        typer.Option(
            "--engine",
            "-e",
            help="Engine used with the model. Required if --model is specified.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    list_only: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="Only list cache files, don't delete them.",
        ),
    ] = False,
):
    """Clear wizard cache files.

    By default, clears all cache files in the output directory.
    Use --model and --engine to clear only a specific model's cache.

    Examples:
        # List all cache files
        oumi onboard clear-cache --list

        # Clear all caches
        oumi onboard clear-cache

        # Clear cache for a specific model
        oumi onboard clear-cache --model claude-sonnet-4-20250514 --engine ANTHROPIC

        # Clear all caches without confirmation
        oumi onboard clear-cache --force
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        cli_utils.CONSOLE.print(f"[yellow]Directory does not exist: {output_dir}[/yellow]")
        raise typer.Exit(0)

    # Validate model/engine combination
    if model and not engine:
        cli_utils.CONSOLE.print(
            "[red]Error: --engine is required when --model is specified.[/red]"
        )
        raise typer.Exit(1)
    if engine and not model:
        cli_utils.CONSOLE.print(
            "[red]Error: --model is required when --engine is specified.[/red]"
        )
        raise typer.Exit(1)

    # List cache files
    cache_files = list_cache_files(output_path)

    if not cache_files:
        cli_utils.CONSOLE.print(f"[dim]No cache files found in {output_dir}[/dim]")
        raise typer.Exit(0)

    # Display cache files
    table = Table(title="Wizard Cache Files", show_edge=False)
    table.add_column("File", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="yellow")

    for cache_file in cache_files:
        stat = cache_file.stat()
        size_kb = stat.st_size / 1024
        from datetime import datetime
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(cache_file.name, f"{size_kb:.1f} KB", modified)

    cli_utils.CONSOLE.print(table)
    cli_utils.CONSOLE.print(f"\n[dim]Found {len(cache_files)} cache file(s) in {output_dir}[/dim]")

    if list_only:
        raise typer.Exit(0)

    # Confirm deletion
    if model and engine:
        target_desc = f"cache for model '{model}' (engine: {engine})"
    else:
        target_desc = f"all {len(cache_files)} cache file(s)"

    if not force:
        from rich.prompt import Confirm
        if not Confirm.ask(f"\n[yellow]Delete {target_desc}?[/yellow]"):
            cli_utils.CONSOLE.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Clear cache
    deleted_count = clear_cache(output_path, model, engine)

    if deleted_count > 0:
        cli_utils.CONSOLE.print(f"[green]Deleted {deleted_count} cache file(s).[/green]")
    else:
        cli_utils.CONSOLE.print("[yellow]No matching cache files found to delete.[/yellow]")
