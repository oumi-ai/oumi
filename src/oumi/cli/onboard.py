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


def _display_annotated_synth_config(config, config_path: Path, schema, goal: str):
    """Display synth config with helpful annotations."""
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


def _display_annotated_judge_config(config, config_path: Path, schema, judge_type: str):
    """Display judge config with helpful annotations."""
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
    config, config_path: Path, base_model: str, use_lora: bool
):
    """Display training config with helpful annotations."""
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
            help="Path to your data file (CSV, JSON, Excel, or Word).",
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
):
    """Interactive wizard to guide you through Oumi setup.

    This wizard analyzes your data and helps you create configurations
    for synthesis, evaluation, and training.

    Example:
        oumi onboard wizard --data ./my_data.csv
    """
    # Delayed imports
    from oumi.onboarding import DataAnalyzer, FieldMapper
    from oumi.onboarding.config_builder import (
        JudgeConfigBuilder,
        SynthConfigBuilder,
        TrainConfigBuilder,
    )

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

    # Step 1: Analyze data
    cli_utils.CONSOLE.print("\n[bold cyan]Step 1/5: Analyzing your data...[/bold cyan]")

    data_path = Path(data)
    if not data_path.exists():
        cli_utils.CONSOLE.print(f"[red]Error: File not found: {data}[/red]")
        raise typer.Exit(1)

    with cli_utils.CONSOLE.status("[green]Analyzing...[/green]", spinner="dots"):
        analyzer = DataAnalyzer()
        try:
            schema = analyzer.analyze(data_path)
        except Exception as e:
            cli_utils.CONSOLE.print(f"[red]Error analyzing data: {e}[/red]")
            raise typer.Exit(1)

    # Display schema info
    _display_schema_info(schema)

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
        commands = _wizard_synth(schema, output_path, analyzer)
    elif choice == 2:
        commands = _wizard_judge(schema, output_path)
    elif choice == 3:
        commands = _wizard_train(schema, output_path)
    elif choice == 4:
        commands = _wizard_pipeline(schema, output_path, analyzer)

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


def _wizard_synth(schema, output_path: Path, analyzer):
    """Configure synthesis."""
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
    config = builder.from_schema(
        schema,
        goal=goal,
        num_samples=num_samples,
        output_path=str(output_path / "synth_output.jsonl"),
    )

    config_path = output_path / "synth_config.yaml"
    config.to_yaml(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]Synth config saved to: {config_path}[/green]")

    # Display annotated config
    _display_annotated_synth_config(config, config_path, schema, goal)

    return [f"oumi synth -c {config_path}"]


def _wizard_judge(schema, output_path: Path):
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
    config = builder.from_schema(
        schema,
        judge_type=judge_type,
        custom_criteria=custom_criteria,
    )

    config_path = output_path / "judge_config.yaml"
    config.to_yaml(str(config_path))

    cli_utils.CONSOLE.print(f"\n[green]Judge config saved to: {config_path}[/green]")

    # Display annotated config
    _display_annotated_judge_config(config, config_path, schema, judge_type)

    return [f"oumi judge dataset -c {config_path} --input {schema.source_path}"]


def _wizard_train(schema, output_path: Path):
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
    _display_annotated_train_config(config, config_path, base_model, use_lora)

    return [f"oumi train -c {config_path}"]


def _wizard_pipeline(schema, output_path: Path, analyzer):
    """Configure full pipeline."""
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
    synth_config = synth_builder.from_schema(
        schema,
        goal=goal,
        num_samples=num_samples,
        output_path=str(output_path / "synth_output.jsonl"),
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
    cli_utils.CONSOLE.print("\n[bold]Pipeline Configuration Summary:[/bold]")

    _display_annotated_synth_config(synth_config, synth_path, schema, goal)
    cli_utils.CONSOLE.print("")
    _display_annotated_judge_config(judge_config, judge_path, schema, judge_type)
    cli_utils.CONSOLE.print("")
    _display_annotated_train_config(
        train_config, train_path, "meta-llama/Llama-3.2-1B-Instruct", use_lora
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
