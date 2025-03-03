import json
import os
import subprocess
from pathlib import Path

import typer
import yaml
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer()
console = Console(width=100)

# Demo configuration
models = [
    {
        "name": "XS model",
        "description": "SmolLM2-135M-Instruct",
        "value": "HuggingFaceTB/SmolLM2-135M-Instruct",
    },
    {
        "name": "Small model",
        "description": "Qwen-1.5B",
        "value": "Qwen/Qwen2.5-Math-1.5B",
    },
    {
        "name": "Medium model",
        "description": "Llama-3.1-8B-Instruct",
        "value": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "Large model",
        "description": "Llama-3.3-70B-Instruct",
        "value": "meta-llama/Llama-3.3-70B-Instruct",
    },
]

datasets = [
    {
        "name": "Alpaca",
        "description": "Instruction tuning",
        "value": "yahma/alpaca-cleaned",
    },
    {
        "name": "MetaMathQA-R1",
        "description": "Math reasoning",
        "value": "oumi-ai/MetaMathQA-R1",
    },
]

benchmarks = [
    {
        "name": "MMLU",
        "description": "General knowledge",
        "value": "mmlu_college_computer_science",
    },
    {
        "name": "GSM8K",
        "description": "Mathematical reasoning",
        "value": "gsm8k_valid",
    },
    {
        "name": "TruthfulQA",
        "description": "Factual accuracy",
        "value": "truthfulqa_mc2",
    },
    {
        "name": "HellaSwag",
        "description": "Common sense reasoning",
        "value": "hellaswag",
    },
]

cloud_providers = [
    {
        "name": "Local",
        "value": "local",
    },
    {
        "name": "Google Cloud Platform",
        "description": "GCP",
        "value": "gcp",
    },
    {
        "name": "AWS",
        "value": "aws",
    },
    {
        "name": "RunPod",
        "value": "runpod",
    },
    {
        "name": "Lambda Labs",
        "value": "lambda",
    },
]

hardware_options = [
    {
        "name": "CPU Only",
        "value": "cpu:32",
    },
    {
        "name": "1 x NVIDIA A100 GPUs",
        "value": "A100:1",
    },
    {
        "name": "4 x NVIDIA A100 GPUs",
        "value": "A100:4",
    },
    {
        "name": "8 x NVIDIA A100 GPUs",
        "value": "A100:8",
    },
    {
        "name": "8 x NVIDIA H100 GPUs",
        "value": "H100:8",
    },
]


training_options = [
    {
        "name": "Quick demo",
        "description": "25 steps",
        "value": "25",
    },
    {
        "name": "Extended training",
        "description": "1000 steps",
        "value": "1000",
    },
    {
        "name": "Full training",
        "description": "5000 steps",
        "value": "5000",
    },
]


def show_logo():
    """Display the Oumi platform logo in a panel."""
    logo_text = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|"""

    tagline = (
        "Everything you need to build state-of-the-art foundation models, end-to-end."
    )

    console.print(
        Panel(
            f"[center]{logo_text}\n\n[bold cyan]Oumi:[/bold cyan] {tagline}[/center]",
            style="green",
            border_style="bright_blue",
        )
    )


def section_header(title):
    """Print a section header with the given title.

    Args:
        title: The title text to display in the header.
    """
    console.print(f"\n[blue]{'━' * console.width}[/blue]")
    console.print(f"[yellow]   {title}[/yellow]")
    console.print(f"[blue]{'━' * console.width}[/blue]\n")


def display_yaml_config(config: dict, title: str = "Configuration"):
    """Display a YAML configuration in a panel with syntax highlighting.

    Args:
        config: The configuration dictionary to display
        title: The title for the panel
    """
    yaml_str = yaml.dump(config)
    console.print(
        Panel(
            Syntax(yaml_str, "yaml"),
            title=title,
            border_style="bright_blue",
            highlight=True,
        )
    )


def show_intro():
    """Display the introduction text about Oumi platform."""
    intro_text = """[bold cyan]Oumi[/bold cyan] is a fully open-source platform that streamlines the entire lifecycle of foundation models - from [yellow]data preparation[/yellow] and [yellow]training[/yellow] to [yellow]evaluation[/yellow] and [yellow]deployment[/yellow]. Whether you're developing on a laptop, launching large scale experiments on a cluster, or deploying models in production, Oumi provides the tools and workflows you need.

[bold green]With Oumi, you can:[/bold green]

[magenta]🚀[/magenta] [white]Train and fine-tune models from 10M to 405B parameters using state-of-the-art techniques (SFT, LoRA, QLoRA, DPO, and more)[/white]
[magenta]🤖[/magenta] [white]Work with both text and multimodal models (Llama, DeepSeek, Qwen, Phi, and others)[/white]
[magenta]🔄[/magenta] [white]Synthesize and curate training data with LLM judges[/white]
[magenta]⚡️[/magenta] [white]Deploy models efficiently with popular inference engines (vLLM, SGLang)[/white]
[magenta]📊[/magenta] [white]Evaluate models comprehensively across standard benchmarks[/white]
[magenta]🌎[/magenta] [white]Run anywhere - from laptops to clusters to clouds (AWS, Azure, GCP, Lambda, and more)[/white]
[magenta]🔌[/magenta] [white]Integrate with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI, Together, Parasail, ...)[/white]
"""
    # console.print(Panel(intro_text, border_style="bright_blue"))
    console.print(intro_text)


def run_command(
    command: str, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result.

    Args:
        command: The command to run
        capture_output: Whether to capture the command output

    Returns:
        The completed process object
    """
    console.print(f"$ [green]{command}[/green]")
    return subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=capture_output,
        check=True,
    )


def pause():
    """Pause the execution of the script and wait for user confirmation."""
    return Confirm.ask("\nPress Enter to continue...", default="y")


def create_config_file(config_data: dict, filename: str):
    """Create a YAML config file.

    Args:
        config_data: The configuration data
        filename: The output filename
    """
    with open(filename, "w") as f:
        yaml.dump(config_data, f)


def select_from_choices(
    prompt: str,
    choices: list[dict[str, str]],
    default: str = "1",
    show_descriptions: bool = True,
) -> tuple[str, str]:
    """Display numbered choices and get user selection.

    Args:
        prompt: The prompt to display to the user
        choices: List of choice dictionaries with name, description (optional), and value,
                or dictionary of choice descriptions to values, or list of choices
        default: Default choice number
        show_descriptions: Whether to show the full descriptions of choices

    Returns:
        A tuple of (selected description, selected value)
    """
    options = []
    for i, choice in enumerate(choices, 1):
        option = Text()
        option.append(f"{i}. ", style="cyan")
        option.append(choice["name"], style="bold")
        if show_descriptions and choice.get("description"):
            option.append(f" ({choice['description']})", style="dim")
        options.append(option)

    # Display options in a nice grid layout
    columns = Columns(options, equal=True, expand=True, padding=(0, 2))
    console.print(
        Panel(
            columns,
            title="[yellow]Available Options[/yellow]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )

    # Get user selection
    choice_idx = Prompt.ask(
        f"\n{prompt}",
        choices=[str(i) for i in range(1, len(choices) + 1)],
        default=default,
    )

    selected = choices[int(choice_idx) - 1]
    return selected["name"], selected["value"]


@app.command()
def run_demo():
    """Run the Oumi Platform end-to-end demonstration with real commands."""
    # Create demo directory
    demo_dir = Path("oumi_demo")
    demo_dir.mkdir(exist_ok=True)
    os.chdir(demo_dir)

    # Clear the terminal and show logo
    console.clear()
    show_logo()

    # Introduction
    section_header("Introduction to Oumi Platform")
    show_intro()
    pause()

    # Setup & Installation
    section_header("1. Setup & Installation")
    run_command("pip install -U oumi")
    pause()

    run_command("oumi env")
    pause()

    # Model Selection
    section_header("2. Model Selection")
    model_choice, model_name = select_from_choices("Select model type", models)
    console.print(f"\nSelected model: [green]{model_name}[/green]")

    # Dataset Selection
    section_header("3. Dataset Selection")
    dataset_choice, dataset_name = select_from_choices("Select dataset", datasets)
    console.print(
        f"\nSelected dataset: [green]{dataset_choice}[/green] ({dataset_name})"
    )

    # Create training configuration
    section_header("4. Creating Configuration Files")

    # Training type selection

    training_choice, steps_str = select_from_choices(
        "Select training mode", training_options
    )
    console.print(f"\nSelected: [green]{training_choice}[/green]")

    # Create configuration
    train_config = {
        "model": {
            "model_name": model_name,
            "torch_dtype_str": "bfloat16",
            "trust_remote_code": True,
        },
        "data": {
            "train": {
                "datasets": [{"dataset_name": dataset_name}],
                "target_col": "prompt",
            }
        },
        "training": {
            "trainer_type": "TRL_SFT",
            "per_device_train_batch_size": 1,
            "max_steps": int(steps_str),
            "run_name": "demo_model",
            "output_dir": "output",
            "save_final_model": True,
            # "include_performance_metrics": True,
            "enable_wandb": True,
        },
    }

    # Save training config
    create_config_file(train_config, "train_config.yaml")
    console.print("\nCreated training configuration:")
    display_yaml_config(train_config, "Training Configuration")
    pause()

    # Training
    section_header("5. Training")
    run_command("oumi train -c train_config.yaml")
    pause()

    # Model Evaluation
    section_header("6. Model Evaluation")
    console.print("Useful resources for evaluation:")
    console.print(
        "- Evaluation Guide: [blue underline]https://oumi.ai/docs/en/latest/user_guides/evaluate/evaluate.html[/blue underline]"
    )
    console.print(
        "- Available Tasks: [blue underline]https://oumi.ai/docs/en/latest/user_guides/evaluate/evaluate.html#available-tasks[/blue underline]"
    )
    console.print(
        "- Example Configs: [blue underline]https://github.com/oumi-ai/oumi/tree/main/configs/recipes[/blue underline]\n"
    )

    # Select benchmarks
    benchmark_choices = list(benchmarks.keys())
    benchmark_indices = Prompt.ask(
        "Select benchmarks to evaluate (comma-separated numbers)", default="1,3"
    )

    selected_indices = [int(idx.strip()) - 1 for idx in benchmark_indices.split(",")]
    selected_benchmarks = [
        benchmark_choices[i]
        for i in selected_indices
        if 0 <= i < len(benchmark_choices)
    ]

    console.print(
        f"\nSelected benchmarks: [green]{', '.join(selected_benchmarks)}[/green]"
    )

    # Create evaluation configuration
    eval_config = {
        "model": {
            "model_name": "output",  # Use the trained model
            "model_max_length": 2048,
            "torch_dtype_str": "bfloat16",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        },
        "generation": {
            "batch_size": 4,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "tasks": [
            {
                "evaluation_platform": "lm_harness",
                "task_name": benchmarks[benchmark],
                "eval_kwargs": {
                    # "num_fewshot": 5,
                    # "limit": 10,  # Evaluate on full dataset
                },
            }
            for benchmark in selected_benchmarks
        ],
        "output_dir": "eval_results",
        "enable_wandb": False,  # Set to True to enable W&B logging
    }

    # Save evaluation config
    create_config_file(eval_config, "eval_config.yaml")
    console.print("\nCreated evaluation configuration:")
    display_yaml_config(eval_config, "Evaluation Configuration")

    # Run evaluation with progress tracking
    try:
        with console.status("[bold green]Running evaluation...") as status:
            results = run_command("oumi evaluate -c eval_config.yaml")
            console.print(results)
        # Display results
        results_dir = Path("eval_results")
        if results_dir.exists():
            table = Table(title="Evaluation Results")
            table.add_column("Benchmark", style="cyan")
            table.add_column("Metric", style="yellow")
            table.add_column("Score", style="green")

            for task_dir in results_dir.glob("lm_harness*"):
                result_file = task_dir / "task_result.json"
                if result_file.exists():
                    with open(result_file) as f:
                        results = yaml.safe_load(f)
                        for metric, value in results.items():
                            if isinstance(value, (int, float)):
                                table.add_row(
                                    task_dir.name.split("_")[0],
                                    metric,
                                    f"{value:.2%}" if value <= 1 else f"{value:.2f}",
                                )

            console.print(table)
            console.print(
                "\n[green]✓ Evaluation complete! Results saved to eval_results/[/green]"
            )
        else:
            console.print("\n[red]! No evaluation results found[/red]")

    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Error during evaluation: {e}[/red]")
        if hasattr(e, "output"):
            error_output = (
                e.output.decode() if isinstance(e.output, bytes) else str(e.output)
            )
            console.print(f"Output: {error_output}")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")

    pause()

    # Cloud deployment
    section_header("7. Cloud Deployment")
    provider_choice, provider_code = select_from_choices(
        "Select cloud provider for deployment", cloud_providers, default="1"
    )

    hardware_choice, hardware_code = select_from_choices(
        "Select hardware configuration", hardware_options, default="1"
    )

    console.print(
        f"\nSelected: [green]{provider_choice}[/green] with "
        f"[green]{hardware_choice}[/green]"
    )

    # Create sample input file
    sample_conversations = [
        {"messages": [{"role": "user", "content": "What is machine learning?"}]},
        {
            "messages": [
                {"role": "user", "content": "Explain how neural networks work."}
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the applications of AI in healthcare?",
                }
            ]
        },
    ]

    # Write conversations in JSONL format
    with open("test_prompt.jsonl", "w") as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + "\n")

    # Create inference configuration
    infer_config = {
        "model": {
            "model_name": "output",  # Use the trained model
            "model_max_length": 2048,
            "torch_dtype_str": "bfloat16",
            "trust_remote_code": True,
        },
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "input_path": "test_prompt.jsonl",  # Path to input prompts
        # "output_path": "responses.jsonl",  # Path to save responses
    }

    # Save inference config
    create_config_file(infer_config, "infer_config.yaml")
    console.print("\nCreated inference configuration:")
    display_yaml_config(infer_config, "Inference Configuration")

    # Create deployment config
    deploy_config = {
        "name": "oumi-demo-job",
        "resources": {
            "cloud": provider_code,
            "accelerators": hardware_code,
        },
        "working_dir": ".",
        "envs": {"MODEL_NAME": "output"},
        "run": "oumi infer -c infer_config.yaml",
    }

    if provider_code != "local":
        deploy_config["setup"] = "pip install uv && uv pip install oumi[gpu]"
    create_config_file(deploy_config, "job_config.yaml")
    console.print("\nCreated deployment configuration:")
    display_yaml_config(deploy_config, "Deployment Configuration")

    # Launch the deployment
    run_command("oumi launch up -c job_config.yaml")
    pause()

    # Final screen
    console.print(
        "\n[green bold]Thank you for attending this demonstration of the "
        "Oumi platform![/green bold]\n"
    )
    console.print("For more information:")
    console.print(
        "- Documentation: [blue underline]https://oumi.ai/docs[/blue underline]"
    )
    console.print(
        "- GitHub Repository: "
        "[blue underline]https://github.com/oumi-ai/oumi[/blue underline]"
    )
    console.print(
        "- Community: [blue underline]https://discord.gg/oumi[/blue underline]"
    )

    console.print("\n[green bold]Demo complete! Ready for questions.[/green bold]\n")


if __name__ == "__main__":
    app()
