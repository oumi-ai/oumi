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

"""Interactive getting started wizard for Oumi."""

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.panel import Panel

from oumi.cli.cli_utils import CONSOLE, section_header

_WIZARD_ASCII = r"""
    __  ___            _
   /  |/  /___ _____ _(_)____
  / /|_/ / __ `/ __ `/ / ___/
 / /  / / /_/ / /_/ / / /__
/_/  /_/\__,_/\__, /_/\___/
             /____/
"""

# Model families with their characteristics
MODEL_CATALOG = {
    "smollm": {
        "name": "SmolLM",
        "description": "Tiny, fast models (135M params) - great for learning & CPU",
        "sizes": ["135m"],
        "recommended": True,
        "quick_start": True,
    },
    "phi3": {
        "name": "Phi-3",
        "description": "Small, efficient models from Microsoft (3.8B params)",
        "sizes": ["3.8b"],
        "recommended": True,
    },
    "phi4": {
        "name": "Phi-4",
        "description": "Advanced reasoning model (14B params)",
        "sizes": ["14b"],
    },
    "llama3_2": {
        "name": "Llama 3.2",
        "description": "Latest Meta models - balanced performance (1B, 3B)",
        "sizes": ["1b", "3b"],
        "recommended": True,
    },
    "llama3_3": {
        "name": "Llama 3.3",
        "description": "Powerful 70B parameter model from Meta",
        "sizes": ["70b"],
    },
    "llama3_1": {
        "name": "Llama 3.1",
        "description": "Powerful models from Meta (8B, 70B, 405B)",
        "sizes": ["8b", "70b", "405b"],
    },
    "llama4": {
        "name": "Llama 4",
        "description": "Next-gen Meta models (Scout & Maverick variants)",
        "sizes": ["scout", "maverick"],
    },
    "qwen3": {
        "name": "Qwen 3",
        "description": "Alibaba's multilingual models (0.6B - 235B)",
        "sizes": ["0.6b", "1.7b", "8b", "32b", "70b", "235b"],
    },
    "qwen2_5": {
        "name": "Qwen 2.5",
        "description": "Previous generation Qwen models (3B, 7B)",
        "sizes": ["3b", "7b"],
    },
    "qwen3_coder": {
        "name": "Qwen 3 Coder",
        "description": "Code-specialized Qwen models",
        "sizes": ["30b"],
    },
    "qwq": {
        "name": "QwQ",
        "description": "Qwen's reasoning-focused model (32B)",
        "sizes": ["32b"],
    },
    "deepseek_r1": {
        "name": "DeepSeek R1",
        "description": "Advanced reasoning models (671B, distilled variants)",
        "sizes": ["671b", "distill_llama_8b", "distill_llama_70b", "distill_qwen_14b"],
    },
    "gemma3": {
        "name": "Gemma 3",
        "description": "Google's efficient models",
        "sizes": ["3n_e4b"],
    },
    "glm4": {
        "name": "GLM-4",
        "description": "Tsinghua's GLM models",
        "sizes": ["air"],
    },
    "falcon_h1": {
        "name": "Falcon H1",
        "description": "TII's Falcon models (0.5B, 1.5B, 7B)",
        "sizes": ["0.5b", "1.5b", "7b"],
    },
    "falcon_e": {
        "name": "Falcon E",
        "description": "Falcon Edge models (1B)",
        "sizes": ["1b"],
    },
    "gpt_oss": {
        "name": "GPT-OSS",
        "description": "Open-source GPT models (20B, 120B)",
        "sizes": ["20b", "120b"],
    },
    "gpt2": {
        "name": "GPT-2",
        "description": "Classic OpenAI model",
        "sizes": ["124m", "355m", "774m", "1.5b"],
    },
}


def _get_configs_dir() -> Path:
    """Get the configs directory path."""
    # Assuming this file is in src/oumi/cli/magic.py
    return Path(__file__).parent.parent.parent.parent / "configs" / "recipes"


def _find_config_files(model_family: str, workflow: str) -> list[Path]:
    """Find config files for a model family and workflow.

    Args:
        model_family: Model family name (e.g., 'smollm', 'llama3_2')
        workflow: Workflow type ('sft', 'evaluation', 'inference', 'dpo', 'kto')

    Returns:
        List of config file paths
    """
    configs_dir = _get_configs_dir()
    model_dir = configs_dir / model_family / workflow

    if not model_dir.exists():
        return []

    # Find all YAML files, excluding job configs
    yaml_files = list(model_dir.rglob("*.yaml"))

    # Filter out job configs and focus on main training/eval/infer configs
    main_configs = [
        f
        for f in yaml_files
        if not any(
            x in f.name
            for x in [
                "gcp_job",
                "slurm_job",
                "polaris_job",
                "lambda_job",
                "fsdp_",
                "leaderboard",
            ]
        )
    ]

    return sorted(main_configs)


def _display_welcome():
    """Display welcome screen."""
    CONSOLE.print(_WIZARD_ASCII, style="cyan", highlight=False)
    CONSOLE.print(
        Panel.fit(
            "[bold cyan]Welcome to Oumi Magic![/bold cyan]\n\n"
            "This wizard will guide you through your first Oumi workflow.\n"
            "You can train, evaluate, or run inference with state-of-the-art models.",
            border_style="cyan",
        )
    )
    CONSOLE.print()


def _prompt_workflow() -> str:
    """Prompt user to select a workflow type.

    Returns:
        Workflow type: 'sft', 'evaluation', 'inference', 'dpo', 'full'
    """
    section_header("Step 1: Choose your workflow")

    CONSOLE.print("What would you like to do?\n")
    CONSOLE.print("  [cyan]1[/cyan]. Train a model (SFT - Supervised Fine-Tuning)")
    CONSOLE.print("  [cyan]2[/cyan]. Evaluate a model")
    CONSOLE.print("  [cyan]3[/cyan]. Run inference (chat with a model)")
    CONSOLE.print("  [cyan]4[/cyan]. Preference alignment (DPO/KTO)")
    CONSOLE.print("  [cyan]5[/cyan]. Full pipeline (train → evaluate → infer)")
    CONSOLE.print()

    choice = typer.prompt("Enter your choice [1-5]", type=int, default=1)

    workflow_map = {
        1: "sft",
        2: "evaluation",
        3: "inference",
        4: "preference",  # Will check both DPO and KTO
        5: "full",
    }

    if choice not in workflow_map:
        CONSOLE.print("[red]Invalid choice. Using default: Training (SFT)[/red]")
        return "sft"

    return workflow_map[choice]


def _prompt_model_selection(workflow: str) -> tuple[str, Optional[Path]]:
    """Prompt user to select a model.

    Args:
        workflow: The selected workflow type

    Returns:
        Tuple of (model_family, config_path)
    """
    section_header("Step 2: Choose a model")

    # Filter models that have configs for this workflow
    available_models = []
    for family, info in MODEL_CATALOG.items():
        configs = _find_config_files(family, workflow)
        if configs:
            available_models.append((family, info, configs))

    if not available_models:
        CONSOLE.print(f"[red]No models found for workflow: {workflow}[/red]")
        raise typer.Exit(1)

    # Show recommended models first
    recommended = [m for m in available_models if m[1].get("recommended", False)]
    others = [m for m in available_models if not m[1].get("recommended", False)]

    CONSOLE.print("Recommended models:\n")
    idx = 1
    model_map = {}

    for family, info, configs in recommended:
        CONSOLE.print(
            f"  [cyan]{idx}[/cyan]. [bold]{info['name']}[/bold] - {info['description']}"
        )
        CONSOLE.print(f"       {len(configs)} config(s) available")
        model_map[idx] = (family, configs)
        idx += 1

    if others:
        CONSOLE.print("\nOther models:\n")
        for family, info, configs in others:
            CONSOLE.print(
                f"  [cyan]{idx}[/cyan]. [bold]{info['name']}[/bold] - {info['description']}"  # noqa: E501
            )
            CONSOLE.print(f"       {len(configs)} config(s) available")
            model_map[idx] = (family, configs)
            idx += 1

    CONSOLE.print()
    choice = typer.prompt(
        f"Enter your choice [1-{len(model_map)}]", type=int, default=1
    )

    if choice not in model_map:
        CONSOLE.print("[yellow]Invalid choice. Using first recommended model.[/yellow]")
        choice = 1

    model_family, configs = model_map[choice]

    # If multiple configs, let user choose
    if len(configs) > 1:
        CONSOLE.print(
            f"\n[bold]Available configurations for {MODEL_CATALOG[model_family]['name']}:[/bold]\n"  # noqa: E501
        )
        for i, cfg in enumerate(configs, 1):
            # Show relative path from configs/recipes
            rel_path = cfg.relative_to(_get_configs_dir())
            CONSOLE.print(f"  [cyan]{i}[/cyan]. {rel_path}")

        CONSOLE.print()
        cfg_choice = typer.prompt(
            f"Choose a configuration [1-{len(configs)}]", type=int, default=1
        )

        if 1 <= cfg_choice <= len(configs):
            config_path = configs[cfg_choice - 1]
        else:
            CONSOLE.print("[yellow]Invalid choice. Using first configuration.[/yellow]")
            config_path = configs[0]
    else:
        config_path = configs[0]

    return model_family, config_path


def _prompt_execution_mode() -> str:
    """Prompt user for execution mode.

    Returns:
        Execution mode: 'run', 'print', 'save'
    """
    section_header("Step 3: Execution mode")

    CONSOLE.print("How would you like to proceed?\n")
    CONSOLE.print("  [cyan]1[/cyan]. Run the command now")
    CONSOLE.print("  [cyan]2[/cyan]. Print the command (don't run)")
    CONSOLE.print("  [cyan]3[/cyan]. Save command to a script file")
    CONSOLE.print()

    choice = typer.prompt("Enter your choice [1-3]", type=int, default=1)

    mode_map = {1: "run", 2: "print", 3: "save"}
    return mode_map.get(choice, "print")


def _build_command(workflow: str, config_path: Optional[Path]) -> list[str]:
    """Build the oumi command to execute.

    Args:
        workflow: Workflow type
        config_path: Path to config file

    Returns:
        Command as list of strings
    """
    workflow_cmd_map = {
        "sft": "train",
        "evaluation": "evaluate",
        "inference": "infer",
        "dpo": "train",  # DPO uses train command
        "kto": "train",  # KTO uses train command
        "preference": "train",
    }

    cmd = workflow_cmd_map.get(workflow, "train")
    return ["oumi", cmd, "-c", str(config_path)]


def _execute_command(command: list[str], mode: str):
    """Execute or display the command based on mode.

    Args:
        command: Command to execute
        mode: Execution mode ('run', 'print', 'save')
    """
    cmd_str = " ".join(command)

    if mode == "print":
        section_header("Command to run")
        CONSOLE.print(Panel(cmd_str, style="green"))
        CONSOLE.print("\n[dim]Copy and paste this command to run it manually.[/dim]")

    elif mode == "save":
        script_path = Path("oumi_wizard.sh")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated by oumi magic\n\n")
            f.write(f"{cmd_str}\n")
        script_path.chmod(0o755)

        CONSOLE.print(f"[green]✓[/green] Command saved to: [bold]{script_path}[/bold]")
        CONSOLE.print(f"\nRun it with: [cyan]./{script_path}[/cyan]")

    elif mode == "run":
        section_header("Running command")
        CONSOLE.print(f"[dim]$ {cmd_str}[/dim]\n")

        try:
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                CONSOLE.print("\n[green]✓ Command completed successfully![/green]")
            else:
                CONSOLE.print(
                    f"\n[yellow]Command exited with code {result.returncode}[/yellow]"
                )
        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]Interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            CONSOLE.print(f"\n[red]Error running command: {e}[/red]")
            raise typer.Exit(1)


def _display_next_steps(workflow: str, model_family: str):
    """Display helpful next steps after wizard completion.

    Args:
        workflow: The workflow that was run
        model_family: The model family that was used
    """
    section_header("Next steps")

    tips = []

    if workflow == "sft":
        tips.append(
            "📊 Evaluate your trained model: [cyan]oumi evaluate -c <eval_config>[/cyan]"  # noqa: E501
        )
        tips.append(
            "💬 Chat with your model: [cyan]oumi infer -c <infer_config> --interactive[/cyan]"  # noqa: E501
        )
        tips.append("📁 Find your model in the [bold]output/[/bold] directory")
    elif workflow == "evaluation":
        tips.append("📈 View results in the output directory")
        tips.append("🔄 Try different evaluation benchmarks")
    elif workflow == "inference":
        tips.append(
            "🚀 Try with vLLM for faster inference: add [bold]engine: vllm[/bold] to config"  # noqa: E501
        )
        tips.append("🌐 Deploy as an API server with: [cyan]oumi infer --server[/cyan]")

    tips.append(
        f"📖 Explore more configs: [bold]configs/recipes/{model_family}/[/bold]"
    )
    tips.append(
        "❓ Get help: [cyan]oumi --help[/cyan] or visit [link=https://oumi.ai/docs]oumi.ai/docs[/link]"
    )
    tips.append("💡 List all models: [cyan]oumi list models[/cyan]")
    tips.append("📊 List all datasets: [cyan]oumi list datasets[/cyan]")

    for tip in tips:
        CONSOLE.print(f"  • {tip}")

    CONSOLE.print()
    CONSOLE.print(
        Panel.fit(
            "⭐ [bold]Enjoying Oumi?[/bold] Star us on GitHub: "
            "[link=https://github.com/oumi-ai/oumi]github.com/oumi-ai/oumi[/link]",
            border_style="cyan",
        )
    )


def magic(
    workflow: Annotated[
        Optional[str],
        typer.Option(
            "--workflow",
            "-w",
            help="Workflow type (sft, evaluation, inference, dpo, kto)",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model family to use"),
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Run in non-interactive mode"),
    ] = False,
):
    """Interactive getting started wizard for Oumi.

    This wizard helps you get started with Oumi by guiding you through:
    - Choosing a workflow (train, evaluate, infer)
    - Selecting a model
    - Running your first command

    Examples:
        oumi magic                                    # Interactive mode
        oumi magic --workflow sft --model smollm      # Non-interactive mode
    """
    try:
        # Check if running in a TTY
        if not non_interactive and not sys.stdout.isatty():
            CONSOLE.print(
                "[yellow]Warning: Not running in a TTY. "
                "Use --non-interactive for non-TTY environments.[/yellow]\n"
            )

        # Interactive mode
        if not non_interactive:
            _display_welcome()

            # Step 1: Choose workflow
            selected_workflow = workflow or _prompt_workflow()

            # Handle full pipeline separately
            if selected_workflow == "full":
                CONSOLE.print(
                    "\n[yellow]Full pipeline mode coming soon! "
                    "For now, run train, evaluate, and infer separately.[/yellow]"
                )
                raise typer.Exit(0)

            # Step 2: Choose model
            model_family, config_path = _prompt_model_selection(selected_workflow)

            # Step 3: Choose execution mode
            exec_mode = _prompt_execution_mode()

        else:
            # Non-interactive mode
            if not workflow:
                CONSOLE.print(
                    "[red]Error: --workflow required in non-interactive mode[/red]"
                )
                raise typer.Exit(1)

            if not model:
                CONSOLE.print(
                    "[red]Error: --model required in non-interactive mode[/red]"
                )
                raise typer.Exit(1)

            selected_workflow = workflow
            model_family = model

            # Find first config for this model/workflow
            configs = _find_config_files(model_family, selected_workflow)
            if not configs:
                CONSOLE.print(
                    f"[red]No configs found for {model_family}/{selected_workflow}[/red]"  # noqa: E501
                )
                raise typer.Exit(1)

            config_path = configs[0]
            exec_mode = "print"

        # Build and execute command
        command = _build_command(selected_workflow, config_path)
        _execute_command(command, exec_mode)

        # Show next steps
        if exec_mode != "run":
            _display_next_steps(selected_workflow, model_family)

    except KeyboardInterrupt:
        CONSOLE.print("\n\n[yellow]Wizard cancelled by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        CONSOLE.print(f"\n[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
