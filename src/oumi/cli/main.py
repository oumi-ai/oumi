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

import os
import sys
import traceback
from typing import Any

import typer

from oumi.cli.alias import AliasType
from oumi.cli.cli_utils import (
    CONSOLE,
    CONTEXT_ALLOW_EXTRA_ARGS,
    create_github_issue_url,
    get_command_help,
)
from oumi.utils.logging import should_use_rich_logging

# =============================================================================
# Lazy CLI module imports
# =============================================================================
# These imports are deferred using TYPE_CHECKING to avoid loading heavy
# dependencies (like pydantic, requests, etc.) at CLI startup time.
# The actual imports happen when commands are registered in get_app().


def _import_cli_modules():
    """Import all CLI command modules. Called lazily when building the app."""
    # Import all CLI modules - they now have optimized (lazy) internal imports
    from oumi.cli import (
        analyze,
        cache,
        deploy,
        distributed_run,
        env,
        evaluate,
        fetch,
        infer,
        judge,
        launch,
        quantize,
        synth,
        train,
        tune,
    )

    return {
        "analyze": analyze.analyze,
        "cache_card": cache.card,
        "cache_get": cache.get,
        "cache_ls": cache.ls,
        "cache_rm": cache.rm,
        "deploy_upload": deploy.upload,
        "deploy_create_endpoint": deploy.create_endpoint,
        "deploy_delete": deploy.delete,
        "deploy_delete_model": deploy.delete_model,
        "deploy_list": deploy.list_deployments,
        "deploy_list_hardware": deploy.list_hardware,
        "deploy_list_models": deploy.list_models,
        "deploy_test": deploy.test,
        "deploy_start": deploy.start,
        "deploy_status": deploy.status,
        "deploy_stop": deploy.stop,
        "deploy_up": deploy.up,
        "accelerate": distributed_run.accelerate,
        "torchrun": distributed_run.torchrun,
        "env": env.env,
        "evaluate": evaluate.evaluate,
        "fetch": fetch.fetch,
        "infer": infer.infer,
        "judge_conversations_file": judge.judge_conversations_file,
        "judge_dataset_file": judge.judge_dataset_file,
        "launch_cancel": launch.cancel,
        "launch_down": launch.down,
        "launch_logs": launch.logs,
        "launch_status": launch.status,
        "launch_stop": launch.stop,
        "launch_up": launch.up,
        "launch_which": launch.which,
        "launch_run": launch.run,
        "quantize": quantize.quantize,
        "synth": synth.synth,
        "train": train.train,
        "tune": tune.tune,
    }


_ASCII_LOGO = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|
"""

_APP_HELP = """\
Examples:

• oumi train -c llama3.1-8b
• oumi infer -c llama3.1-8b --interactive
• oumi train -c config.yaml --training.max_steps 100
"""

_TIPS_FOOTER = """
[bold]Tips:[/bold]
  • List available model configs: [cyan]oumi train --list[/cyan]
  • Enable shell completion: [cyan]oumi --install-completion[/cyan]
"""


def experimental_features_enabled():
    """Check if experimental features are enabled."""
    is_enabled = os.environ.get("OUMI_ENABLE_EXPERIMENTAL_FEATURES", "False")
    return is_enabled.lower() in ("1", "true", "yes", "on")


def _oumi_welcome(
    ctx: typer.Context,
    help_flag: bool = typer.Option(
        False, "--help", "-h", is_eager=True, help="Show this message and exit."
    ),
):
    if ctx.invoked_subcommand == "distributed":
        return
    # Skip logo for rank>0 for multi-GPU jobs to reduce noise in logs.
    if int(os.environ.get("RANK", 0)) > 0:
        return
    CONSOLE.print(_ASCII_LOGO, style="green", highlight=False)

    # Show help when no subcommand is provided or help is requested
    if help_flag or ctx.invoked_subcommand is None:
        CONSOLE.print(ctx.get_help(), end="")
        CONSOLE.print(_TIPS_FOOTER)
        raise typer.Exit


_HELP_OPTION_NAMES = {"help_option_names": ["--help", "-h"]}


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    # Import CLI modules lazily when building the app
    cmds = _import_cli_modules()

    app = typer.Typer(
        pretty_exceptions_enable=False,
        rich_markup_mode="rich",
        context_settings=_HELP_OPTION_NAMES,
        add_completion=True,
    )
    app.callback(invoke_without_command=True, help=_APP_HELP)(_oumi_welcome)

    # Model
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Run benchmarks and evaluations on a model.", AliasType.EVAL
        ),
        rich_help_panel="Model",
    )(cmds["evaluate"])
    app.command(  # Alias for evaluate
        name="eval",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Run benchmarks and evaluations on a model.", AliasType.EVAL
        ),
    )(cmds["evaluate"])
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Generate text or predictions using a model.", AliasType.INFER
        ),
        rich_help_panel="Model",
    )(cmds["infer"])
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help("Fine-tune or pre-train a model.", AliasType.TRAIN),
        rich_help_panel="Model",
    )(cmds["train"])
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help("Search for optimal hyperparameters.", AliasType.TUNE),
        rich_help_panel="Model",
    )(cmds["tune"])
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Compress a model to reduce size and speed up inference.",
            AliasType.QUANTIZE,
        ),
        rich_help_panel="Model",
    )(cmds["quantize"])

    # Data
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Compute statistics and metrics for a dataset.", AliasType.ANALYZE
        ),
        rich_help_panel="Data",
    )(cmds["analyze"])
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Generate synthetic training & evaluation data.", AliasType.SYNTH
        ),
        rich_help_panel="Data",
    )(cmds["synth"])
    app.command(  # Alias for synth
        name="synthesize",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help(
            "Generate synthetic training & evaluation data.", AliasType.SYNTH
        ),
    )(cmds["synth"])
    judge_app = typer.Typer(
        pretty_exceptions_enable=False, context_settings=_HELP_OPTION_NAMES
    )

    # Create callback for --list on top-level judge command
    from oumi.cli.cli_utils import create_list_configs_callback

    _judge_list_callback = create_list_configs_callback(
        AliasType.JUDGE, "Available Judge Configs", "judge dataset"
    )

    _judge_help = get_command_help(
        "Score and evaluate outputs using an LLM judge.", AliasType.JUDGE
    )

    @judge_app.callback(invoke_without_command=True, help=_judge_help)
    def judge_callback(
        ctx: typer.Context,
        list_configs: bool = typer.Option(
            False,
            "--list",
            help="List all available judge configs.",
            callback=_judge_list_callback,
            is_eager=True,
        ),
    ):
        if ctx.invoked_subcommand is None and not list_configs:
            # Show help if no subcommand provided
            CONSOLE.print(ctx.get_help())
            raise typer.Exit(0)

    judge_app.command(
        name="dataset",
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help("Judge a dataset.", AliasType.JUDGE),
    )(cmds["judge_dataset_file"])
    judge_app.command(
        name="conversations",
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help=get_command_help("Judge conversations.", AliasType.JUDGE),
    )(cmds["judge_conversations_file"])
    app.add_typer(
        judge_app,
        name="judge",
        rich_help_panel="Data",
    )

    # Compute
    launch_app = typer.Typer(
        pretty_exceptions_enable=False, context_settings=_HELP_OPTION_NAMES
    )
    launch_app.command(help="Cancel a running job.")(cmds["launch_cancel"])
    launch_app.command(help="Tear down a cluster and release resources.")(
        cmds["launch_down"]
    )
    launch_app.command(
        name="run", context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Execute a job."
    )(cmds["launch_run"])
    launch_app.command(help="Show status of jobs launched from Oumi.")(
        cmds["launch_status"]
    )
    launch_app.command(help="Stop a cluster without tearing it down.")(
        cmds["launch_stop"]
    )
    launch_app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Start a cluster and run a job."
    )(cmds["launch_up"])
    launch_app.command(help="List available cloud providers.")(cmds["launch_which"])
    launch_app.command(help="Fetch logs from a running or completed job.")(
        cmds["launch_logs"]
    )
    app.add_typer(
        launch_app,
        name="launch",
        help="Deploy and manage jobs on cloud infrastructure.",
        rich_help_panel="Compute",
    )
    deploy_app = typer.Typer(
        pretty_exceptions_enable=False, context_settings=_HELP_OPTION_NAMES
    )
    deploy_app.command(help="Upload a model to an inference provider")(
        cmds["deploy_upload"]
    )
    deploy_app.command(help="Create an inference endpoint")(
        cmds["deploy_create_endpoint"]
    )
    deploy_app.command(name="list", help="List all deployments")(cmds["deploy_list"])
    deploy_app.command(name="list-models", help="List uploaded models")(
        cmds["deploy_list_models"]
    )
    deploy_app.command(name="status", help="Get deployment status")(
        cmds["deploy_status"]
    )
    deploy_app.command(name="start", help="Start a stopped endpoint")(
        cmds["deploy_start"]
    )
    deploy_app.command(name="stop", help="Stop an endpoint to save cost")(
        cmds["deploy_stop"]
    )
    deploy_app.command(help="Delete an endpoint")(cmds["deploy_delete"])
    deploy_app.command(name="delete-model", help="Delete an uploaded model")(
        cmds["deploy_delete_model"]
    )
    deploy_app.command(help="List available hardware options")(
        cmds["deploy_list_hardware"]
    )
    deploy_app.command(help="Test endpoint with a sample request")(cmds["deploy_test"])
    deploy_app.command(help="Deploy model end-to-end (upload + endpoint)")(
        cmds["deploy_up"]
    )
    app.add_typer(
        deploy_app,
        name="deploy",
        help="Deploy models to inference providers.",
        rich_help_panel="Compute",
    )
    distributed_app = typer.Typer(
        pretty_exceptions_enable=False, context_settings=_HELP_OPTION_NAMES
    )
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        cmds["accelerate"]
    )
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(cmds["torchrun"])
    app.add_typer(
        distributed_app,
        name="distributed",
        help="Run multi-GPU training locally.",
        rich_help_panel="Compute",
    )
    app.command(
        help="Show status of launched jobs and clusters.",
        rich_help_panel="Compute",
    )(cmds["launch_status"])

    # Tools
    app.command(
        help="Show Oumi environment and system information.",
        rich_help_panel="Tools",
    )(cmds["env"])
    app.command(
        help="Download example configs from the Oumi repository.",
        rich_help_panel="Tools",
    )(cmds["fetch"])
    cache_app = typer.Typer(
        pretty_exceptions_enable=False, context_settings=_HELP_OPTION_NAMES
    )
    cache_app.command(name="ls", help="List cached models and datasets.")(
        cmds["cache_ls"]
    )
    cache_app.command(
        name="get", help="Download a model or dataset from Hugging Face."
    )(cmds["cache_get"])
    cache_app.command(name="card", help="Show details for a cached item.")(
        cmds["cache_card"]
    )
    cache_app.command(name="rm", help="Remove items from the local cache.")(
        cmds["cache_rm"]
    )
    app.add_typer(
        cache_app,
        name="cache",
        help="Manage locally cached models and datasets.",
        rich_help_panel="Tools",
    )

    return app


def _get_cli_event() -> tuple[str, dict[str, Any]]:
    """Extract the CLI command and context from sys.argv."""
    args = sys.argv[1:]
    help_requested = "--help" in args or "-h" in args

    # Extract positional arguments that appear before any flag.
    # This correctly handles the common CLI patterns where commands/subcommands
    # come first, followed by flags and their values.
    positional_args = []
    for arg in args:
        if arg.startswith("-"):
            break
        positional_args.append(arg)
        if len(positional_args) >= 2:
            break

    command = positional_args[0] if positional_args else None
    subcommand = positional_args[1] if len(positional_args) > 1 else None

    event_name = f"cli-{command}" if command else "cli"
    properties: dict[str, Any] = {
        "subcommand": subcommand,
        "help": help_requested,
    }

    return event_name, properties


def _is_completion_mode() -> bool:
    """Check if running in shell completion mode."""
    return any(
        os.environ.get(key)
        for key in ("_OUMI_COMPLETE", "COMP_WORDS", "COMP_LINE", "_TYPER_COMPLETE")
    )


def run():
    """The entrypoint for the CLI."""
    app = get_app()

    # Skip telemetry for shell completions and help requests
    # Telemetry imports torch which adds ~0.6s overhead
    if _is_completion_mode():
        return app()

    try:
        event_name, event_properties = _get_cli_event()
        if event_properties.get("help"):
            return app()
        else:
            from oumi.telemetry import TelemetryManager

            telemetry = TelemetryManager.get_instance()
            with telemetry.capture_operation(event_name, event_properties):
                return app()
    except Exception as e:
        tb_str = traceback.format_exc()
        CONSOLE.print(tb_str)
        issue_url = create_github_issue_url(e, tb_str)
        CONSOLE.print(
            "\n[red]If you believe this is a bug, please file an issue:[/red]"
        )
        if should_use_rich_logging():
            CONSOLE.print(
                f"📝 [yellow]Templated issue:[/yellow] "
                f"[link={issue_url}]Click here to report[/link]"
            )
        else:
            CONSOLE.print(
                "https://github.com/oumi-ai/oumi/issues/new?template=bug-report.yaml"
            )

        sys.exit(1)


if "sphinx" in sys.modules:
    # Create the CLI app when building the docs to auto-generate the CLI reference.
    app = get_app()
