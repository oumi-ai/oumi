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

import typer

from oumi.cli.analyze import analyze
from oumi.cli.cache import card as cache_card
from oumi.cli.cache import get as cache_get
from oumi.cli.cache import ls as cache_ls
from oumi.cli.cache import rm as cache_rm
from oumi.cli.cli_utils import (
    CONSOLE,
    CONTEXT_ALLOW_EXTRA_ARGS,
    create_github_issue_url,
)
from oumi.cli.distributed_run import accelerate, torchrun
from oumi.cli.env import env
from oumi.cli.evaluate import evaluate
from oumi.cli.fetch import fetch
from oumi.cli.infer import infer
from oumi.cli.judge import judge_conversations_file, judge_dataset_file
from oumi.cli.launch import cancel, down, logs, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.cli.quantize import quantize
from oumi.cli.synth import synth
from oumi.cli.train import train
from oumi.cli.tune import tune
from oumi.utils.logging import should_use_rich_logging

_ASCII_LOGO = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|
"""


def experimental_features_enabled():
    """Check if experimental features are enabled."""
    is_enabled = os.environ.get("OUMI_ENABLE_EXPERIMENTAL_FEATURES", "False")
    return is_enabled.lower() in ("1", "true", "yes", "on")


def _oumi_welcome(ctx: typer.Context):
    if ctx.invoked_subcommand == "distributed":
        return
    # Skip logo for rank>0 for multi-GPU jobs to reduce noise in logs.
    if int(os.environ.get("RANK", 0)) > 0:
        return
    CONSOLE.print(_ASCII_LOGO, style="green", highlight=False)


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer(pretty_exceptions_enable=False)
    app.callback(context_settings={"help_option_names": ["-h", "--help"]})(
        _oumi_welcome
    )

    # Model
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Run benchmarks and evaluations on a model.",
        rich_help_panel="Model",
    )(evaluate)
    app.command(  # Alias for evaluate
        name="eval",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Run benchmarks and evaluations on a model.",
    )(evaluate)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Generate text or predictions using a model.",
        rich_help_panel="Model",
    )(infer)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Fine-tune or pre-train a model.",
        rich_help_panel="Model",
    )(train)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Search for optimal hyperparameters.",
        rich_help_panel="Model",
    )(tune)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Compress a model to reduce size and speed up inference.",
        rich_help_panel="Model",
    )(quantize)

    # Data
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Compute statistics and metrics for a dataset.",
        rich_help_panel="Data",
    )(analyze)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Generate synthetic training & evaluation data.",
        rich_help_panel="Data",
    )(synth)
    app.command(  # Alias for synth
        name="synthesize",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Generate synthetic training data.",
    )(synth)
    judge_app = typer.Typer(pretty_exceptions_enable=False)
    judge_app.command(name="dataset", context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        judge_dataset_file
    )
    judge_app.command(name="conversations", context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        judge_conversations_file
    )
    app.add_typer(
        judge_app,
        name="judge",
        help="Score and evaluate outputs using an LLM judge.",
        rich_help_panel="Data",
    )

    # Compute
    launch_app = typer.Typer(pretty_exceptions_enable=False)
    launch_app.command(help="Cancel a running job.")(cancel)
    launch_app.command(help="Tear down a cluster and release resources.")(down)
    launch_app.command(
        name="run", context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Execute a job."
    )(launcher_run)
    launch_app.command(help="Show status of jobs launched from Oumi.")(status)
    launch_app.command(help="Stop a cluster without tearing it down.")(stop)
    launch_app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Start a cluster and run a job."
    )(up)
    launch_app.command(help="List available cloud providers.")(which)
    launch_app.command(help="Fetch logs from a running or completed job.")(logs)
    app.add_typer(
        launch_app,
        name="launch",
        help="Deploy and manage jobs on cloud infrastructure.",
        rich_help_panel="Compute",
    )
    distributed_app = typer.Typer(pretty_exceptions_enable=False)
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(accelerate)
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(torchrun)
    app.add_typer(
        distributed_app,
        name="distributed",
        help="Run multi-GPU training locally.",
        rich_help_panel="Compute",
    )
    app.command(
        help="Show status of launched jobs and clusters.",
        rich_help_panel="Compute",
    )(status)

    # Tools
    app.command(
        help="Show Oumi environment and system information.",
        rich_help_panel="Tools",
    )(env)
    app.command(
        help="Download example configs from the Oumi repository.",
        rich_help_panel="Tools",
    )(fetch)
    cache_app = typer.Typer(pretty_exceptions_enable=False)
    cache_app.command(name="ls", help="List cached models and datasets.")(cache_ls)
    cache_app.command(
        name="get", help="Download a model or dataset from Hugging Face."
    )(cache_get)
    cache_app.command(name="card", help="Show details for a cached item.")(cache_card)
    cache_app.command(name="rm", help="Remove items from the local cache.")(cache_rm)
    app.add_typer(
        cache_app,
        name="cache",
        help="Manage locally cached models and datasets.",
        rich_help_panel="Tools",
    )

    return app


def run():
    """The entrypoint for the CLI."""
    app = get_app()
    try:
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
