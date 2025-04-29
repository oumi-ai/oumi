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
from typing import Optional

import typer
from rich.panel import Panel

from oumi.cli.cli_utils import CONSOLE, CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.distributed_run import accelerate, torchrun
from oumi.cli.env import env
from oumi.cli.evaluate import evaluate
from oumi.cli.fetch import fetch
from oumi.cli.infer import infer
from oumi.cli.judge import conversations, dataset, model
from oumi.cli.launch import cancel, down, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.cli.train import train
from oumi.utils.cli_styling import StyleLevel, get_style_level

_ASCII_LOGO = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|"""

_FANCY_LOGO = r"""
  ██████╗ ██╗   ██╗███╗   ███╗██╗
 ██╔═══██╗██║   ██║████╗ ████║██║
 ██║   ██║██║   ██║██╔████╔██║██║
 ██║   ██║██║   ██║██║╚██╔╝██║██║
 ╚██████╔╝╚██████╔╝██║ ╚═╝ ██║██║
  ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝"""


def _oumi_welcome(ctx: typer.Context):
    # Skip logo for distributed subcommand or non-primary ranks
    if ctx.invoked_subcommand == "distributed" or int(os.environ.get("RANK", 0)) > 0:
        return

    # Get version information for display
    version = ctx.obj.get("version") if ctx.obj else None
    version_text = f" v{version}" if version else ""

    # Select logo style based on styling setting
    if get_style_level() == StyleLevel.NONE:
        CONSOLE.print(_ASCII_LOGO)
    else:
        # Create fancy panel with logo for FULL styling mode
        CONSOLE.print(
            Panel(
                _FANCY_LOGO,
                title=f"Oumi{version_text}",
                title_align="center",
                subtitle="AI Model Platform",
                subtitle_align="center",
                border_style="primary",
                padding=(1, 2),
            )
        )


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer(pretty_exceptions_enable=False)

    @app.callback()
    def callback(ctx: typer.Context):
        # Initialize ctx.obj if it doesn't exist
        if ctx.obj is None:
            ctx.obj = {}

        # Get version from package
        try:
            from oumi.utils.version_utils import get_version

            ctx.obj["version"] = get_version()
        except ImportError:
            ctx.obj["version"] = "unknown"

        # Display welcome message
        _oumi_welcome(ctx)

    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Evaluate a model.",
    )(evaluate)
    app.command()(env)
    app.command(  # Alias for evaluate
        name="eval",
        hidden=True,
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Evaluate a model.",
    )(evaluate)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Run inference on a model.",
    )(infer)
    app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS,
        help="Train a model.",
    )(train)

    judge_app = typer.Typer(pretty_exceptions_enable=False)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(conversations)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(dataset)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(model)
    app.add_typer(
        judge_app, name="judge", help="Judge datasets, models or conversations."
    )

    launch_app = typer.Typer(pretty_exceptions_enable=False)
    launch_app.command(help="Cancels a job.")(cancel)
    launch_app.command(help="Turns down a cluster.")(down)
    launch_app.command(
        name="run", context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Runs a job."
    )(launcher_run)
    launch_app.command(help="Prints the status of jobs launched from Oumi.")(status)
    launch_app.command(help="Stops a cluster.")(stop)
    launch_app.command(
        context_settings=CONTEXT_ALLOW_EXTRA_ARGS, help="Launches a job."
    )(up)
    launch_app.command(help="Prints the available clouds.")(which)
    app.add_typer(launch_app, name="launch", help="Launch jobs remotely.")

    distributed_app = typer.Typer(pretty_exceptions_enable=False)
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(accelerate)
    distributed_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(torchrun)
    app.add_typer(
        distributed_app,
        name="distributed",
        help=(
            "A wrapper for torchrun/accelerate "
            "with reasonable default values for distributed training."
        ),
    )

    app.command(
        help="Fetch configuration files from the oumi GitHub repository.",
    )(fetch)

    return app


def run():
    """The entrypoint for the CLI."""
    app = get_app()
    return app()


if "sphinx" in sys.modules:
    # Create the CLI app when building the docs to auto-generate the CLI reference.
    app = get_app()
