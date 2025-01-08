import os
import sys

import typer

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.distributed_run import accelerate, torchrun
from oumi.cli.env import env
from oumi.cli.evaluate import evaluate
from oumi.cli.infer import infer
from oumi.cli.judge import conversations, dataset, model
from oumi.cli.launch import cancel, down, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.cli.train import train

_ASCII_LOGO = """
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@************%@@%**#@@@@@@***@@@@@
@@@@@.           #@@*  :@@@@@@. .@@@@@
@@@@@. .%@@@@@:  #@@*  :@@@@@@. .@@@@@
@@@@@. .@@@@@@:  #@@*  :@@@@@@. .@@@@@
@@@@@. .@@@@@@:  #@@*  :@@@@@@. .@@@@@
@@@@@. .+*****.  #@@*  .*****+. .@@@@@
@@@@@.           #@@*           .@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@.                    .@@@. .@@@@@
@@@@@. .++++++.  =+++++=  .@@@. .@@@@@
@@@@@. .@@@@@@:  #@@@@@%  .@@@. .@@@@@
@@@@@. .@@@@@@:  #@@@@@%  .@@@. .@@@@@
@@@@@. .@@@@@@:  #@@@@@%  .@@@. .@@@@@
@@@@@. .@@@@@@:  #@@@@@%  .@@@. .@@@@@
@@@@@***@@@@@@***%@@@@@@***@@@***@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""


def _oumi_welcome(ctx: typer.Context):
    if ctx.invoked_subcommand == "distributed":
        return
    # Skip logo for rank>0 for multi-GPU jobs to reduce noise in logs.
    if int(os.environ.get("RANK", 0)) > 0:
        return
    print(_ASCII_LOGO)


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer(pretty_exceptions_enable=False)
    app.callback()(_oumi_welcome)
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
    return app


def run():
    """The entrypoint for the CLI."""
    app = get_app()
    return app()


if "sphinx" in sys.modules:
    # Create the CLI app when building the docs to auto-generate the CLI reference.
    app = get_app()
