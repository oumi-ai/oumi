import typer

from oumi.core.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.cli.evaluate import evaluate
from oumi.core.cli.infer import infer
from oumi.core.cli.judge import conversations, dataset
from oumi.core.cli.launch import down, status, stop, up, which
from oumi.core.cli.launch import run as launcher_run
from oumi.core.cli.train import train


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(evaluate)
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(infer)
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(train)

    judge_app = typer.Typer(pretty_exceptions_enable=False)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(conversations)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(dataset)
    app.add_typer(judge_app, name="judge", help="Judge datasets or conversations.")

    launch_app = typer.Typer(pretty_exceptions_enable=False)
    launch_app.command()(down)
    launch_app.command(name="run", context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        launcher_run
    )
    launch_app.command()(status)
    launch_app.command()(stop)
    launch_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(up)
    launch_app.command()(which)
    app.add_typer(launch_app, name="launch", help="Launch jobs remotely.")
    return app


def run():
    """The entrypoint for the CLI."""
    app = get_app()
    return app()
