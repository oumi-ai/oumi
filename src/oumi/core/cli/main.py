import typer

from oumi.core.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.cli.evaluate import evaluate
from oumi.core.cli.infer import infer
from oumi.core.cli.train import train


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer()
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(evaluate)
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(infer)
    app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(train)
    return app


def run():
    """The entrypoint for the CLI."""
    app = get_app()
    return app()
