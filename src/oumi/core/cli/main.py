import typer

from oumi.core.cli.evaluate import evaluate
from oumi.core.cli.infer import infer
from oumi.core.cli.train import train


def get_app() -> typer.Typer:
    """Create the Typer CLI app."""
    app = typer.Typer()
    app.command()(evaluate)
    app.command()(infer)
    app.command()(train)
    return app


def run():
    """The entrypoint for the CLI."""
    app = get_app()
    return app()
