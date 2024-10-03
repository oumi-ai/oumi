import typer

import oumi.core.cli.evaluate
import oumi.core.cli.train

app = typer.Typer()
app.command()(oumi.core.cli.train.train)
app.command()(oumi.core.cli.evaluate.evaluate)


def run():
    """The entrypoint for the CLI."""
    return app()
