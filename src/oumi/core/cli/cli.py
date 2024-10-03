import typer

import oumi.core.cli.train

app = typer.Typer()
app.command()(oumi.core.cli.train.train)


if __name__ == "__main__":
    app()
