import shutil
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def copy_doc_files(
    docs_sourcedir: Path = typer.Argument(
        ..., help="Path to the docs source directory"
    ),
):
    """Copy documentation files as specified in _doc_links.txt."""
    doc_links_file = docs_sourcedir / "_doc_links.txt"

    if not doc_links_file.exists():
        typer.echo(f"Warning: {doc_links_file} not found. Skipping file copy.")
        return

    with doc_links_file.open("r") as f:
        for line in f:
            src, dst = map(str.strip, line.strip().split("|"))
            src_path = (docs_sourcedir / Path(src)).resolve()
            dst_path = docs_sourcedir / dst

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dst_path)
            typer.echo(f"Copied {src_path} to {dst_path}")


if __name__ == "__main__":
    app()
