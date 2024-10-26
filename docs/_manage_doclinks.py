"""Script to copy and clean up documentation files.

This is needed to generate the documentation for the oumi package.

Sphinx requires the documentation files to be in the source tree in order
to build the docs, but many files are better suited to be stored outside
the docs source tree (e.g. notebooks, README files).

Usage examples:

.. code-block:: bash

    # Copy documentation files
    python _manage_doclinks.py copy path/to/doc_links.csv

    # Clean up documentation files
    python _manage_doclinks.py clean path/to/doc_links.csv

    # Adding --absolute-paths will use absolute paths
    # for the source and destination files
    python _manage_doclinks.py copy path/to/doc_links.csv --absolute-paths


The doc_links.csv file should be a CSV file with '|' as the delimiter,
containing two columns: source path and destination path.
"""

import csv
import shutil
from collections.abc import Iterable
from pathlib import Path

import typer

app = typer.Typer()

# Get the root directory of the oumi git repository
OUMI_ROOT: Path = Path(__file__).parent.parent


def read_doc_links(doc_links_config: Path) -> Iterable[tuple[str, str]]:
    """Read the doc links config and yield source and destination paths.

    The doc links file should be a CSV file with '|' as the delimiter.
    Each row should contain two columns: source path and destination path.

    Args:
        doc_links_config (Path): Path to the doc links configuration file.

    Yields:
        tuple: A tuple containing (source_path, destination_path), both as Path objects.
    """
    with doc_links_config.open("r") as f:
        reader = csv.reader(f, delimiter="|")
        for source, destination in reader:
            source = source.strip()
            destination = destination.strip()
            yield source, destination


@app.command()
def copy(
    doc_links_config: Path = typer.Argument(..., help="Path to the doc links config"),
    absolute_paths: bool = typer.Option(
        False, help="Whether to use absolute paths for the source files"
    ),
):
    """Copy documentation files as specified in the doc links config.

    This function reads the doc links configuration file and copies the specified
    files from their source locations to the destination paths. It can handle both
    relative and absolute paths based on the `absolute_paths` parameter.

    Args:
        doc_links_config (Path): Path to the doc links configuration file.
        absolute_paths (bool): If True, use absolute paths for source files.
            If False, paths are relative to OUMI_ROOT.
    """
    if not doc_links_config.exists():
        typer.echo(f"Warning: {doc_links_config} not found. Skipping file copy.")
        return

    for source, destination in read_doc_links(doc_links_config):
        if not absolute_paths:
            source_path = OUMI_ROOT / source
            destination_path = OUMI_ROOT / destination
        else:
            source_path = Path(source)
            destination_path = Path(destination)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, destination_path)
        typer.echo(f"Copied {source_path} to {destination_path}")


@app.command()
def clean(
    doc_links_config: Path = typer.Argument(..., help="Path to the doc links config"),
    absolute_paths: bool = typer.Option(
        False, help="Whether to use absolute paths for the destination files"
    ),
):
    """Delete destination files specified in the doc links config.

    This function reads the doc links configuration file and deletes the specified
    destination files. It can handle both relative and absolute paths based on the
    `absolute_paths` parameter.

    Args:
        doc_links_config (Path): Path to the doc links configuration file.
        absolute_paths (bool): If True, use absolute paths for destination files.
            If False, paths are relative to OUMI_ROOT.
    """
    if not doc_links_config.exists():
        typer.echo(f"Warning: {doc_links_config} not found. Skipping cleanup.")
        return

    for _, destination in read_doc_links(doc_links_config):
        if not absolute_paths:
            destination_path = OUMI_ROOT / destination
        else:
            destination_path = Path(destination)

        if destination_path.exists():
            destination_path.unlink()
            typer.echo(f"Deleted {destination_path}")
        else:
            typer.echo(f"Warning: {destination_path} not found. Skipping deletion.")


if __name__ == "__main__":
    app()
