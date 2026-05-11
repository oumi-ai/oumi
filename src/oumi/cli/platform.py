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

"""``oumi platform ...`` subcommand: talk to the Oumi Enterprise platform.

Provides a small ergonomic surface over :mod:`oumi.platform.client`:

* ``oumi platform login`` / ``logout`` / ``whoami``
* ``oumi platform datasets {list, pull}``
* ``oumi platform models {list, pull}``
* ``oumi platform judges {list}``
* ``oumi platform operations {status}``
"""

from pathlib import Path
from typing import Annotated, Any

import typer

from oumi.cli.cli_utils import CONSOLE

_DEFAULT_TIMEOUT_SECONDS = 600.0


# ----------------------------------------------------------------- helpers ---


def _client():
    """Build a :class:`oumi.platform.Client`; surface clear errors as Typer exits."""
    from oumi.platform import (
        Client,
        CredentialsNotFoundError,
        PlatformError,
    )

    try:
        return Client()
    except CredentialsNotFoundError as exc:
        CONSOLE.print(f"[red]Not logged in:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except PlatformError as exc:
        CONSOLE.print(f"[red]Platform error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    from rich.table import Table

    table = Table(show_lines=False, show_edge=False)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(row.get(c, "")) for c in columns))
    CONSOLE.print(table)


# ----------------------------------------------------------------- auth ---


def login(
    api_url: Annotated[
        str | None,
        typer.Option(
            "--api-url",
            help=(
                "Platform base URL. Defaults to https://api.oumi.ai if "
                "neither --api-url nor OUMI_API_URL is set."
            ),
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help=(
                "Platform API key. If omitted, you'll be prompted "
                "interactively. The value is hidden as you type."
            ),
            prompt=False,
        ),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Default project id to use for subsequent commands.",
        ),
    ] = None,
):
    """Persist Oumi Enterprise credentials to ~/.config/oumi/credentials.json.

    Args:
        api_url: Platform base URL.
        api_key: Platform API key. Prompted if omitted.
        project_id: Default project id stored alongside the key.
    """
    from oumi.platform import Credentials, save_credentials

    resolved_key = api_key or typer.prompt("API key", hide_input=True)
    resolved_url = api_url or typer.prompt(
        "API URL", default="https://api.oumi.ai", show_default=True
    )

    creds = Credentials(
        api_url=resolved_url.rstrip("/"),
        api_key=resolved_key,
        project_id=project_id,
    )
    path = save_credentials(creds)
    CONSOLE.print(
        f"[green]Saved credentials to {path}[/green] (api_url={creds.api_url})."
    )


def logout(
    credentials_path: Annotated[
        Path | None,
        typer.Option("--path", help="Override the credentials file path."),
    ] = None,
):
    """Delete the on-disk credentials file (env vars are not touched).

    Args:
        credentials_path: Override the file path. Defaults to
            ``~/.config/oumi/credentials.json``.
    """
    from oumi.platform.credentials import default_credentials_path

    path = credentials_path or default_credentials_path()
    if path.exists():
        path.unlink()
        CONSOLE.print(f"[green]Removed credentials file at {path}.[/green]")
    else:
        CONSOLE.print(f"[yellow]No credentials file at {path}.[/yellow]")


def whoami():
    """Show which platform and project the current credentials point at."""
    from oumi.platform import load_credentials

    creds = load_credentials()
    CONSOLE.print(f"api_url:    {creds.api_url}")
    CONSOLE.print(f"project_id: {creds.project_id or '(none)'}")
    suffix = creds.api_key[-4:] if len(creds.api_key) >= 4 else ""
    CONSOLE.print(f"api_key:    ****{suffix}")


# ----------------------------------------------------------------- datasets ---


def list_datasets(
    project: Annotated[
        str | None,
        typer.Option("--project", help="Override default project id."),
    ] = None,
):
    """List the datasets in your project.

    Args:
        project: Override the default project id.
    """
    client = _client()
    page = client.datasets.list(project_id=project)
    rows = _items_field(page)
    _print_table(
        rows,
        columns=["id", "displayName", "schemaType", "version", "versionName"],
    )


def pull_dataset(
    dataset_id: Annotated[str, typer.Argument(help="The dataset id to pull.")],
    destination: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Where to write the file. Defaults to ./<id>.jsonl."
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
):
    """Download a dataset to a local file.

    Args:
        dataset_id: The platform's dataset id.
        destination: Output path. Defaults to ``./<id>.jsonl``.
        project: Override the default project id.
    """
    client = _client()
    target = destination or Path(f"./{dataset_id}.jsonl")
    result = client.datasets.download(dataset_id, target, project_id=project)
    CONSOLE.print(f"[green]Downloaded[/green] {dataset_id} -> {result}")


def push_dataset(
    source: Annotated[
        Path, typer.Argument(help="Path to the local file to upload.")
    ],
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Display name for the new dataset. Defaults to the file's basename.",
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--detach",
            help="Block until the platform finishes ingestion.",
        ),
    ] = True,
):
    """Upload a local dataset file to the platform.

    Args:
        source: Path to the local file to upload.
        name: Display name for the new dataset.
        project: Override the default project id.
        wait: When set, block until ingestion completes.
    """
    client = _client()
    if not source.is_file():
        CONSOLE.print(f"[red]No such file:[/red] {source}")
        raise typer.Exit(code=2)
    response = client.datasets.upload(
        source, display_name=name, project_id=project, wait=wait
    )
    op = response.get("operation", {}) if isinstance(response, dict) else {}
    CONSOLE.print(
        f"[green]Uploaded[/green] {source} (operation id: {op.get('id')})."
    )


# ----------------------------------------------------------------- models ---


def list_models_(
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
):
    """List models in your project.

    Args:
        project: Override the default project id.
    """
    client = _client()
    page = client.models.list(project_id=project)
    rows = _items_field(page)
    _print_table(
        rows, columns=["id", "displayName", "version", "versionName", "latest"]
    )


def pull_model(
    model_id: Annotated[str, typer.Argument(help="The model id to pull.")],
    destination: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Destination directory. Defaults to ./<id>.",
        ),
    ] = None,
    version_id: Annotated[
        str | None,
        typer.Option(
            "--version",
            help="Pin a version. Defaults to the latest version.",
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
):
    """Download a model checkpoint into a local directory.

    Args:
        model_id: The platform's model id.
        destination: Local directory. Defaults to ``./<id>``.
        version_id: Pin a model version; defaults to latest.
        project: Override the default project id.
    """
    client = _client()
    target = destination or Path(f"./{model_id}")
    result = client.models.download(
        model_id, target, version_id=version_id, project_id=project
    )
    CONSOLE.print(f"[green]Downloaded[/green] {model_id} -> {result}")


# ----------------------------------------------------------------- judges ---


def list_judges(
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
):
    """List judges (evaluators of type ``judge``) in your project.

    Args:
        project: Override the default project id.
    """
    client = _client()
    page = client.evaluators.list(
        evaluator_type="judge", project_id=project
    )
    rows = _items_field(page)
    _print_table(
        rows,
        columns=["id", "displayName", "evaluatorType", "version", "versionName"],
    )


# ----------------------------------------------------------------- ops ---


def operation_status(
    operation_id: Annotated[
        str, typer.Argument(help="The platform operation id to inspect.")
    ],
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help="Block until the operation reaches a terminal state.",
        ),
    ] = False,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="When --wait is set, give up after this many seconds.",
        ),
    ] = _DEFAULT_TIMEOUT_SECONDS,
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
):
    """Print the platform operation's current status (optionally waiting).

    Args:
        operation_id: The operation id.
        wait: When set, poll until the operation reaches a terminal state.
        timeout: Maximum seconds to wait when ``--wait`` is set.
        project: Override the default project id.
    """
    client = _client()
    if wait:
        op = client.operations.wait(
            operation_id, project_id=project, timeout=timeout
        )
    else:
        op = client.operations.get(operation_id, project_id=project)
    CONSOLE.print(
        f"id={op.get('id')}  status={op.get('status')}  done={op.get('done')}"
    )
    if op.get("type"):
        CONSOLE.print(f"type:    {op.get('type')}")
    if op.get("error"):
        CONSOLE.print(f"[red]error:[/red]   {op.get('error')}")
    if op.get("result"):
        CONSOLE.print(f"result:  {op.get('result')}")


def operation_stop(
    operation_id: Annotated[
        str, typer.Argument(help="The platform operation id to cancel.")
    ],
    project: Annotated[
        str | None, typer.Option("--project", help="Override default project id.")
    ] = None,
):
    """Request cancellation of an in-flight platform operation.

    Args:
        operation_id: The operation id.
        project: Override the default project id.
    """
    client = _client()
    op = client.operations.stop(operation_id, project_id=project)
    CONSOLE.print(
        f"[green]Cancel requested.[/green] status={op.get('status')}"
    )


# ----------------------------------------------------------------- helpers ---


def _items_field(page: Any) -> list[dict[str, Any]]:
    """Extract the array-of-items field from a list-page response.

    Platform list endpoints wrap results in ``{<resource>s: [...]}`` (e.g.
    ``{"datasets": [...]}``). We accept either shape and a plain list.
    """
    if isinstance(page, list):
        return page
    if isinstance(page, dict):
        for value in page.values():
            if isinstance(value, list):
                return value
    return []
