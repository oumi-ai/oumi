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

"""CLI command for listing installed oumi plugins."""

import typer
from rich.table import Table

from oumi.cli.cli_utils import CONSOLE
from oumi.plugins.discovery import discover_plugins


def plugins():
    """List installed oumi plugins."""
    discovered = discover_plugins()

    if not discovered:
        CONSOLE.print("No plugins installed.")
        raise typer.Exit

    table = Table(title="Installed Oumi Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Package")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("CLI")
    table.add_column("Registry Modules")

    error_details: list[tuple[str, str]] = []

    for info in discovered:
        if info.error:
            status = "[red]error[/red]"
            error_details.append((info.entry_point_name, info.error))
        else:
            status = "[green]ok[/green]"

        cli_indicator = "\u2713" if info.register_cli_fn else ""
        registry = ", ".join(info.registry_modules) if info.registry_modules else ""

        table.add_row(
            info.entry_point_name,
            info.package_name or "?",
            info.package_version or "?",
            status,
            cli_indicator,
            registry,
        )

    CONSOLE.print(table)

    if error_details:
        CONSOLE.print()
        for name, err in error_details:
            CONSOLE.print(f"  [red]Error in '{name}':[/red] {err}")

    CONSOLE.print(f"\n{len(discovered)} plugin(s) installed.")
