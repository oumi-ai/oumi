# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""``oumi agent`` CLI: REPL shell over :class:`AgentSession`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.utils.logging import logger

_HISTORY_PATH = str(Path("~/.oumi/agent_history").expanduser())


def _build_input_reader(console):  # noqa: ANN001
    """Return a ``read() -> str`` callable.

    Prefers ``prompt_toolkit`` for ↑/↓ history, ctrl-r search, and
    persistent history. Falls back to ``Console.input`` if the optional
    dep is missing — the agent CLI should not crash on a clean install.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return lambda: console.input("[bold cyan]you ›[/bold cyan] ")

    Path(_HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    pt_session = PromptSession(history=FileHistory(_HISTORY_PATH))

    def _read() -> str:
        return pt_session.prompt(HTML("<ansicyan><b>you ›</b></ansicyan> "))

    return _read


def _print_help(console) -> None:  # noqa: ANN001
    console.print(
        "[bold]commands[/bold]\n"
        "  [cyan]/help[/cyan]       show this help\n"
        "  [cyan]/tools[/cyan]      list registered tools\n"
        "  [cyan]/clear[/cyan]      clear the screen\n"
        "  [cyan]/exit[/cyan]       end the session (or Ctrl-D)\n"
    )


def agent(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to an AgentHarnessConfig YAML file.",
            rich_help_panel="Options",
        ),
    ],
    initial_message: Annotated[
        str | None,
        typer.Option(
            "--message",
            "-m",
            help=(
                "Send a single user message and exit. Skips the REPL — useful "
                "for scripting and one-shot agent runs."
            ),
            rich_help_panel="Options",
        ),
    ] = None,
    transcript: Annotated[
        str | None,
        typer.Option(
            "--transcript",
            help=(
                "Append a JSON-lines event log of the session (user messages, "
                "tool calls, tool results, assistant replies) to this file."
            ),
            rich_help_panel="Options",
        ),
    ] = None,
    show_tools: Annotated[
        bool,
        typer.Option(
            "--show-tools/--hide-tools",
            help=(
                "Render each tool call and result inline. Disable for clean "
                "demo runs that only show assistant replies."
            ),
            rich_help_panel="Options",
        ),
    ] = True,
    level: Annotated[
        cli_utils.LogLevel | None,
        typer.Option(
            "--log-level",
            "-log",
            help="Logging level.",
            show_default=False,
            show_choices=True,
            case_sensitive=False,
            callback=cli_utils.set_log_level,
            rich_help_panel="Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Print the full resolved config before starting the session.",
            rich_help_panel="Options",
        ),
    ] = False,
):
    """Start a tool-using agent chat session.

    Loads an :class:`AgentHarnessConfig`, builds the inference engine and
    every environment it references, flattens tools across environments,
    and starts a REPL. Each user line is sent through the loop until the
    LLM stops calling tools, then the assistant's reply is printed.

    Slash commands: ``/help``, ``/tools``, ``/clear``, ``/exit`` (or Ctrl-D).
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    option_overrides = cli_utils.collect_config_overrides(ctx)
    all_overrides = extra_args + option_overrides

    resolved = str(cli_utils.resolve_and_fetch_config(config))

    # Delayed imports — keep CLI startup snappy.
    from oumi.agents import AgentSession
    from oumi.cli.agent_tui import (
        AgentTUI,
        build_env_color_map,
        render_header,
        render_tools_table,
    )
    from oumi.core.configs import AgentHarnessConfig

    # ``ignore_interpolation=False`` resolves ``${oc.env:VAR,default}`` in
    # nested env_kwargs (e.g. DB paths). Diverges from the codebase default.
    parsed: AgentHarnessConfig = AgentHarnessConfig.from_yaml_and_arg_list(
        resolved, all_overrides, logger=logger, ignore_interpolation=False
    )
    parsed.finalize_and_validate()

    if verbose:
        parsed.print_config(logger)

    console = cli_utils.CONSOLE
    tui = AgentTUI(
        console,
        show_tool_calls=show_tools,
        transcript_path=transcript,
        env_color_map=build_env_color_map(parsed.environment),
    )
    render_header(console, parsed)

    try:
        with AgentSession.from_config(parsed) as session:
            session.set_event_handler(tui)

            if initial_message is not None:
                session.send(initial_message)
                return

            read_input = _build_input_reader(console)

            while True:
                try:
                    user_input = read_input()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                stripped = user_input.strip()
                if not stripped:
                    continue
                if stripped in {"/exit", "/quit"}:
                    break
                if stripped == "/help":
                    _print_help(console)
                    continue
                if stripped == "/tools":
                    render_tools_table(console, parsed)
                    continue
                if stripped == "/clear":
                    console.clear()
                    render_header(console, parsed)
                    continue

                tui.turn_separator()
                try:
                    session.send(user_input)
                except KeyboardInterrupt:
                    console.print("\n[yellow]· cancelled[/yellow]")
                    continue
                except Exception as e:
                    console.print(
                        f"[red]session error[/red] · "
                        f"[bold]{type(e).__name__}[/bold]: {e}"
                    )
                    continue
    finally:
        tui.close()
