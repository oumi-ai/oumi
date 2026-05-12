# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Rich-based renderer for :class:`AgentSession` events.

The renderer is a thin event consumer — it never reaches into the
session. The session emits events through the ``on_event`` hook (see
``agents.agent_session.EventKind``); this module turns each event into
panels, spinners, and rules on a Rich console. It also doubles as a
JSON-lines transcript writer when ``transcript_path`` is set, so a
single hook drives both terminal rendering and machine-readable logs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table

_ENV_PALETTE = ["cyan", "blue", "magenta", "yellow", "green"]

# Chars of a tool result rendered inline; the full payload still lives
# in the conversation log.
_TOOL_RESULT_PREVIEW_CHARS = 600


class AgentTUI:
    """Stateful Rich renderer for agent events."""

    def __init__(
        self,
        console: Console,
        *,
        show_tool_calls: bool = True,
        transcript_path: str | Path | None = None,
        env_color_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the renderer; optionally tee events to a JSONL transcript."""
        self._console = console
        self._show_tool_calls = show_tool_calls
        self._env_color_map = env_color_map or {}
        self._transcript: TextIO | None = (
            open(transcript_path, "a", encoding="utf-8")
            if transcript_path is not None
            else None
        )
        self._live: Live | None = None
        self._turn_started_at: float | None = None
        self._last_usage: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Event dispatch                                                     #
    # ------------------------------------------------------------------ #

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        """Dispatch ``kind`` to ``_on_<kind>`` and tee the event to the transcript."""
        # Always record to transcript first — even if rendering throws,
        # the log is intact.
        if self._transcript is not None:
            record = {"ts": time.time(), "kind": kind, **data}
            self._transcript.write(json.dumps(record, default=str) + "\n")
            self._transcript.flush()

        try:
            handler = getattr(self, f"_on_{kind}", None)
            if handler is not None:
                handler(data)
        except Exception as e:
            # A broken renderer must not crash the chat loop.
            self._stop_spinner()
            self._console.print(f"[red]TUI render error:[/red] {e}")

    # ------------------------------------------------------------------ #
    # Per-event handlers                                                 #
    # ------------------------------------------------------------------ #

    def _on_user(self, _data: dict[str, Any]) -> None:
        self._turn_started_at = time.monotonic()

    def _on_inference_start(self, _data: dict[str, Any]) -> None:
        self._start_spinner("thinking…")

    def _on_inference_end(self, data: dict[str, Any]) -> None:
        self._stop_spinner()
        usage = data.get("usage")
        self._last_usage = usage if isinstance(usage, dict) else None

    def _on_tool_call(self, data: dict[str, Any]) -> None:
        if not self._show_tool_calls:
            return
        try:
            args_pretty = json.dumps(json.loads(data["arguments"] or "{}"), indent=2)
        except json.JSONDecodeError:
            args_pretty = data["arguments"]
        color = self._color_for_tool(data["name"])
        body = Syntax(
            args_pretty, "json", theme="ansi_dark", background_color="default"
        )
        self._console.print(
            Panel(
                body,
                title=f"[{color}]→ {data['name']}[/{color}]",
                title_align="left",
                border_style=color,
                padding=(0, 1),
            )
        )
        self._start_spinner(f"running {data['name']}…")

    def _on_tool_result(self, data: dict[str, Any]) -> None:
        self._stop_spinner()
        if not self._show_tool_calls:
            return
        is_err = bool(data.get("is_error"))
        duration = self._format_duration(data.get("duration_ms"))
        payload = data.get("payload")
        if is_err and self._is_compact_error(payload):
            assert isinstance(payload, dict)
            kind = payload.get("error", "error")
            msg = str(payload.get("message", "")).strip().splitlines()[0]
            self._console.print(
                f"[red]✗ {data['name']}[/red] · [bold]{kind}[/bold]"
                + (f" — {msg}" if msg else "")
                + (f" [dim]· {duration}[/dim]" if duration else "")
            )
            return
        style = "red" if is_err else "green"
        glyph = "✗" if is_err else "✓"
        try:
            body = json.dumps(payload, indent=2, default=str)
        except (TypeError, ValueError):
            body = repr(payload)
        if len(body) > _TOOL_RESULT_PREVIEW_CHARS:
            extra = len(body) - _TOOL_RESULT_PREVIEW_CHARS
            body = body[:_TOOL_RESULT_PREVIEW_CHARS] + f"\n… [{extra} more chars]"
        title = f"[{style}]{glyph} {data['name']}[/{style}]"
        if duration:
            title += f" [dim]· {duration}[/dim]"
        self._console.print(
            Panel(
                Syntax(body, "json", theme="ansi_dark", background_color="default"),
                title=title,
                title_align="left",
                border_style=style,
                padding=(0, 1),
            )
        )

    def _on_assistant_text(self, data: dict[str, Any]) -> None:
        text = data.get("text") or ""
        if not text.strip():
            return
        self._console.print(
            Panel(
                Markdown(text),
                title="[bold magenta]assistant[/bold magenta]",
                title_align="left",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    def _on_turn_end(self, data: dict[str, Any]) -> None:
        n = int(data.get("tool_calls", 0))
        if self._turn_started_at is None:
            return
        elapsed = time.monotonic() - self._turn_started_at
        self._turn_started_at = None

        parts: list[str] = []
        parts.append(f"{n} tool call{'s' if n != 1 else ''}" if n else "no tools")
        parts.append(f"{elapsed:.1f}s")
        if self._last_usage is not None:
            in_t = int(self._last_usage.get("prompt_tokens", 0))
            out_t = int(self._last_usage.get("completion_tokens", 0))
            cached = int(self._last_usage.get("cached_tokens", 0))
            piece = f"{in_t}↑ {out_t}↓"
            if cached:
                piece += f" ({cached} cached)"
            parts.append(piece)
        session_total = int(data.get("session_total_tokens", 0))
        if session_total:
            parts.append(f"{session_total:,} session")
        self._console.print("[dim]· " + " · ".join(parts) + "[/dim]")
        self._last_usage = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _color_for_tool(self, tool_name: str) -> str:
        """Return the panel color for a tool's owning env."""
        return self._env_color_map.get(tool_name, _ENV_PALETTE[0])

    def _start_spinner(self, text: str) -> None:
        """Start or hot-swap the spinner text."""
        if self._live is not None:
            self._stop_spinner()
        live = Live(
            Spinner("dots", text=text),
            console=self._console,
            transient=True,
            refresh_per_second=10,
        )
        live.start()
        self._live = live

    def _stop_spinner(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    @staticmethod
    def _format_duration(duration_ms: Any) -> str:
        """Render ``duration_ms`` as ``12ms`` or ``1.4s`` for inline display."""
        if duration_ms is None:
            return ""
        ms = float(duration_ms)
        return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"

    @staticmethod
    def _is_compact_error(payload: Any) -> bool:
        """True for ``{status, error, message}`` shapes (router-layer rejections)."""
        if not isinstance(payload, dict):
            return False
        return set(payload) <= {"status", "error", "message"}

    def turn_separator(self) -> None:
        """Render a horizontal rule between user turns. Called by the CLI."""
        self._console.print(Rule(style="dim"))

    def close(self) -> None:
        """Stop any active spinner and close the transcript file."""
        self._stop_spinner()
        if self._transcript is not None:
            self._transcript.close()
            self._transcript = None


def build_env_color_map(env_config) -> dict[str, str]:  # noqa: ANN001
    """Map ``tool_id -> color`` so each env gets a distinct panel color.

    Single-env configs collapse to a single color; multi-env configs cycle
    through ``_ENV_PALETTE``. Used by :meth:`AgentTUI._color_for_tool`.
    """
    color_for_env = {
        env.id: _ENV_PALETTE[i % len(_ENV_PALETTE)]
        for i, env in enumerate(env_config.environments)
    }
    return {
        tool.id: color_for_env[env.id]
        for env in env_config.environments
        for tool in env.tools
    }


def render_header(console: Console, parsed) -> None:  # noqa: ANN001
    """Print the session header — model, engine, envs, tools."""
    inference = parsed.inference
    env_config = parsed.environment

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("model", inference.model.model_name or "—")
    table.add_row("engine", str(inference.engine or "default"))
    table.add_row(
        "envs",
        ", ".join(f"{e.id} ({e.env_type})" for e in env_config.environments) or "—",
    )
    tool_ids = [t.id for t in env_config.all_tools]
    tool_summary = f"{len(tool_ids)} · " + ", ".join(tool_ids) if tool_ids else "—"
    table.add_row("tools", tool_summary)

    console.print(
        Panel(
            table,
            title="[bold green]oumi agent[/bold green]",
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )
    )
    console.print("[dim]/help · /tools · /clear · /exit[/dim]\n")


def render_tools_table(console: Console, parsed) -> None:  # noqa: ANN001
    """Print the ``/tools`` slash command output."""
    table = Table(
        title="Registered tools",
        show_lines=False,
        title_style="bold",
        title_justify="left",
    )
    table.add_column("env", style="dim")
    table.add_column("tool", style="cyan")
    table.add_column("read-only", justify="center")
    table.add_column("description", overflow="fold")

    for env in parsed.environment.environments:
        for tool in env.tools:
            table.add_row(
                env.id,
                tool.id,
                "✓" if getattr(tool, "read_only", True) else "[yellow]write[/yellow]",
                tool.description,
            )
    console.print(table)
