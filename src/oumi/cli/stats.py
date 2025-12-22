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

"""CLI command for displaying usage statistics."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from oumi.cli.cli_utils import CONSOLE
from oumi.core.activity_tracker import Activity, ActivityTracker

# Heatmap characters (from less to more activity)
HEATMAP_CHARS = ["·", "░", "▒", "▓", "█"]
DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _format_duration(seconds: float | None) -> str:
    """Format duration in human-readable format."""
    if seconds is None:
        return "-"

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.0f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        if minutes > 0:
            return f"{hours:.0f}h {minutes:.0f}m"
        return f"{hours:.0f}h"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        return f"{days:.0f}d {hours:.0f}h"


def _format_number(n: int | float) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def _generate_heatmap(activities: list[Activity], weeks: int = 12) -> str:
    """Generate a GitHub-style activity heatmap.

    Args:
        activities: List of activities to visualize
        weeks: Number of weeks to show

    Returns:
        String representation of the heatmap
    """
    # Count activities per day
    day_counts: dict[str, int] = defaultdict(int)
    for activity in activities:
        try:
            dt = datetime.fromisoformat(activity.timestamp.replace("Z", "+00:00"))
            day_key = dt.strftime("%Y-%m-%d")
            day_counts[day_key] += 1
        except (ValueError, AttributeError):
            continue

    # Calculate date range
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(weeks=weeks, days=today.weekday())

    # Find max count for normalization
    max_count = max(day_counts.values()) if day_counts else 1

    # Build month labels
    months: list[str] = []
    current_month = ""
    for week in range(weeks):
        week_start = start_date + timedelta(weeks=week)
        month = week_start.strftime("%b")
        if month != current_month:
            months.append(month)
            current_month = month
        else:
            months.append("   ")

    # Build the heatmap grid
    lines = []

    # Month header
    month_line = "     " + " ".join(months[:weeks])
    lines.append(f"[dim]{month_line}[/dim]")

    # Day rows (Mon, Wed, Fri for compactness)
    for day_idx in [0, 2, 4]:  # Mon, Wed, Fri
        row = f"{DAYS_OF_WEEK[day_idx]:>3}  "
        for week in range(weeks):
            date = start_date + timedelta(weeks=week, days=day_idx)
            day_key = date.strftime("%Y-%m-%d")
            count = day_counts.get(day_key, 0)

            if date > today:
                char = " "
            elif count == 0:
                char = HEATMAP_CHARS[0]
            else:
                # Normalize to 1-4 range
                level = min(4, max(1, int((count / max_count) * 4)))
                char = HEATMAP_CHARS[level]

            # Color based on intensity
            if char == HEATMAP_CHARS[0]:
                row += f"[dim]{char}[/dim]  "
            elif char in HEATMAP_CHARS[1:3]:
                row += f"[yellow]{char}[/yellow]  "
            else:
                row += f"[bright_yellow]{char}[/bright_yellow]  "

        lines.append(row)

    # Legend
    legend = "     Less " + " ".join(
        f"[yellow]{c}[/yellow]" if i > 0 else f"[dim]{c}[/dim]"
        for i, c in enumerate(HEATMAP_CHARS)
    ) + " More"
    lines.append("")
    lines.append(legend)

    return "\n".join(lines)


def _calculate_streaks(activities: list[Activity]) -> tuple[int, int]:
    """Calculate current and longest activity streaks.

    Returns:
        Tuple of (current_streak, longest_streak) in days
    """
    if not activities:
        return 0, 0

    # Get unique active days
    active_days: set[str] = set()
    for activity in activities:
        try:
            dt = datetime.fromisoformat(activity.timestamp.replace("Z", "+00:00"))
            active_days.add(dt.strftime("%Y-%m-%d"))
        except (ValueError, AttributeError):
            continue

    if not active_days:
        return 0, 0

    # Sort days
    sorted_days = sorted(active_days)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    # Calculate current streak (must include today or yesterday)
    current_streak = 0
    if today in active_days or yesterday in active_days:
        check_date = datetime.now(timezone.utc).date()
        if today not in active_days:
            check_date = check_date - timedelta(days=1)

        while check_date.strftime("%Y-%m-%d") in active_days:
            current_streak += 1
            check_date = check_date - timedelta(days=1)

    # Calculate longest streak
    longest_streak = 0
    current_run = 1

    for i in range(1, len(sorted_days)):
        prev_date = datetime.strptime(sorted_days[i - 1], "%Y-%m-%d").date()
        curr_date = datetime.strptime(sorted_days[i], "%Y-%m-%d").date()

        if (curr_date - prev_date).days == 1:
            current_run += 1
        else:
            longest_streak = max(longest_streak, current_run)
            current_run = 1

    longest_streak = max(longest_streak, current_run)

    return current_streak, longest_streak


def _calculate_peak_hour(activities: list[Activity]) -> str:
    """Calculate the most active hour of the day.

    Returns:
        String like "14:00-15:00" or "-" if no data
    """
    if not activities:
        return "-"

    hour_counts: dict[int, int] = defaultdict(int)
    for activity in activities:
        try:
            dt = datetime.fromisoformat(activity.timestamp.replace("Z", "+00:00"))
            # Convert to local time
            local_dt = dt.astimezone()
            hour_counts[local_dt.hour] += 1
        except (ValueError, AttributeError):
            continue

    if not hour_counts:
        return "-"

    peak_hour = max(hour_counts, key=lambda h: hour_counts[h])
    return f"{peak_hour:02d}:00-{(peak_hour + 1) % 24:02d}:00"


def _get_favorite_value(
    activities: list[Activity], metadata_key: str
) -> str | None:
    """Get the most common value for a metadata key.

    Args:
        activities: List of activities
        metadata_key: Key to look up in metadata

    Returns:
        Most common value or None
    """
    counts: dict[str, int] = defaultdict(int)
    for activity in activities:
        value = activity.metadata.get(metadata_key)
        if value:
            counts[str(value)] += 1

    if not counts:
        return None

    return max(counts, key=lambda k: counts[k])


def _show_overview(days: int) -> None:
    """Show the main stats overview."""
    tracker = ActivityTracker()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    activities = tracker.get_activities(since=since)

    if not activities:
        CONSOLE.print(
            "\n[dim]No activity recorded yet. "
            "Run some Oumi commands to start tracking![/dim]\n"
        )
        return

    # Generate heatmap
    CONSOLE.print()
    CONSOLE.print("[bold]Activity[/bold]")
    CONSOLE.print(_generate_heatmap(activities, weeks=min(12, days // 7 + 1)))
    CONSOLE.print()

    # Calculate stats
    completed = [a for a in activities if a.status == "completed"]
    failed = [a for a in activities if a.status == "failed"]

    total_duration = sum(
        a.duration_seconds for a in activities if a.duration_seconds
    )
    longest_duration = max(
        (a.duration_seconds for a in activities if a.duration_seconds),
        default=0,
    )

    current_streak, longest_streak = _calculate_streaks(activities)
    peak_hour = _calculate_peak_hour(activities)

    # Count active days
    active_days = len(set(
        datetime.fromisoformat(a.timestamp.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        for a in activities
    ))

    # Group by command
    command_counts: dict[str, int] = defaultdict(int)
    for activity in activities:
        command_counts[activity.command] += 1

    # Build stats panels
    # Training stats
    train_activities = [a for a in activities if a.command == "train"]
    train_table = Table.grid(padding=(0, 2))
    train_table.add_column(style="dim")
    train_table.add_column(style="cyan")

    train_table.add_row("Total runs:", str(len(train_activities)))
    total_samples = sum(
        a.metadata.get("samples", 0) for a in train_activities
    )
    train_table.add_row("Samples trained:", _format_number(total_samples))
    train_time = sum(
        a.duration_seconds for a in train_activities if a.duration_seconds
    )
    train_table.add_row("Total time:", _format_duration(train_time))
    favorite_model = _get_favorite_value(train_activities, "model")
    train_table.add_row("Favorite model:", favorite_model or "-")

    train_panel = Panel(
        train_table,
        title="[bold]Training[/bold]",
        border_style="blue",
        expand=True,
    )

    # Inference stats
    infer_activities = [a for a in activities if a.command == "infer"]
    infer_table = Table.grid(padding=(0, 2))
    infer_table.add_column(style="dim")
    infer_table.add_column(style="cyan")

    infer_table.add_row("Sessions:", str(len(infer_activities)))
    total_tokens = sum(
        a.metadata.get("tokens_generated", 0) for a in infer_activities
    )
    infer_table.add_row("Tokens generated:", _format_number(total_tokens))
    infer_model = _get_favorite_value(infer_activities, "model")
    infer_table.add_row("Favorite model:", infer_model or "-")

    infer_panel = Panel(
        infer_table,
        title="[bold]Inference[/bold]",
        border_style="blue",
        expand=True,
    )

    # Evaluation stats
    eval_activities = [a for a in activities if a.command in ("evaluate", "eval")]
    eval_table = Table.grid(padding=(0, 2))
    eval_table.add_column(style="dim")
    eval_table.add_column(style="cyan")

    eval_table.add_row("Evaluations:", str(len(eval_activities)))
    eval_model = _get_favorite_value(eval_activities, "model")
    eval_table.add_row("Favorite model:", eval_model or "-")

    eval_panel = Panel(
        eval_table,
        title="[bold]Evaluation[/bold]",
        border_style="blue",
        expand=True,
    )

    # Data stats
    data_commands = ["analyze", "synth", "synthesize", "judge"]
    data_activities = [a for a in activities if a.command in data_commands]
    data_table = Table.grid(padding=(0, 2))
    data_table.add_column(style="dim")
    data_table.add_column(style="cyan")

    synth_count = len([a for a in data_activities if a.command in ("synth", "synthesize")])
    analyze_count = len([a for a in data_activities if a.command == "analyze"])
    judge_count = len([a for a in data_activities if a.command == "judge"])

    data_table.add_row("Datasets analyzed:", str(analyze_count))
    data_table.add_row("Synth runs:", str(synth_count))
    data_table.add_row("Judge runs:", str(judge_count))

    data_panel = Panel(
        data_table,
        title="[bold]Data[/bold]",
        border_style="blue",
        expand=True,
    )

    # Compute stats
    compute_commands = ["launch", "distributed"]
    compute_activities = [a for a in activities if a.command in compute_commands]
    compute_table = Table.grid(padding=(0, 2))
    compute_table.add_column(style="dim")
    compute_table.add_column(style="cyan")

    compute_table.add_row("Jobs launched:", str(len(compute_activities)))
    success_rate = (
        len([a for a in compute_activities if a.status == "completed"])
        / len(compute_activities)
        * 100
        if compute_activities
        else 0
    )
    compute_table.add_row("Success rate:", f"{success_rate:.0f}%")

    compute_panel = Panel(
        compute_table,
        title="[bold]Compute[/bold]",
        border_style="blue",
        expand=True,
    )

    # Overall stats
    overall_table = Table.grid(padding=(0, 2))
    overall_table.add_column(style="dim")
    overall_table.add_column(style="cyan")

    overall_table.add_row("Total commands:", str(len(activities)))
    overall_table.add_row("Completed:", f"{len(completed)} ({len(completed)*100//len(activities) if activities else 0}%)")
    overall_table.add_row("Failed:", str(len(failed)))
    overall_table.add_row("Total time:", _format_duration(total_duration))
    overall_table.add_row("Longest run:", _format_duration(longest_duration))

    overall_panel = Panel(
        overall_table,
        title="[bold]Overall[/bold]",
        border_style="green",
        expand=True,
    )

    # Print panels in a grid layout
    from rich.columns import Columns

    # Row 1: Training + Inference
    CONSOLE.print(Columns([train_panel, infer_panel], equal=True, expand=True))

    # Row 2: Evaluation + Data
    CONSOLE.print(Columns([eval_panel, data_panel], equal=True, expand=True))

    # Row 3: Compute + Overall
    CONSOLE.print(Columns([compute_panel, overall_panel], equal=True, expand=True))

    # Footer stats
    CONSOLE.print()
    footer = (
        f"[cyan]Current streak:[/cyan] {current_streak} days    "
        f"[cyan]Longest streak:[/cyan] {longest_streak} days    "
        f"[cyan]Active days:[/cyan] {active_days}/{days}    "
        f"[cyan]Peak hour:[/cyan] {peak_hour}"
    )
    CONSOLE.print(footer)
    CONSOLE.print(f"[dim]Stats from the last {days} days[/dim]")
    CONSOLE.print()


def _show_command_stats(command: str, days: int) -> None:
    """Show detailed stats for a specific command."""
    tracker = ActivityTracker()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    activities = tracker.get_activities(command=command, since=since)

    if not activities:
        CONSOLE.print(f"\n[dim]No {command} activity recorded in the last {days} days.[/dim]\n")
        return

    CONSOLE.print(f"\n[bold]{command.upper()} STATS[/bold]\n")

    # Summary
    completed = [a for a in activities if a.status == "completed"]
    failed = [a for a in activities if a.status == "failed"]
    total_duration = sum(a.duration_seconds for a in activities if a.duration_seconds)
    avg_duration = total_duration / len(activities) if activities else 0

    summary_table = Table.grid(padding=(0, 4))
    summary_table.add_column(style="dim")
    summary_table.add_column(style="cyan")
    summary_table.add_column(style="dim")
    summary_table.add_column(style="cyan")

    summary_table.add_row(
        "Total runs:", str(len(activities)),
        "Completed:", f"{len(completed)} ({len(completed)*100//len(activities)}%)",
    )
    summary_table.add_row(
        "Failed:", str(len(failed)),
        "Avg duration:", _format_duration(avg_duration),
    )
    summary_table.add_row(
        "Total time:", _format_duration(total_duration),
        "", "",
    )

    CONSOLE.print(Panel(summary_table, title="Summary", border_style="blue"))

    # Recent runs table
    CONSOLE.print("\n[bold]Recent Runs[/bold]\n")

    runs_table = Table(show_header=True, header_style="bold")
    runs_table.add_column("Date", style="dim")
    runs_table.add_column("Duration")
    runs_table.add_column("Status")

    # Add metadata columns based on command
    if command == "train":
        runs_table.add_column("Model")
        runs_table.add_column("Samples")
    elif command == "infer":
        runs_table.add_column("Model")
        runs_table.add_column("Tokens")
    elif command in ("evaluate", "eval"):
        runs_table.add_column("Model")
        runs_table.add_column("Benchmark")

    for activity in activities[:10]:
        try:
            dt = datetime.fromisoformat(activity.timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            date_str = "-"

        status_str = {
            "completed": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
            "cancelled": "[yellow]○[/yellow]",
            "running": "[blue]●[/blue]",
        }.get(activity.status, activity.status)

        row = [
            date_str,
            _format_duration(activity.duration_seconds),
            status_str,
        ]

        if command == "train":
            row.append(activity.metadata.get("model", "-")[:30])
            row.append(_format_number(activity.metadata.get("samples", 0)))
        elif command == "infer":
            row.append(activity.metadata.get("model", "-")[:30])
            row.append(_format_number(activity.metadata.get("tokens_generated", 0)))
        elif command in ("evaluate", "eval"):
            row.append(activity.metadata.get("model", "-")[:30])
            row.append(activity.metadata.get("benchmark", "-"))

        runs_table.add_row(*row)

    CONSOLE.print(runs_table)
    CONSOLE.print()


def stats(
    command: Annotated[
        str | None,
        typer.Argument(
            help="Filter by command (train, infer, evaluate, etc.)"
        ),
    ] = None,
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Number of days of history to show"),
    ] = 90,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Clear all activity history"),
    ] = False,
) -> None:
    """Show usage statistics across all Oumi commands.

    Examples:
        oumi stats              # Show overview
        oumi stats train        # Show training stats
        oumi stats --days 30    # Last 30 days only
        oumi stats --clear      # Clear history
    """
    if clear:
        tracker = ActivityTracker()
        count = tracker.clear()
        CONSOLE.print(f"[green]Cleared {count} activity records.[/green]")
        return

    if command:
        _show_command_stats(command, days)
    else:
        _show_overview(days)
