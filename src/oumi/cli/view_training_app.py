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

"""Textual TUI application for viewing training output folders."""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Rule,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets.tree import TreeNode

# Try to import plotext for charts
try:
    from textual_plotext import PlotextPlot

    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class LogEntry:
    """Represents a single log entry from the training log."""

    timestamp: datetime
    level: str  # INFO, WARNING, ERROR
    source: str  # e.g., "train.py:165"
    message: str
    raw_line: str


@dataclass
class CheckpointInfo:
    """Information about a training checkpoint."""

    path: Path
    step: int
    size_bytes: int
    has_optimizer: bool
    has_model: bool


@dataclass
class TrainingData:
    """Consolidated training output data."""

    folder_path: Path

    # From trainer_state.json (required)
    trainer_state: dict
    log_history: list[dict]
    global_step: int
    max_steps: int
    epoch: float
    total_flos: float
    num_input_tokens_seen: int
    train_batch_size: int

    # Optional data
    telemetry: Optional[dict] = None
    metrics_summary: Optional[dict] = None
    training_config: Optional[dict] = None
    model_config: Optional[dict] = None

    # Parsed data
    log_entries: list[LogEntry] = field(default_factory=list)
    checkpoints: list[CheckpointInfo] = field(default_factory=list)

    # Computed
    is_complete: bool = False
    warnings_count: int = 0
    errors_count: int = 0


# =============================================================================
# Data Loading Functions
# =============================================================================


def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_yaml(path: Path) -> Optional[dict]:
    """Load a YAML file, returning None on error."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        return None


def _parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse a single log line into a LogEntry.

    Expected format:
    [2025-11-21 09:59:33,027][oumi][rank0][pid:30141][MainThread][INFO]][file.py:123] Message
    """
    # Pattern to match the log format
    pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\].*\[(INFO|WARNING|ERROR|DEBUG)\]\]\[([^\]]+)\] (.*)"
    match = re.match(pattern, line)
    if not match:
        return None

    timestamp_str, level, source, message = match.groups()
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        timestamp = datetime.now()

    return LogEntry(
        timestamp=timestamp,
        level=level,
        source=source,
        message=message.strip(),
        raw_line=line,
    )


def _parse_log_file(log_path: Path) -> list[LogEntry]:
    """Parse a training log file into LogEntry objects."""
    entries = []
    if not log_path.exists():
        return entries

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            current_entry = None
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                parsed = _parse_log_line(line)
                if parsed:
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = parsed
                elif current_entry:
                    # Continuation of previous message
                    current_entry.message += "\n" + line

            if current_entry:
                entries.append(current_entry)
    except Exception:
        pass

    return entries


def _discover_checkpoints(folder_path: Path) -> list[CheckpointInfo]:
    """Discover checkpoint directories in the training folder."""
    checkpoints = []

    for item in folder_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
            except (IndexError, ValueError):
                continue

            # Calculate size
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())

            # Check contents
            has_optimizer = (item / "optimizer.pt").exists()
            has_model = (item / "model.safetensors").exists() or (
                item / "pytorch_model.bin"
            ).exists()

            checkpoints.append(
                CheckpointInfo(
                    path=item,
                    step=step,
                    size_bytes=size,
                    has_optimizer=has_optimizer,
                    has_model=has_model,
                )
            )

    return sorted(checkpoints, key=lambda c: c.step)


def load_training_data(folder_path: Path) -> TrainingData:
    """Load all training data from a folder.

    Args:
        folder_path: Path to the training output folder.

    Returns:
        TrainingData object with all available data.

    Raises:
        FileNotFoundError: If trainer_state.json doesn't exist.
    """
    trainer_state_path = folder_path / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {folder_path}")

    trainer_state = _load_json(trainer_state_path)
    if trainer_state is None:
        raise ValueError(f"Failed to parse trainer_state.json in {folder_path}")

    # Extract core data from trainer_state
    log_history = trainer_state.get("log_history", [])
    global_step = trainer_state.get("global_step", 0)
    max_steps = trainer_state.get("max_steps", global_step)
    epoch = trainer_state.get("epoch", 0.0)
    total_flos = trainer_state.get("total_flos", 0.0)
    num_input_tokens_seen = trainer_state.get("num_input_tokens_seen", 0)
    train_batch_size = trainer_state.get("train_batch_size", 0)

    # Load optional files
    telemetry_dir = folder_path / "telemetry"
    telemetry = _load_json(telemetry_dir / "telemetry_callback_rank0000.json")
    metrics_summary = _load_json(telemetry_dir / "telemetry_callback_metrics_rank0000.json")
    training_config = _load_yaml(telemetry_dir / "training_config.yaml")
    model_config = _load_json(folder_path / "config.json")

    # Parse log file
    log_path = folder_path / "logs" / "rank_0000.log"
    log_entries = _parse_log_file(log_path)

    # Count warnings and errors
    warnings_count = sum(1 for e in log_entries if e.level == "WARNING")
    errors_count = sum(1 for e in log_entries if e.level == "ERROR")

    # Discover checkpoints
    checkpoints = _discover_checkpoints(folder_path)

    # Determine if training is complete
    is_complete = global_step >= max_steps if max_steps > 0 else True

    return TrainingData(
        folder_path=folder_path,
        trainer_state=trainer_state,
        log_history=log_history,
        global_step=global_step,
        max_steps=max_steps,
        epoch=epoch,
        total_flos=total_flos,
        num_input_tokens_seen=num_input_tokens_seen,
        train_batch_size=train_batch_size,
        telemetry=telemetry,
        metrics_summary=metrics_summary,
        training_config=training_config,
        model_config=model_config,
        log_entries=log_entries,
        checkpoints=checkpoints,
        is_complete=is_complete,
        warnings_count=warnings_count,
        errors_count=errors_count,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def format_size(size_bytes: int) -> str:
    """Format bytes into human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_number(n: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if abs(n) >= 1e12:
        return f"{n/1e12:.1f}T"
    if abs(n) >= 1e9:
        return f"{n/1e9:.1f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.2f}" if isinstance(n, float) else str(n)


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif sys.platform == "linux":
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(),
                check=True,
            )
        else:
            # Try pyperclip as fallback
            import pyperclip

            pyperclip.copy(text)
        return True
    except Exception:
        return False


# =============================================================================
# Help Screen
# =============================================================================


class HelpScreen(ModalScreen):
    """Help screen showing keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    HelpScreen .title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    HelpScreen .section {
        margin-top: 1;
        text-style: bold;
        color: $primary;
    }

    HelpScreen .keybinding {
        margin-left: 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Training Dashboard Help", classes="title")
            yield Rule()

            yield Static("Navigation", classes="section")
            yield Static("  1-5        Switch to tab", classes="keybinding")
            yield Static("  Tab        Next tab", classes="keybinding")
            yield Static("  Shift+Tab  Previous tab", classes="keybinding")
            yield Static("  j/k/↑/↓    Scroll up/down", classes="keybinding")
            yield Static("  PgUp/PgDn  Page up/down", classes="keybinding")
            yield Static("  g/G        Go to top/bottom", classes="keybinding")

            yield Static("Search & Filter", classes="section")
            yield Static("  /          Search", classes="keybinding")
            yield Static("  Escape     Clear search", classes="keybinding")
            yield Static("  w          Warnings only (Logs)", classes="keybinding")
            yield Static("  e          Errors only (Logs)", classes="keybinding")

            yield Static("Actions", classes="section")
            yield Static("  y          Copy current item", classes="keybinding")
            yield Static("  t          Copy TensorBoard command", classes="keybinding")
            yield Static("  o          Open folder", classes="keybinding")

            yield Static("General", classes="section")
            yield Static("  ?          Show this help", classes="keybinding")
            yield Static("  q/Esc      Quit", classes="keybinding")

            yield Rule()
            yield Static("Press Escape or q to close", classes="keybinding")


# =============================================================================
# Panel Components
# =============================================================================


class SummaryPanel(VerticalScroll):
    """Summary panel showing training overview."""

    DEFAULT_CSS = """
    SummaryPanel {
        padding: 1 2;
        height: 1fr;
    }

    SummaryPanel .status-banner {
        width: 100%;
        padding: 1 2;
        margin-bottom: 1;
        text-align: center;
    }

    SummaryPanel .status-complete {
        background: $success 30%;
        border: solid $success;
    }

    SummaryPanel .status-incomplete {
        background: $warning 30%;
        border: solid $warning;
    }

    SummaryPanel .section-title {
        text-style: bold;
        margin-top: 1;
        color: $primary;
    }

    SummaryPanel .card {
        background: $surface;
        padding: 1;
        margin: 1 0;
        border: solid $primary-background;
        height: auto;
    }

    SummaryPanel .metric-row {
        height: auto;
    }

    SummaryPanel .metric-label {
        width: 20;
        text-style: bold;
    }

    SummaryPanel .metric-value {
        width: auto;
    }

    SummaryPanel .metric-sparkline {
        height: 1;
        margin: 0 0 1 20;
    }

    SummaryPanel .timeline-card {
        padding: 0 1;
    }

    SummaryPanel .timeline-event {
        height: 1;
    }

    SummaryPanel .timeline-step {
        height: 1;
        margin-left: 2;
    }

    SummaryPanel .issue-warning {
        color: $warning;
    }

    SummaryPanel .issue-ok {
        color: $success;
    }
    """

    def __init__(self, data: TrainingData, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def compose(self) -> ComposeResult:
        # Status banner
        if self.data.is_complete:
            yield Static(
                f"[bold green]TRAINING COMPLETE[/bold green]\n"
                f"{self.data.global_step}/{self.data.max_steps} steps",
                classes="status-banner status-complete",
            )
        else:
            yield Static(
                f"[bold yellow]TRAINING IN PROGRESS[/bold yellow]\n"
                f"{self.data.global_step}/{self.data.max_steps} steps",
                classes="status-banner status-incomplete",
            )

        # Model info
        yield Static("MODEL", classes="section-title")
        with Container(classes="card"):
            model_name = "Unknown"
            if self.data.training_config:
                model_name = self.data.training_config.get("model", {}).get(
                    "model_name", "Unknown"
                )
            elif self.data.model_config:
                model_name = self.data.model_config.get("_name_or_path", "Unknown")

            yield Horizontal(
                Static("Name:", classes="metric-label"),
                Static(model_name, classes="metric-value"),
                classes="metric-row",
            )

            if self.data.model_config:
                arch = self.data.model_config.get("architectures", ["Unknown"])[0]
                yield Horizontal(
                    Static("Architecture:", classes="metric-label"),
                    Static(arch, classes="metric-value"),
                    classes="metric-row",
                )

        # Training info
        yield Static("TRAINING", classes="section-title")
        with Container(classes="card"):
            yield Horizontal(
                Static("Steps:", classes="metric-label"),
                Static(
                    f"{self.data.global_step} / {self.data.max_steps}",
                    classes="metric-value",
                ),
                classes="metric-row",
            )
            yield Horizontal(
                Static("Tokens:", classes="metric-label"),
                Static(
                    format_number(self.data.num_input_tokens_seen),
                    classes="metric-value",
                ),
                classes="metric-row",
            )
            yield Horizontal(
                Static("Epoch:", classes="metric-label"),
                Static(f"{self.data.epoch:.4f}", classes="metric-value"),
                classes="metric-row",
            )
            if self.data.metrics_summary:
                runtime = self.data.metrics_summary.get("train_runtime", 0)
                yield Horizontal(
                    Static("Runtime:", classes="metric-label"),
                    Static(f"{runtime:.1f}s", classes="metric-value"),
                    classes="metric-row",
                )

        # Key metrics with sparklines
        yield Static("KEY METRICS", classes="section-title")
        with Container(classes="card"):
            # Extract metrics from log_history
            losses = [
                e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e
            ]
            if losses:
                change = ((losses[-1] - losses[0]) / losses[0] * 100) if losses[0] != 0 else 0
                direction = "↓" if change < 0 else "↑"
                color = "green" if change < 0 else "yellow"
                yield Horizontal(
                    Static("Loss:", classes="metric-label"),
                    Static(
                        f"[{color}]{losses[-1]:.4f}[/{color}] ({direction} {abs(change):.1f}%)",
                        classes="metric-value",
                    ),
                    classes="metric-row",
                )
                # Normalize and expand data for better sparkline visualization
                scaled_losses = self._scale_for_sparkline(losses)
                yield Sparkline(scaled_losses, classes="metric-sparkline")

            accuracies = [
                e["mean_token_accuracy"]
                for e in self.data.log_history
                if "mean_token_accuracy" in e
            ]
            if accuracies:
                change = (
                    ((accuracies[-1] - accuracies[0]) / accuracies[0] * 100)
                    if accuracies[0] != 0
                    else 0
                )
                direction = "↑" if change > 0 else "↓"
                color = "green" if change > 0 else "yellow"
                yield Horizontal(
                    Static("Accuracy:", classes="metric-label"),
                    Static(
                        f"[{color}]{accuracies[-1]*100:.1f}%[/{color}] ({direction} {abs(change):.1f}%)",
                        classes="metric-value",
                    ),
                    classes="metric-row",
                )
                scaled_accs = self._scale_for_sparkline(accuracies)
                yield Sparkline(scaled_accs, classes="metric-sparkline")

            # Throughput
            tps_values = [
                e["train_tokens_per_second"]
                for e in self.data.log_history
                if "train_tokens_per_second" in e
            ]
            if tps_values:
                avg_tps = sum(tps_values) / len(tps_values)
                yield Horizontal(
                    Static("Throughput:", classes="metric-label"),
                    Static(f"[cyan]{avg_tps:,.0f}[/cyan] tok/s avg", classes="metric-value"),
                    classes="metric-row",
                )

        # Timeline (compact version)
        yield Static("TIMELINE", classes="section-title")
        with Container(classes="card timeline-card"):
            yield from self._compose_compact_timeline()

        # Issues
        yield Static("ISSUES", classes="section-title")
        with Container(classes="card"):
            if self.data.warnings_count > 0:
                yield Button(
                    f"⚠ {self.data.warnings_count} warning(s) → View",
                    id="btn-warnings",
                    variant="warning",
                )
            else:
                yield Static("[green]✓[/green] No warnings", classes="issue-ok")

            if self.data.errors_count > 0:
                yield Button(
                    f"✗ {self.data.errors_count} error(s) → View",
                    id="btn-errors",
                    variant="error",
                )
            else:
                yield Static("[green]✓[/green] No errors", classes="issue-ok")

    def _scale_for_sparkline(self, values: list[float], min_points: int = 20) -> list[float]:
        """Scale and interpolate values for better sparkline visualization.

        - Normalizes to 0-1 range
        - Interpolates to have at least min_points for smooth display
        """
        if not values:
            return []

        # Normalize to 0-1 range
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            # Add small variation to show something
            return [0.5] * max(len(values), min_points)

        normalized = [(v - min_val) / (max_val - min_val) for v in values]

        # Interpolate if we have too few points
        if len(normalized) < min_points:
            result = []
            for i in range(len(normalized) - 1):
                steps = min_points // (len(normalized) - 1)
                for j in range(steps):
                    t = j / steps
                    result.append(normalized[i] * (1 - t) + normalized[i + 1] * t)
            result.append(normalized[-1])
            return result

        return normalized

    def _compose_compact_timeline(self) -> ComposeResult:
        """Compose a compact timeline for the summary view."""
        # Get timeline phases from TimelinePanel logic
        phases = self._get_timeline_phases()

        prev_time = None
        for phase in phases:
            time_str = phase["time"].strftime("%H:%M:%S") if phase["time"] else ""

            # Duration since previous
            duration_str = ""
            if prev_time and phase["time"]:
                delta = (phase["time"] - prev_time).total_seconds()
                if delta >= 60:
                    duration_str = f"+{delta/60:.1f}m"
                elif delta >= 1:
                    duration_str = f"+{delta:.0f}s"

            info = phase.get("info", "")
            line = f"[dim]{time_str}[/dim] [bold cyan]●[/bold cyan] {phase['title']}"
            if duration_str:
                line += f" [dim]{duration_str}[/dim]"
            if info:
                line += f" [dim]({info})[/dim]"

            yield Static(line, classes="timeline-event")

            # Show step metrics inline for training
            if phase.get("steps"):
                for step in phase["steps"]:
                    yield Static(f"  [green]{step}[/green]", classes="timeline-step")

            prev_time = phase["time"]

    def _get_timeline_phases(self) -> list[dict]:
        """Get timeline phases (reusing TimelinePanel logic)."""
        import re

        all_phases = []
        training_start_indices = []
        dataset_count = 0

        for i, entry in enumerate(self.data.log_entries):
            msg = entry.message.lower()
            msg_orig = entry.message

            if "resolved" in msg and "dataloader" in msg:
                match = re.search(r"to\s*['\"]?(?:[\w.]*=)?(\d+)['\"]?", msg_orig)
                info = f"workers={match.group(1)}" if match else ""
                all_phases.append({
                    "time": entry.timestamp,
                    "title": "INIT",
                    "info": info,
                })
                dataset_count = 0
            elif "creating map dataset" in msg:
                dataset_count += 1
                ds_type = "Train" if dataset_count == 1 else "Eval"
                name_match = re.search(r"dataset_name:\s*['\"]?([^'\"]+)['\"]?", msg_orig)
                ds_name = name_match.group(1).split("/")[-1] if name_match else ""
                all_phases.append({
                    "time": entry.timestamp,
                    "title": f"DATASET",
                    "info": ds_name[:20] if ds_name else ds_type,
                })
            elif "model parameters summary" in msg:
                all_phases.append({
                    "time": entry.timestamp,
                    "title": "MODEL",
                    "info": "",
                })
            elif "starting training" in msg:
                training_start_indices.append(len(all_phases))
                all_phases.append({
                    "time": entry.timestamp,
                    "title": "TRAINING",
                    "info": f"{self.data.max_steps} steps",
                    "steps": self._get_step_summaries(),
                })
            elif "training is complete" in msg:
                losses = [e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
                info = f"loss={losses[-1]:.3f}" if losses else ""
                all_phases.append({
                    "time": entry.timestamp,
                    "title": "COMPLETE",
                    "info": info,
                })

        # Return only the most recent run
        if training_start_indices:
            last_idx = training_start_indices[-1]
            for i in range(last_idx - 1, -1, -1):
                if all_phases[i]["title"] == "INIT":
                    return all_phases[i:]
        return all_phases

    def _get_step_summaries(self) -> list[str]:
        """Get formatted step summaries."""
        summaries = []
        max_steps = self.data.max_steps
        for entry in self.data.log_history:
            if "loss" in entry and "train_loss" not in entry:
                step = entry.get("step", 0)
                loss = entry.get("loss", 0)
                pct = (step / max_steps * 100) if max_steps > 0 else 0
                summaries.append(f"Step {step} ({pct:.0f}%) loss={loss:.3f}")
        return summaries

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses to navigate to filtered logs."""
        if event.button.id == "btn-warnings":
            # Switch to Logs tab and filter by warnings
            app = self.app
            app.query_one(TabbedContent).active = "tab-logs"
            logs_panel = app.query_one(LogsPanel)
            logs_panel.filter_by_level("WARNING")
        elif event.button.id == "btn-errors":
            # Switch to Logs tab and filter by errors
            app = self.app
            app.query_one(TabbedContent).active = "tab-logs"
            logs_panel = app.query_one(LogsPanel)
            logs_panel.filter_by_level("ERROR")


class TimelinePanel(VerticalScroll):
    """Timeline panel showing training events."""

    DEFAULT_CSS = """
    TimelinePanel {
        padding: 1 2;
    }

    TimelinePanel .event-row {
        height: auto;
    }

    TimelinePanel .event-time {
        color: $text-muted;
        width: 10;
    }

    TimelinePanel .event-marker {
        width: 2;
        color: $primary;
    }

    TimelinePanel .event-title {
        text-style: bold;
        color: $primary;
        width: 20;
    }

    TimelinePanel .event-duration {
        color: $text-muted;
        width: 10;
    }

    TimelinePanel .event-info {
        color: $text;
    }

    TimelinePanel .event-detail {
        margin-left: 12;
        color: $text-muted;
    }

    TimelinePanel .event-warning {
        margin-left: 12;
        color: $warning;
    }

    TimelinePanel .metric-step {
        margin-left: 12;
        color: $success;
    }
    """

    def __init__(self, data: TrainingData, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def compose(self) -> ComposeResult:
        yield Static("[bold]TRAINING TIMELINE[/bold]")
        yield Rule()

        # Group log entries into phases
        phases = self._group_into_phases()

        # Calculate total training time
        if len(phases) >= 2:
            total_time = (phases[-1]["time"] - phases[0]["time"]).total_seconds()
        else:
            total_time = 0

        prev_time = None
        for phase in phases:
            time_str = phase["time"].strftime("%H:%M:%S") if phase["time"] else ""

            # Calculate duration since previous phase
            duration_str = ""
            if prev_time and phase["time"]:
                delta = (phase["time"] - prev_time).total_seconds()
                if delta >= 60:
                    duration_str = f"+{delta/60:.1f}m"
                else:
                    duration_str = f"+{delta:.1f}s"

            # Build info string
            info_str = phase.get("info", "")

            yield Horizontal(
                Static(time_str, classes="event-time"),
                Static("●", classes="event-marker"),
                Static(phase["title"], classes="event-title"),
                Static(duration_str, classes="event-duration"),
                Static(info_str, classes="event-info"),
                classes="event-row",
            )

            # Show details (warnings, etc)
            if phase.get("details"):
                for detail in phase["details"]:
                    yield Static(f"⚠ {detail}", classes="event-warning")

            # Show step metrics if this is a training phase
            if phase.get("steps"):
                for step_info in phase["steps"]:
                    yield Static(step_info, classes="metric-step")

            prev_time = phase["time"]

        # Summary line
        if total_time > 0:
            yield Rule()
            yield Static(f"[dim]Total time: {total_time:.1f}s ({total_time/60:.1f}m)[/dim]")

    def _group_into_phases(self) -> list[dict]:
        """Group log entries into training phases.

        Only shows the most recent training run (based on the last "starting training" message).
        """
        all_phases = []
        training_start_indices = []
        dataset_count = 0  # Track which dataset we're on

        for i, entry in enumerate(self.data.log_entries):
            msg = entry.message.lower()
            msg_orig = entry.message

            if "resolved" in msg and "dataloader" in msg:
                all_phases.append(
                    {
                        "time": entry.timestamp,
                        "title": "INIT",
                        "info": self._extract_config_info(msg_orig),
                        "details": [],
                        "entry_index": i,
                    }
                )
                dataset_count = 0  # Reset dataset counter for new run
            elif "creating map dataset" in msg:
                dataset_count += 1
                # Extract dataset type and name from message
                # Example: "Creating map dataset (type: AlpacaDataset)... dataset_name: 'yahma/alpaca-cleaned'"
                ds_type, ds_name = self._extract_dataset_type(msg_orig)

                all_phases.append(
                    {
                        "time": entry.timestamp,
                        "title": f"DATASET ({ds_type})",
                        "info": ds_name,
                        "details": [],
                        "entry_index": i,
                    }
                )
            elif "finished transforming dataset" in msg:
                # Update the last dataset phase with example count
                # Example: "Finished transforming dataset (AlpacaDataset)! ... Examples: 51760."
                if all_phases and "DATASET" in all_phases[-1].get("title", ""):
                    import re
                    match = re.search(r"Examples:\s*(\d+)", msg_orig)
                    if match:
                        count = int(match.group(1))
                        current_info = all_phases[-1].get("info", "")
                        all_phases[-1]["info"] = f"{current_info} ({count:,} examples)" if current_info else f"{count:,} examples"
            elif "model parameters summary" in msg:
                # Extract model info
                model_info = self._extract_model_info()
                all_phases.append(
                    {
                        "time": entry.timestamp,
                        "title": "MODEL LOADED",
                        "info": model_info,
                        "details": [],
                        "entry_index": i,
                    }
                )
            elif "starting training" in msg:
                training_start_indices.append(len(all_phases))
                all_phases.append(
                    {
                        "time": entry.timestamp,
                        "title": "TRAINING",
                        "info": f"{self.data.max_steps} steps",
                        "details": [],
                        "entry_index": i,
                    }
                )
            elif "training is complete" in msg:
                # Calculate training stats
                train_info = self._get_training_summary()
                all_phases.append(
                    {
                        "time": entry.timestamp,
                        "title": "COMPLETE",
                        "info": train_info,
                        "details": [],
                        "entry_index": i,
                    }
                )
            elif entry.level == "WARNING":
                # Add warnings inline
                if all_phases:
                    if "details" not in all_phases[-1]:
                        all_phases[-1]["details"] = []
                    # Don't add ⚠ prefix here, it's added in the CSS class
                    all_phases[-1]["details"].append(entry.message[:70])

        # Only add step metrics to the LAST training started phase
        if training_start_indices:
            last_training_idx = training_start_indices[-1]
            all_phases[last_training_idx]["steps"] = self._get_step_summaries()

        # Only return phases from the most recent training run
        if training_start_indices:
            last_training_idx = training_start_indices[-1]
            start_idx = 0
            for i in range(last_training_idx - 1, -1, -1):
                if all_phases[i]["title"] == "INIT":
                    start_idx = i
                    break
            return all_phases[start_idx:]

        return all_phases

    def _extract_config_info(self, msg: str) -> str:
        """Extract config info from init message."""
        # Example: "Resolved 'training.dataloader_num_workers=auto' to 'training.dataloader_num_workers=2'"
        import re
        # Look for the final value after "to"
        match = re.search(r"to\s*['\"]?(?:[\w.]*=)?(\d+)['\"]?", msg)
        if match:
            return f"workers={match.group(1)}"
        return ""

    def _extract_dataset_type(self, msg: str) -> tuple[str, str]:
        """Extract dataset type and name from creation message.

        Returns (type, name) tuple.
        Example input: "Creating map dataset (type: AlpacaDataset)... dataset_name: 'yahma/alpaca-cleaned'"
        """
        import re
        ds_type = "Train"  # Default
        ds_name = ""

        # Check for explicit train/eval in message
        msg_lower = msg.lower()
        if "eval" in msg_lower or "valid" in msg_lower or "test" in msg_lower:
            ds_type = "Eval"

        # Extract dataset name
        name_match = re.search(r"dataset_name:\s*['\"]?([^'\"]+)['\"]?", msg)
        if name_match:
            ds_name = name_match.group(1).strip()
            # Shorten if too long
            if len(ds_name) > 30:
                ds_name = "..." + ds_name[-27:]

        return ds_type, ds_name

    def _extract_model_info(self) -> str:
        """Extract model info from config."""
        if self.data.model_config:
            params = self.data.model_config.get("num_parameters", 0)
            if params:
                if params >= 1e9:
                    return f"{params/1e9:.1f}B params"
                elif params >= 1e6:
                    return f"{params/1e6:.0f}M params"
        if self.data.training_config:
            model_name = self.data.training_config.get("model", {}).get("model_name", "")
            if model_name:
                # Extract size from name like "SmolLM2-135M"
                import re
                match = re.search(r"(\d+(?:\.\d+)?[BMK])", model_name)
                if match:
                    return match.group(1)
        return ""

    def _get_training_summary(self) -> str:
        """Get training completion summary."""
        parts = []
        if self.data.log_history:
            losses = [e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
            if losses:
                parts.append(f"loss={losses[-1]:.3f}")
        if self.data.metrics_summary:
            runtime = self.data.metrics_summary.get("train_runtime", 0)
            if runtime:
                parts.append(f"{runtime:.1f}s")
        return " | ".join(parts) if parts else ""

    def _get_step_summaries(self) -> list[str]:
        """Get formatted step metric summaries."""
        summaries = []
        max_steps = self.data.max_steps
        for entry in self.data.log_history:
            if "loss" in entry and "train_loss" not in entry:
                step = entry.get("step", 0)
                loss = entry.get("loss", 0)
                acc = entry.get("mean_token_accuracy", 0)
                tps = entry.get("train_tokens_per_second", 0)
                pct = (step / max_steps * 100) if max_steps > 0 else 0
                summaries.append(
                    f"Step {step}/{max_steps} ({pct:.0f}%) · loss={loss:.3f} · acc={acc*100:.1f}% · {tps:.0f} tok/s"
                )
        return summaries


class MetricsPanel(VerticalScroll):
    """Metrics panel with charts."""

    DEFAULT_CSS = """
    MetricsPanel {
        padding: 1 2;
    }

    MetricsPanel .chart-box {
        height: 12;
        margin: 1 0;
        border: solid $primary-background;
    }

    MetricsPanel .metric-title {
        text-style: bold;
        color: $primary;
        margin-top: 1;
    }

    MetricsPanel .metric-stats {
        color: $text-muted;
    }

    MetricsPanel .stats-table {
        margin-top: 1;
    }

    MetricsPanel .no-plotext {
        padding: 1;
        color: $warning;
    }
    """

    def __init__(self, data: TrainingData, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def compose(self) -> ComposeResult:
        if not HAS_PLOTEXT:
            yield Static(
                "[yellow]⚠[/yellow] Install textual-plotext for charts: pip install textual-plotext",
                classes="no-plotext",
            )
            # Fallback to text display
            yield from self._compose_text_fallback()
        else:
            yield from self._compose_plotext_charts()

        # Stats table at the bottom
        yield Static("[bold]STATISTICS[/bold]", classes="metric-title")
        yield self._compose_stats_table()

    def _compose_plotext_charts(self) -> ComposeResult:
        """Compose PlotextPlot charts for each metric."""
        # Training Loss
        losses = [e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
        steps_loss = [e["step"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
        if losses:
            yield Static("[bold]Training Loss[/bold]", classes="metric-title")
            yield PlotextPlot(id="chart-loss", classes="chart-box")
            change = ((losses[-1] - losses[0]) / losses[0] * 100) if losses[0] != 0 else 0
            direction = "↓" if change < 0 else "↑"
            yield Static(
                f"Start: {losses[0]:.4f} → End: {losses[-1]:.4f} ({direction} {abs(change):.1f}%)",
                classes="metric-stats",
            )

        # Learning Rate
        lrs = [e["learning_rate"] for e in self.data.log_history if "learning_rate" in e]
        steps_lr = [e["step"] for e in self.data.log_history if "learning_rate" in e]
        if lrs:
            yield Static("[bold]Learning Rate[/bold]", classes="metric-title")
            yield PlotextPlot(id="chart-lr", classes="chart-box")
            yield Static(
                f"Start: {lrs[0]:.2e} → End: {lrs[-1]:.2e}",
                classes="metric-stats",
            )

        # Accuracy
        accs = [e["mean_token_accuracy"] for e in self.data.log_history if "mean_token_accuracy" in e]
        steps_acc = [e["step"] for e in self.data.log_history if "mean_token_accuracy" in e]
        if accs:
            yield Static("[bold]Token Accuracy[/bold]", classes="metric-title")
            yield PlotextPlot(id="chart-acc", classes="chart-box")
            change = ((accs[-1] - accs[0]) / accs[0] * 100) if accs[0] != 0 else 0
            direction = "↑" if change > 0 else "↓"
            yield Static(
                f"Start: {accs[0]*100:.1f}% → End: {accs[-1]*100:.1f}% ({direction} {abs(change):.1f}%)",
                classes="metric-stats",
            )

        # Gradient Norm
        grad_norms = [e["grad_norm"] for e in self.data.log_history if "grad_norm" in e]
        steps_grad = [e["step"] for e in self.data.log_history if "grad_norm" in e]
        if grad_norms:
            yield Static("[bold]Gradient Norm[/bold]", classes="metric-title")
            yield PlotextPlot(id="chart-grad", classes="chart-box")
            yield Static(
                f"Start: {grad_norms[0]:.4f} → End: {grad_norms[-1]:.4f}",
                classes="metric-stats",
            )

    def _compose_text_fallback(self) -> ComposeResult:
        """Text-based fallback when plotext is unavailable."""
        losses = [e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
        if losses:
            yield Static(f"[bold]Loss:[/bold] {losses[0]:.4f} → {losses[-1]:.4f}")

        lrs = [e["learning_rate"] for e in self.data.log_history if "learning_rate" in e]
        if lrs:
            yield Static(f"[bold]LR:[/bold] {lrs[0]:.2e} → {lrs[-1]:.2e}")

        accs = [e["mean_token_accuracy"] for e in self.data.log_history if "mean_token_accuracy" in e]
        if accs:
            yield Static(f"[bold]Accuracy:[/bold] {accs[0]*100:.1f}% → {accs[-1]*100:.1f}%")

    def on_mount(self) -> None:
        """Initialize charts after mount."""
        if HAS_PLOTEXT:
            self._update_charts()

    def _update_charts(self) -> None:
        """Update all plotext charts."""
        # Loss chart
        losses = [e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
        steps_loss = [e["step"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
        if losses and steps_loss:
            try:
                chart = self.query_one("#chart-loss", PlotextPlot)
                plt = chart.plt
                plt.clear_figure()
                plt.plot(steps_loss, losses, marker="braille")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                chart.refresh()
            except Exception:
                pass

        # Learning Rate chart
        lrs = [e["learning_rate"] for e in self.data.log_history if "learning_rate" in e]
        steps_lr = [e["step"] for e in self.data.log_history if "learning_rate" in e]
        if lrs and steps_lr:
            try:
                chart = self.query_one("#chart-lr", PlotextPlot)
                plt = chart.plt
                plt.clear_figure()
                plt.plot(steps_lr, lrs, marker="braille")
                plt.xlabel("Step")
                plt.ylabel("LR")
                chart.refresh()
            except Exception:
                pass

        # Accuracy chart
        accs = [e["mean_token_accuracy"] for e in self.data.log_history if "mean_token_accuracy" in e]
        steps_acc = [e["step"] for e in self.data.log_history if "mean_token_accuracy" in e]
        if accs and steps_acc:
            try:
                chart = self.query_one("#chart-acc", PlotextPlot)
                plt = chart.plt
                plt.clear_figure()
                plt.plot(steps_acc, [a * 100 for a in accs], marker="braille")
                plt.xlabel("Step")
                plt.ylabel("Acc %")
                chart.refresh()
            except Exception:
                pass

        # Gradient Norm chart
        grad_norms = [e["grad_norm"] for e in self.data.log_history if "grad_norm" in e]
        steps_grad = [e["step"] for e in self.data.log_history if "grad_norm" in e]
        if grad_norms and steps_grad:
            try:
                chart = self.query_one("#chart-grad", PlotextPlot)
                plt = chart.plt
                plt.clear_figure()
                plt.plot(steps_grad, grad_norms, marker="braille")
                plt.xlabel("Step")
                plt.ylabel("Grad")
                chart.refresh()
            except Exception:
                pass

    def _compose_stats_table(self) -> DataTable:
        """Compose statistics table."""
        table = DataTable(classes="stats-table")
        table.add_columns("Metric", "Start", "End", "Change", "Min", "Max")

        losses = [e["loss"] for e in self.data.log_history if "loss" in e and "train_loss" not in e]
        if losses:
            change = (losses[-1] - losses[0]) / losses[0] * 100 if losses[0] != 0 else 0
            table.add_row(
                "Loss",
                f"{losses[0]:.4f}",
                f"{losses[-1]:.4f}",
                f"{change:+.1f}%",
                f"{min(losses):.4f}",
                f"{max(losses):.4f}",
            )

        lrs = [e["learning_rate"] for e in self.data.log_history if "learning_rate" in e]
        if lrs:
            change = (lrs[-1] - lrs[0]) / lrs[0] * 100 if lrs[0] != 0 else 0
            table.add_row(
                "LR",
                f"{lrs[0]:.2e}",
                f"{lrs[-1]:.2e}",
                f"{change:+.1f}%",
                f"{min(lrs):.2e}",
                f"{max(lrs):.2e}",
            )

        accs = [e["mean_token_accuracy"] for e in self.data.log_history if "mean_token_accuracy" in e]
        if accs:
            change = (accs[-1] - accs[0]) / accs[0] * 100 if accs[0] != 0 else 0
            table.add_row(
                "Accuracy",
                f"{accs[0]*100:.1f}%",
                f"{accs[-1]*100:.1f}%",
                f"{change:+.1f}%",
                f"{min(accs)*100:.1f}%",
                f"{max(accs)*100:.1f}%",
            )

        grad_norms = [e["grad_norm"] for e in self.data.log_history if "grad_norm" in e]
        if grad_norms:
            change = (grad_norms[-1] - grad_norms[0]) / grad_norms[0] * 100 if grad_norms[0] != 0 else 0
            table.add_row(
                "Grad Norm",
                f"{grad_norms[0]:.4f}",
                f"{grad_norms[-1]:.4f}",
                f"{change:+.1f}%",
                f"{min(grad_norms):.4f}",
                f"{max(grad_norms):.4f}",
            )

        return table


class LogsPanel(Container):
    """Logs panel with filtering."""

    DEFAULT_CSS = """
    LogsPanel {
        padding: 1;
    }

    LogsPanel .filter-bar {
        height: 3;
        margin-bottom: 1;
    }

    LogsPanel .log-viewer {
        height: 100%;
        border: solid $primary-background;
    }

    LogsPanel .stats-bar {
        height: 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, data: TrainingData, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.filter_level: Optional[str] = None
        self.search_term: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="filter-bar"):
            yield Input(placeholder="Search logs...", id="log-search")
            yield Button("All", id="filter-all", variant="primary")
            yield Button("Warn", id="filter-warn", variant="default")
            yield Button("Error", id="filter-error", variant="default")

        yield Static(
            f"Showing {len(self.data.log_entries)} entries | "
            f"⚠ {self.data.warnings_count} warnings | "
            f"✗ {self.data.errors_count} errors",
            id="stats-bar",
            classes="stats-bar",
        )

        yield RichLog(id="log-viewer", classes="log-viewer", highlight=True, markup=True)

    def on_mount(self) -> None:
        """Populate log viewer after mount."""
        self._refresh_logs()

    def _refresh_logs(self) -> None:
        """Refresh the log display with current filters."""
        log_viewer = self.query_one("#log-viewer", RichLog)
        log_viewer.clear()

        shown_count = 0
        for entry in self.data.log_entries:
            # Apply level filter
            if self.filter_level and entry.level != self.filter_level:
                continue

            # Apply search filter
            if self.search_term and self.search_term.lower() not in entry.message.lower():
                continue

            shown_count += 1

            # Format and display
            time_str = entry.timestamp.strftime("%H:%M:%S")
            level_color = {"INFO": "blue", "WARNING": "yellow", "ERROR": "red"}.get(
                entry.level, "white"
            )

            log_viewer.write(
                f"[dim]{time_str}[/dim] [{level_color}]{entry.level:5}[/{level_color}] "
                f"[dim]{entry.source}[/dim]"
            )
            log_viewer.write(f"  {entry.message}")

        # Update stats bar
        filter_text = f"[{self.filter_level}]" if self.filter_level else "[All]"
        try:
            stats_bar = self.query_one("#stats-bar", Static)
            stats_bar.update(
                f"Showing {shown_count}/{len(self.data.log_entries)} entries {filter_text} | "
                f"⚠ {self.data.warnings_count} warnings | "
                f"✗ {self.data.errors_count} errors"
            )
        except Exception:
            pass

    def _update_button_states(self) -> None:
        """Update button variants to reflect current filter."""
        try:
            btn_all = self.query_one("#filter-all", Button)
            btn_warn = self.query_one("#filter-warn", Button)
            btn_error = self.query_one("#filter-error", Button)

            btn_all.variant = "primary" if self.filter_level is None else "default"
            btn_warn.variant = "warning" if self.filter_level == "WARNING" else "default"
            btn_error.variant = "error" if self.filter_level == "ERROR" else "default"
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle filter button presses."""
        button_id = event.button.id
        if button_id == "filter-all":
            self.filter_level = None
        elif button_id == "filter-warn":
            self.filter_level = "WARNING"
        elif button_id == "filter-error":
            self.filter_level = "ERROR"
        self._update_button_states()
        self._refresh_logs()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "log-search":
            self.search_term = event.value
            self._refresh_logs()

    def filter_by_level(self, level: str) -> None:
        """Filter logs by level (called externally from Summary panel)."""
        self.filter_level = level
        self._update_button_states()
        self._refresh_logs()


class ConfigPanel(VerticalScroll):
    """Config panel with tree view."""

    DEFAULT_CSS = """
    ConfigPanel {
        padding: 1;
    }

    ConfigPanel .no-config {
        padding: 2;
        text-align: center;
        color: $text-muted;
    }

    ConfigPanel Tree {
        background: transparent;
    }

    ConfigPanel Tree:focus {
        background: transparent;
    }

    ConfigPanel Tree > .tree--cursor {
        background: $primary 20%;
    }

    ConfigPanel Tree:focus > .tree--cursor {
        background: $primary 30%;
    }
    """

    def __init__(self, data: TrainingData, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def compose(self) -> ComposeResult:
        if not self.data.training_config:
            yield Static(
                "No training configuration found.\n"
                "Expected file: telemetry/training_config.yaml",
                classes="no-config",
            )
            return

        tree: Tree[dict] = Tree("Training Configuration", id="config-tree")
        tree.root.expand()
        self._add_config_to_tree(tree.root, self.data.training_config)
        yield tree

    def _add_config_to_tree(self, node: TreeNode, data: Any, key: str = "", depth: int = 0) -> None:
        """Recursively add configuration data to tree."""
        # Expand first 2 levels by default
        expand_by_default = depth < 2

        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    child = node.add(f"[bold]{k}[/bold]", expand=expand_by_default)
                    self._add_config_to_tree(child, v, k, depth + 1)
                else:
                    formatted = self._format_value(v)
                    node.add_leaf(f"{k}: {formatted}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    child = node.add(f"[{i}]", expand=expand_by_default)
                    self._add_config_to_tree(child, item, depth=depth + 1)
                else:
                    formatted = self._format_value(item)
                    node.add_leaf(f"[{i}]: {formatted}")
        else:
            formatted = self._format_value(data)
            node.add_leaf(formatted)

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "[dim]null[/dim]"
        elif isinstance(value, bool):
            color = "green" if value else "red"
            return f"[{color}]{value}[/{color}]"
        elif isinstance(value, (int, float)):
            return f"[yellow]{value}[/yellow]"
        elif isinstance(value, str):
            return f"[green]{value}[/green]"
        return str(value)


class FilesPanel(VerticalScroll):
    """Files panel showing checkpoints and outputs."""

    DEFAULT_CSS = """
    FilesPanel {
        padding: 1 2;
    }

    FilesPanel .section-title {
        text-style: bold;
        margin-top: 1;
        color: $primary;
    }

    FilesPanel .file-table {
        margin: 1 0;
    }

    FilesPanel .usage-box {
        background: $surface;
        padding: 1;
        margin: 1 0;
        border: solid $primary-background;
    }

    FilesPanel .code {
        background: $surface-darken-1;
        padding: 0 1;
    }
    """

    def __init__(self, data: TrainingData, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def compose(self) -> ComposeResult:
        # Checkpoints table
        yield Static("CHECKPOINTS", classes="section-title")
        if self.data.checkpoints:
            table = DataTable(classes="file-table", id="checkpoints-table")
            table.add_columns("Name", "Step", "Size", "Contents")
            for cp in self.data.checkpoints:
                contents = []
                if cp.has_model:
                    contents.append("model")
                if cp.has_optimizer:
                    contents.append("optimizer")
                table.add_row(
                    cp.path.name,
                    str(cp.step),
                    format_size(cp.size_bytes),
                    ", ".join(contents) if contents else "minimal",
                )
            yield table
        else:
            yield Static("[dim]No checkpoints found[/dim]")

        # Output files
        yield Static("OUTPUT FILES", classes="section-title")
        files_table = DataTable(classes="file-table", id="files-table")
        files_table.add_columns("File", "Size", "Description")

        # List key files
        key_files = [
            ("model.safetensors", "Final model weights"),
            ("tokenizer.json", "Tokenizer vocabulary"),
            ("config.json", "Model configuration"),
            ("trainer_state.json", "Training state & history"),
        ]

        for filename, desc in key_files:
            filepath = self.data.folder_path / filename
            if filepath.exists():
                size = format_size(filepath.stat().st_size)
                files_table.add_row(filename, size, desc)

        yield files_table

        # TensorBoard
        runs_dir = self.data.folder_path / "runs"
        if runs_dir.exists():
            event_count = len(list(runs_dir.glob("**/events.out.tfevents.*")))
            yield Static("TENSORBOARD", classes="section-title")
            with Container(classes="usage-box"):
                yield Static(f"Found {event_count} event file(s) in runs/")
                yield Static("")
                yield Static("Run command:", classes="code")
                yield Static(
                    f"  tensorboard --logdir {self.data.folder_path / 'runs'}",
                    classes="code",
                )

        # Usage
        yield Static("USAGE", classes="section-title")
        with Container(classes="usage-box"):
            yield Static("Load model:", classes="code")
            yield Static(
                f'  model = AutoModelForCausalLM.from_pretrained("{self.data.folder_path}")',
                classes="code",
            )


# =============================================================================
# Main Application
# =============================================================================


class TrainingViewerApp(App):
    """TUI application for viewing training output folders."""

    TITLE = "Oumi Training Dashboard"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("1", "tab_1", "Summary", show=False),
        Binding("2", "tab_2", "Metrics", show=False),
        Binding("3", "tab_3", "Logs", show=False),
        Binding("4", "tab_4", "Config", show=False),
        Binding("5", "tab_5", "Files", show=False),
        Binding("t", "copy_tensorboard", "Copy TB cmd"),
        Binding("o", "open_folder", "Open folder"),
        # Vim-style navigation (priority=True to override widget bindings)
        Binding("j", "scroll_down", "Scroll ↓", show=False, priority=True),
        Binding("k", "scroll_up", "Scroll ↑", show=False, priority=True),
        Binding("down", "scroll_down", "Scroll ↓", show=False, priority=True),
        Binding("up", "scroll_up", "Scroll ↑", show=False, priority=True),
        Binding("g", "goto_top", "Top", show=False, priority=True),
        Binding("G", "goto_bottom", "Bottom", show=False, priority=True),
        Binding("pagedown", "page_down", "Page Down", show=False, priority=True),
        Binding("pageup", "page_up", "Page Up", show=False, priority=True),
        # Search
        Binding("/", "focus_search", "Search", show=False),
        Binding("escape", "clear_search", "Clear", show=False),
        # Log filters
        Binding("w", "filter_warnings", "Warnings", show=False),
        Binding("e", "filter_errors", "Errors", show=False),
        # Copy
        Binding("y", "copy_item", "Copy", show=False),
    ]

    DEFAULT_CSS = """
    TrainingViewerApp {
        background: $background;
        layout: vertical;
    }

    #tabs {
        height: 1fr;
        min-height: 10;
    }

    #tabs > ContentSwitcher {
        height: 1fr;
    }

    TabPane {
        padding: 0;
        height: 1fr;
    }

    SummaryPanel, TimelinePanel, MetricsPanel, LogsPanel, ConfigPanel, FilesPanel {
        height: 1fr;
        min-height: 5;
    }

    .status-bar {
        dock: bottom;
        height: 1;
        background: $primary-background;
        padding: 0 1;
    }
    """

    def __init__(self, folder_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.theme = "flexoki"
        self.folder_path = folder_path
        self.data: Optional[TrainingData] = None

    def compose(self) -> ComposeResult:
        yield Header()

        # Load data synchronously for now
        try:
            self.data = load_training_data(self.folder_path)
        except Exception as e:
            yield Static(f"[red]Error loading training data:[/red] {e}")
            yield Footer()
            return

        with TabbedContent(id="tabs"):
            with TabPane("Summary", id="tab-summary"):
                yield SummaryPanel(self.data)
            with TabPane("Metrics", id="tab-metrics"):
                yield MetricsPanel(self.data)
            with TabPane("Logs", id="tab-logs"):
                yield LogsPanel(self.data)
            with TabPane("Config", id="tab-config"):
                yield ConfigPanel(self.data)
            with TabPane("Files", id="tab-files"):
                yield FilesPanel(self.data)

        # Status bar
        status_text = f"{self.folder_path.name}"
        if self.data:
            status_text += f" | Step {self.data.global_step}/{self.data.max_steps}"
            if self.data.log_history:
                losses = [e.get("loss") for e in self.data.log_history if "loss" in e and "train_loss" not in e]
                if losses:
                    status_text += f" | Loss: {losses[-1]:.4f}"
        yield Static(status_text, classes="status-bar")

        yield Footer()

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())

    def action_tab_1(self) -> None:
        """Switch to Summary tab."""
        self.query_one(TabbedContent).active = "tab-summary"

    def action_tab_2(self) -> None:
        """Switch to Metrics tab."""
        self.query_one(TabbedContent).active = "tab-metrics"

    def action_tab_3(self) -> None:
        """Switch to Logs tab."""
        self.query_one(TabbedContent).active = "tab-logs"

    def action_tab_4(self) -> None:
        """Switch to Config tab."""
        self.query_one(TabbedContent).active = "tab-config"

    def action_tab_5(self) -> None:
        """Switch to Files tab."""
        self.query_one(TabbedContent).active = "tab-files"

    def action_copy_tensorboard(self) -> None:
        """Copy TensorBoard command to clipboard."""
        cmd = f"tensorboard --logdir {self.folder_path / 'runs'}"
        if copy_to_clipboard(cmd):
            self.notify("TensorBoard command copied to clipboard")
        else:
            self.notify("Failed to copy to clipboard", severity="error")

    def action_open_folder(self) -> None:
        """Open the training folder in file manager."""
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(self.folder_path)])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(self.folder_path)])
            else:
                subprocess.run(["explorer", str(self.folder_path)])
        except Exception:
            self.notify("Failed to open folder", severity="error")

    def _get_active_scrollable(self):
        """Get the scrollable widget in the active tab."""
        try:
            tabs = self.query_one(TabbedContent)
            active_tab = tabs.active

            # Handle each tab specifically for best scrolling behavior
            if active_tab == "tab-logs":
                # LogsPanel contains a RichLog that should be scrolled
                return self.query_one("#log-viewer", RichLog)
            elif active_tab == "tab-config":
                # Config tab has a Tree widget - scroll the tree directly
                return self.query_one("#config-tree", Tree)
            else:
                # For other tabs (Summary, Metrics, Files, Timeline),
                # find the VerticalScroll panel
                active_pane = tabs.get_pane(active_tab)
                if active_pane:
                    for child in active_pane.walk_children():
                        if isinstance(child, VerticalScroll):
                            return child
        except Exception:
            pass
        return None

    def action_scroll_down(self) -> None:
        """Scroll down in the current panel."""
        scrollable = self._get_active_scrollable()
        if scrollable:
            scrollable.scroll_down()

    def action_scroll_up(self) -> None:
        """Scroll up in the current panel."""
        scrollable = self._get_active_scrollable()
        if scrollable:
            scrollable.scroll_up()

    def action_goto_top(self) -> None:
        """Scroll to the top of the current panel."""
        scrollable = self._get_active_scrollable()
        if scrollable and hasattr(scrollable, "scroll_home"):
            scrollable.scroll_home()

    def action_goto_bottom(self) -> None:
        """Scroll to the bottom of the current panel."""
        scrollable = self._get_active_scrollable()
        if scrollable and hasattr(scrollable, "scroll_end"):
            scrollable.scroll_end()

    def action_page_down(self) -> None:
        """Page down in the current panel."""
        scrollable = self._get_active_scrollable()
        if scrollable and hasattr(scrollable, "scroll_page_down"):
            scrollable.scroll_page_down()

    def action_page_up(self) -> None:
        """Page up in the current panel."""
        scrollable = self._get_active_scrollable()
        if scrollable and hasattr(scrollable, "scroll_page_up"):
            scrollable.scroll_page_up()

    def action_focus_search(self) -> None:
        """Focus the search input in the Logs panel."""
        try:
            # Switch to logs tab and focus search
            self.query_one(TabbedContent).active = "tab-logs"
            logs_panel = self.query_one(LogsPanel)
            search_input = logs_panel.query_one("#log-search", Input)
            search_input.focus()
        except Exception:
            pass

    def action_clear_search(self) -> None:
        """Clear the search in the Logs panel, or quit if nothing is active."""
        try:
            logs_panel = self.query_one(LogsPanel)
            search_input = logs_panel.query_one("#log-search", Input)

            # If search has content or is focused, clear it
            if search_input.value or search_input.has_focus:
                search_input.value = ""
                logs_panel.search_term = ""
                logs_panel._refresh_logs()
            else:
                # Nothing active, quit the app
                self.exit()
        except Exception:
            # If we can't find the logs panel, just quit
            self.exit()

    def action_filter_warnings(self) -> None:
        """Filter logs to show only warnings."""
        try:
            self.query_one(TabbedContent).active = "tab-logs"
            logs_panel = self.query_one(LogsPanel)
            if logs_panel.filter_level == "WARNING":
                logs_panel.filter_level = None  # Toggle off
            else:
                logs_panel.filter_level = "WARNING"
            logs_panel._update_button_states()
            logs_panel._refresh_logs()
            self.notify("Showing warnings only" if logs_panel.filter_level else "Showing all logs")
        except Exception:
            pass

    def action_filter_errors(self) -> None:
        """Filter logs to show only errors."""
        try:
            self.query_one(TabbedContent).active = "tab-logs"
            logs_panel = self.query_one(LogsPanel)
            if logs_panel.filter_level == "ERROR":
                logs_panel.filter_level = None  # Toggle off
            else:
                logs_panel.filter_level = "ERROR"
            logs_panel._update_button_states()
            logs_panel._refresh_logs()
            self.notify("Showing errors only" if logs_panel.filter_level else "Showing all logs")
        except Exception:
            pass

    def action_copy_item(self) -> None:
        """Copy the current item to clipboard."""
        try:
            # Get the active tab
            tabs = self.query_one(TabbedContent)
            active_tab = tabs.active

            if active_tab == "tab-summary" and self.data:
                # Copy summary info
                info = f"Training: {self.folder_path.name}\n"
                info += f"Steps: {self.data.global_step}/{self.data.max_steps}\n"
                if self.data.log_history:
                    losses = [e.get("loss") for e in self.data.log_history if "loss" in e and "train_loss" not in e]
                    if losses:
                        info += f"Loss: {losses[-1]:.4f}\n"
                if copy_to_clipboard(info):
                    self.notify("Summary copied to clipboard")
                else:
                    self.notify("Failed to copy", severity="error")
            elif active_tab == "tab-files":
                # Copy folder path
                if copy_to_clipboard(str(self.folder_path)):
                    self.notify("Folder path copied to clipboard")
                else:
                    self.notify("Failed to copy", severity="error")
            else:
                self.notify("Nothing to copy in this tab", severity="warning")
        except Exception:
            self.notify("Failed to copy", severity="error")
