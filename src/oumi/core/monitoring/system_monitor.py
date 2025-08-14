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

"""System monitoring for GPU/CPU/RAM utilization and context window tracking."""

import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class SystemStats:
    """Container for system resource statistics."""

    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_vram_used_gb: Optional[float] = None
    gpu_vram_total_gb: Optional[float] = None
    gpu_vram_percent: Optional[float] = None
    gpu_compute_percent: Optional[float] = None
    context_used_tokens: int = 0
    context_max_tokens: int = 0
    context_percent: float = 0.0
    conversation_turns: int = 0


class SystemMonitor:
    """Monitor system resources and display HUD."""

    def __init__(self, max_context_tokens: int = 4096, update_interval: float = 15.0):
        """Initialize the system monitor.

        Args:
            max_context_tokens: Maximum context window size in tokens.
            update_interval: Seconds between HUD updates.
        """
        self.max_context_tokens = max_context_tokens
        self.update_interval = update_interval
        self.last_update_time = 0.0
        self._context_used_tokens = 0
        self._conversation_turns = 0

        # Try to import optional monitoring libraries
        self._psutil = None
        self._nvidia_ml = None
        self._nvidia_available = False

        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            pass

        try:
            import pynvml

            self._nvidia_ml = pynvml
            try:
                self._nvidia_ml.nvmlInit()
                self._nvidia_available = True
                # Get first GPU handle for monitoring
                self._gpu_handle = self._nvidia_ml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvidia_available = False
        except ImportError:
            pass

    def update_context_usage(self, tokens_used: int):
        """Update the current context window usage.

        Args:
            tokens_used: Number of tokens currently in use.
        """
        self._context_used_tokens = tokens_used

    def update_max_context_tokens(self, max_context_tokens: int):
        """Update the maximum context window size.

        Args:
            max_context_tokens: New maximum context window size in tokens.
        """
        # Ensure we never set None as max_context_tokens
        if max_context_tokens is None:
            print(f"🔧 DEBUG: Attempted to set max_context_tokens to None, defaulting to 4096")
            self.max_context_tokens = 4096
        else:
            self.max_context_tokens = max_context_tokens

    def update_conversation_turns(self, turns: int):
        """Update the number of conversation turns.

        Args:
            turns: Current number of conversation turns (user+assistant exchanges).
        """
        self._conversation_turns = turns

    def get_stats(self) -> SystemStats:
        """Collect current system statistics.

        Returns:
            SystemStats object with current metrics.
        """
        stats = SystemStats(
            cpu_percent=0.0,
            ram_used_gb=0.0,
            ram_total_gb=0.0,
            ram_percent=0.0,
            context_used_tokens=self._context_used_tokens,
            context_max_tokens=self.max_context_tokens,
            context_percent=(
                (self._context_used_tokens / self.max_context_tokens * 100)
                if self.max_context_tokens is not None and self.max_context_tokens > 0
                else 0.0
            ),
            conversation_turns=self._conversation_turns,
        )

        # CPU and RAM stats using psutil
        if self._psutil:
            try:
                # CPU usage (average over 0.1 seconds)
                stats.cpu_percent = self._psutil.cpu_percent(interval=0.1)

                # RAM usage
                memory = self._psutil.virtual_memory()
                stats.ram_used_gb = memory.used / (1024**3)
                stats.ram_total_gb = memory.total / (1024**3)
                stats.ram_percent = memory.percent
            except Exception:
                pass

        # GPU stats using nvidia-ml-py
        if self._nvidia_available and self._nvidia_ml:
            try:
                # GPU memory info
                mem_info = self._nvidia_ml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                stats.gpu_vram_used_gb = mem_info.used / (1024**3)
                stats.gpu_vram_total_gb = mem_info.total / (1024**3)
                stats.gpu_vram_percent = (
                    (mem_info.used / mem_info.total * 100)
                    if mem_info.total > 0
                    else 0.0
                )

                # GPU compute utilization
                util_rates = self._nvidia_ml.nvmlDeviceGetUtilizationRates(
                    self._gpu_handle
                )
                stats.gpu_compute_percent = util_rates.gpu
            except Exception:
                # GPU monitoring failed, leave as None
                pass

        return stats

    def should_update(self) -> bool:
        """Check if enough time has passed for a HUD update.

        Returns:
            True if HUD should be updated.
        """
        # Defensive checks for None values that might occur during engine swaps
        if self.last_update_time is None or self.update_interval is None:
            print(f"🔧 DEBUG: SystemMonitor has None values - last_update_time: {self.last_update_time}, update_interval: {self.update_interval}")
            self.last_update_time = 0.0
            self.update_interval = 15.0
            return True
        
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def format_hud(self, stats: SystemStats, style_params=None) -> Panel:
        """Format system stats as a Rich panel for display.

        Args:
            stats: System statistics to display.
            style_params: Optional style parameters.

        Returns:
            Rich Panel object for display.
        """
        # Create a table for neat alignment
        table = Table(show_header=False, box=None, padding=(0, 1))

        # Add columns
        table.add_column(style="bold cyan", min_width=15)
        table.add_column(style="white")

        # CPU row
        cpu_color = self._get_usage_color(stats.cpu_percent)
        table.add_row("CPU:", f"[{cpu_color}]{stats.cpu_percent:.1f}%[/{cpu_color}]")

        # RAM row
        ram_color = self._get_usage_color(stats.ram_percent)
        table.add_row(
            "RAM:",
            (
                f"[{ram_color}]{stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB"
                f" ({stats.ram_percent:.1f}%)[/{ram_color}]"
            ),
        )

        # GPU rows (if available)
        if stats.gpu_vram_percent is not None:
            gpu_mem_color = self._get_usage_color(stats.gpu_vram_percent)
            table.add_row(
                "GPU VRAM:",
                (
                    f"[{gpu_mem_color}]{stats.gpu_vram_used_gb:.1f}/"
                    f"{stats.gpu_vram_total_gb:.1f} GB "
                    f"({stats.gpu_vram_percent:.1f}%)[/{gpu_mem_color}]"
                ),
            )

        if stats.gpu_compute_percent is not None:
            gpu_compute_color = self._get_usage_color(stats.gpu_compute_percent)
            table.add_row(
                "GPU Compute:",
                f"[{gpu_compute_color}]{stats.gpu_compute_percent:.1f}%[/{gpu_compute_color}]",
            )

        # Context window row
        context_color = self._get_usage_color(stats.context_percent)
        remaining_tokens = stats.context_max_tokens - stats.context_used_tokens
        table.add_row(
            "Context:",
            (
                f"[{context_color}]{stats.context_used_tokens}/"
                f"{stats.context_max_tokens} tokens "
                f"({remaining_tokens} free)[/{context_color}]"
            ),
        )

        # Conversation turns row
        table.add_row(
            "Turns:",
            f"[cyan]{stats.conversation_turns} exchanges[/cyan]",
        )

        # Get style settings
        use_emoji = getattr(style_params, "use_emoji", True) if style_params else True
        border_style = (
            getattr(style_params, "status_border_style", "dim cyan")
            if style_params
            else "dim cyan"
        )
        title_style = (
            getattr(style_params, "status_title_style", "bold cyan")
            if style_params
            else "bold cyan"
        )

        emoji = "📊 " if use_emoji else ""

        return Panel(
            table,
            title=f"[{title_style}]{emoji}System Monitor[/{title_style}]",
            border_style=border_style,
            padding=(0, 1),
            expand=False,
        )

    def display_hud(self, console: Console, style_params=None):
        """Display the HUD if update interval has passed.

        Args:
            console: Rich console for output.
            style_params: Optional style parameters.
        """
        if self.should_update():
            stats = self.get_stats()
            hud_panel = self.format_hud(stats, style_params)
            console.print(hud_panel)

    def _get_usage_color(self, percent: float) -> str:
        """Get color based on usage percentage.

        Args:
            percent: Usage percentage (0-100).

        Returns:
            Color string for Rich formatting.
        """
        # Handle None values that might occur during engine swaps
        if percent is None:
            return "dim"
        
        if percent >= 90:
            return "red"
        elif percent >= 70:
            return "yellow"
        elif percent >= 50:
            return "cyan"
        else:
            return "green"
