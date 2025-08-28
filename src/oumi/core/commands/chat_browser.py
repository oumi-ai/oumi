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

"""Chat browser for browsing and playing back recent conversations."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from oumi.core.configs import InferenceConfig
from oumi.utils.logging import logger


class ChatBrowser:
    """Browser for recent chat conversations with playback support."""

    def __init__(self, console: Console = None, config: InferenceConfig = None):
        """Initialize chat browser with console and config."""
        # Support both old signature (config) and new signature (console)
        if console is not None:
            self.console = console
            self.config = config  # May be None for command usage
        else:
            # Legacy usage - config is the first parameter
            self.config = console if console is not None else config
            self.console = Console()

        self.cache_dir = Path.home() / ".oumi" / "chat_cache"
        self.metadata_file = self.cache_dir / "metadata.json"

    def browse_recent_chats(self) -> Optional[str]:
        """Browse recent chats and return selected chat ID for command integration."""
        # Load recent chats
        recent_chats = self._load_recent_chats()
        if not recent_chats:
            self.console.print("[yellow]No recent chats found.[/yellow]")
            return None

        # Display recent chats
        self._display_chat_list(recent_chats)

        # Get user selection
        choice = self._get_user_choice(recent_chats)
        if choice is None or choice in ["refresh", "invalid"]:
            return None
        elif isinstance(choice, int):
            # Return the chat ID for the selected chat
            chat_metadata = recent_chats[choice]
            return chat_metadata.get("id")

        return None

    def run(self) -> None:
        """Main entry point for chat browsing."""
        emoji = "📚 " if self.config and self.config.style.use_emoji else ""
        self.console.print(
            Panel(
                Text(f"{emoji}Oumi Chat Browser", style="bold cyan"),
                subtitle="[dim]Browse and play back your recent conversations[/dim]",
                border_style="cyan",
                expand=self.config.style.expand_panels if self.config else False,
            )
        )

        # Load recent chats
        recent_chats = self._load_recent_chats()
        if not recent_chats:
            self.console.print("[yellow]No recent chats found.[/yellow]")
            return

        while True:
            # Display recent chats
            self._display_chat_list(recent_chats)

            # Get user selection
            choice = self._get_user_choice(recent_chats)
            if choice is None:
                break
            elif choice == "refresh":
                recent_chats = self._load_recent_chats()
                continue
            elif isinstance(choice, int):
                # Load and play back selected chat
                chat_metadata = recent_chats[choice]
                self._play_chat(chat_metadata)

    def _load_recent_chats(self) -> list[dict[str, Any]]:
        """Load recent chat metadata from cache directory."""
        if not self.cache_dir.exists():
            return []

        chats = []
        try:
            # Load individual chat files from cache directory
            for chat_file in self.cache_dir.glob("*.json"):
                if chat_file.name == "metadata.json":
                    continue

                try:
                    with open(chat_file, encoding="utf-8") as f:
                        chat_data = json.load(f)

                    # Handle both basic and comprehensive history formats
                    if chat_data.get("format") == "oumi_conversation_history":
                        # Comprehensive history format
                        session = chat_data.get("session", {})
                        chat_id = session.get("chat_id", chat_file.stem)

                        # Get model from configuration
                        config = chat_data.get("configuration", {})
                        model_config = config.get("model", {})
                        model_name = model_config.get("model_name", "Unknown")

                        # Get conversation history from current branch
                        branches = chat_data.get("branches", {})
                        current_branch_id = session.get("current_branch_id", "main")
                        current_branch = branches.get(
                            current_branch_id, branches.get("main", {})
                        )
                        conversation_history = current_branch.get(
                            "conversation_history", []
                        )

                        created_at = chat_data.get("created_at", "")
                    else:
                        # Basic format (legacy)
                        chat_id = chat_data.get("chat_id", chat_file.stem)
                        model_name = chat_data.get("model_name", "Unknown")
                        conversation_history = chat_data.get("conversation_history", [])
                        created_at = chat_data.get(
                            "created_at", chat_data.get("last_updated", "")
                        )

                    # Get first user message as preview
                    preview = "No preview available"
                    for msg in conversation_history:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            preview = content[:100] + (
                                "..." if len(content) > 100 else ""
                            )
                            break

                    # Convert ISO timestamp to unix timestamp
                    timestamp = 0
                    if created_at:
                        try:
                            from datetime import datetime

                            dt = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                            timestamp = dt.timestamp()
                        except Exception:
                            timestamp = 0

                    chats.append(
                        {
                            "id": chat_id,
                            "model_name": model_name,
                            "message_count": len(conversation_history),
                            "preview": preview,
                            "timestamp": timestamp,
                            "file_path": str(chat_file),
                        }
                    )

                except (OSError, json.JSONDecodeError) as e:
                    logger.debug(f"Failed to load chat file {chat_file}: {e}")
                    continue

            # Sort by timestamp, most recent first
            return sorted(chats, key=lambda x: x.get("timestamp", 0), reverse=True)

        except Exception as e:
            logger.warning(f"Failed to load chat cache: {e}")
            return []

    def _display_chat_list(self, chats: list[dict[str, Any]]) -> None:
        """Display a list of recent chats."""
        table = Table(
            title="Recent Chat Conversations",
            title_style="bold cyan",
            show_edge=False,
            show_lines=True,
        )

        table.add_column("#", style="cyan", width=3)
        table.add_column("Date", style="green", width=12)
        table.add_column("Model", style="yellow", width=20)
        table.add_column("Messages", style="blue", width=8)
        table.add_column("Preview", style="white", width=50)

        for i, chat in enumerate(chats[:20]):  # Show up to 20 recent chats
            timestamp = chat.get("timestamp", 0)
            date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )

            model_name = chat.get("model_name", "Unknown")
            if "/" in model_name:
                model_name = model_name.split("/")[-1]

            message_count = chat.get("message_count", 0)
            preview = chat.get("preview", "No preview available")

            # Truncate preview if too long
            if len(preview) > 47:
                preview = preview[:47] + "..."

            table.add_row(str(i + 1), date_str, model_name, str(message_count), preview)

        self.console.print(table)


    def _get_user_choice(self, chats: list[dict[str, Any]]) -> Optional[Any]:
        """Get user's menu choice."""
        self.console.print()
        self.console.print("[dim]Commands:[/dim]")
        self.console.print(f"[dim]  1-{len(chats)}: Select chat to play back[/dim]")
        self.console.print("[dim]  r: Refresh chat list[/dim]")
        self.console.print("[dim]  q: Quit browser[/dim]")

        choice = Prompt.ask("Select option", default="q").strip().lower()

        if choice in ["q", "quit", "exit"]:
            return None
        elif choice in ["r", "refresh"]:
            return "refresh"
        else:
            try:
                chat_num = int(choice)
                if 1 <= chat_num <= len(chats):
                    return chat_num - 1  # Convert to 0-based index
                else:
                    self.console.print("[red]Invalid chat number.[/red]")
                    return "invalid"
            except ValueError:
                self.console.print("[red]Invalid choice.[/red]")
                return "invalid"


    def _play_chat(self, chat_metadata: dict[str, Any]) -> None:
        """Play back a selected chat conversation."""
        chat_id = chat_metadata.get("id")
        if not chat_id:
            self.console.print("[red]Invalid chat metadata.[/red]")
            return

        # Load the full chat
        chat_file = self.cache_dir / f"{chat_id}.json"
        if not chat_file.exists():
            self.console.print(f"[red]Chat file not found: {chat_file}[/red]")
            return

        try:
            with open(chat_file, encoding="utf-8") as f:
                chat_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Failed to load chat: {e}[/red]")
            return

        # Display chat header
        model_name = chat_metadata.get("model_name", "Unknown Model")
        timestamp = chat_metadata.get("timestamp", 0)
        date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        emoji = "🎬 " if self.config.style.use_emoji else ""
        self.console.print()
        self.console.print(
            Panel(
                f"{emoji}Playing Chat Conversation\n"
                f"Model: {model_name}\n"
                f"Date: {date_str}\n"
                f"Messages: {len(chat_data.get('conversation', []))}",
                title="Chat Playback",
                border_style="green",
                expand=self.config.style.expand_panels,
            )
        )

        # Get playback mode
        mode = self._get_playback_mode()
        if mode is None:
            return

        # Play back the conversation
        conversation = chat_data.get("conversation", [])
        if mode == "auto":
            self._auto_playback(conversation, model_name)
        else:  # step
            self._step_playback(conversation, model_name)

    def _get_playback_mode(self) -> Optional[str]:
        """Get the playback mode from user."""
        self.console.print()
        self.console.print("[dim]Playback modes:[/dim]")
        self.console.print("[dim]  a: Auto-play (continuous)[/dim]")
        self.console.print("[dim]  s: Step-through (manual)[/dim]")
        self.console.print("[dim]  b: Back to browser[/dim]")

        choice = Prompt.ask("Select playback mode", default="a").strip().lower()

        if choice in ["a", "auto"]:
            return "auto"
        elif choice in ["s", "step"]:
            return "step"
        elif choice in ["b", "back", "browser"]:
            return None
        else:
            self.console.print("[red]Invalid choice, using auto-play.[/red]")
            return "auto"

    def _auto_playback(
        self, conversation: list[dict[str, Any]], model_name: str
    ) -> None:
        """Auto-play the conversation with timing delays."""
        self.console.print()
        self.console.print(
            "[dim]Starting auto-playback... (Press Ctrl+C to stop)[/dim]"
        )

        try:
            for i, message in enumerate(conversation):
                role = message.get("role", "unknown")
                content = message.get("content", "")

                if role == "user":
                    self._display_message("You", content, "blue")
                elif role == "assistant":
                    display_name = (
                        model_name.split("/")[-1] if "/" in model_name else model_name
                    )
                    self._display_message(display_name, content, "cyan")


                # Add delay between messages (except for last message)
                if i < len(conversation) - 1:
                    time.sleep(2.0)  # 2 second delay between messages

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Playback stopped.[/yellow]")

    def _step_playback(
        self, conversation: list[dict[str, Any]], model_name: str
    ) -> None:
        """Step-through playback with manual control."""
        self.console.print()
        self.console.print("[dim]Starting step-through playback...[/dim]")
        self.console.print(
            "[dim]Commands: Enter/Space = next, q = quit, a = auto mode[/dim]"
        )

        for i, message in enumerate(conversation):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if role == "user":
                self._display_message("You", content, "blue")
            elif role == "assistant":
                display_name = (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                )
                self._display_message(display_name, content, "cyan")


            # Wait for user input (except for last message)
            if i < len(conversation) - 1:
                choice = (
                    Prompt.ask(
                        f"[dim]Message {i + 1}/{len(conversation)}[/dim]",
                        default="",
                        show_default=False,
                    )
                    .strip()
                    .lower()
                )

                if choice in ["q", "quit"]:
                    break
                elif choice in ["a", "auto"]:
                    # Switch to auto mode for remaining messages
                    self.console.print("[yellow]Switching to auto-playback...[/yellow]")
                    remaining = conversation[i + 1 :]
                    if remaining:
                        self._auto_playback(remaining, model_name)
                    break

    def _display_message(self, sender: str, content: str, style: str) -> None:
        """Display a message with consistent formatting."""
        # Process content to remove any thinking tags for cleaner playback
        from oumi.core.thinking import ThinkingProcessor

        processor = ThinkingProcessor()
        thinking_result = processor.extract_thinking(content)

        if thinking_result.has_thinking:
            # Show thinking in compressed form during playback
            processor.render_thinking(
                thinking_result, self.console, self.config.style, compressed=True
            )
            display_content = thinking_result.final_content
        else:
            display_content = content

        self.console.print(
            Panel(
                Text(display_content, style="white"),
                title=f"[{style}]{sender}[/{style}]",
                border_style=style,
                padding=(1, 2),
                expand=self.config.style.expand_panels,
            )
        )

