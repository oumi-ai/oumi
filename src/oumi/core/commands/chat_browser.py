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
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from oumi.core.configs import InferenceConfig
from oumi.utils.logging import logger


class ChatBrowser:
    """Browser for recent chat conversations with playback support."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.console = Console()
        self.cache_dir = Path(os.path.expanduser("~")) / ".oumi" / "chats"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.tts_enabled = False
        self.tts_engine = None

    def run(self) -> None:
        """Main entry point for chat browsing."""
        emoji = "📚 " if self.config.style.use_emoji else ""
        self.console.print(
            Panel(
                Text(f"{emoji}Oumi Chat Browser", style="bold cyan"),
                subtitle="[dim]Browse and play back your recent conversations[/dim]",
                border_style="cyan",
                expand=self.config.style.expand_panels,
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
            elif choice == "tts":
                self._toggle_tts()
                continue
            elif isinstance(choice, int):
                # Load and play back selected chat
                chat_metadata = recent_chats[choice]
                self._play_chat(chat_metadata)

    def _load_recent_chats(self) -> list[dict[str, Any]]:
        """Load recent chat metadata from cache."""
        if not self.metadata_file.exists():
            return []

        try:
            with open(self.metadata_file, encoding="utf-8") as f:
                data = json.load(f)

            # Sort by timestamp, most recent first
            chats = data.get("chats", [])
            return sorted(chats, key=lambda x: x.get("timestamp", 0), reverse=True)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load chat metadata: {e}")
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

        # Show TTS status
        tts_status = "🔊 ON" if self.tts_enabled else "🔇 OFF"
        if self.config.style.use_emoji:
            self.console.print(f"TTS: {tts_status}")
        else:
            status_text = "ON" if self.tts_enabled else "OFF"
            self.console.print(f"TTS: {status_text}")

    def _get_user_choice(self, chats: list[dict[str, Any]]) -> Optional[Any]:
        """Get user's menu choice."""
        self.console.print()
        self.console.print("[dim]Commands:[/dim]")
        self.console.print(f"[dim]  1-{len(chats)}: Select chat to play back[/dim]")
        self.console.print("[dim]  r: Refresh chat list[/dim]")
        self.console.print("[dim]  t: Toggle text-to-speech[/dim]")
        self.console.print("[dim]  q: Quit browser[/dim]")

        choice = Prompt.ask("Select option", default="q").strip().lower()

        if choice in ["q", "quit", "exit"]:
            return None
        elif choice in ["r", "refresh"]:
            return "refresh"
        elif choice in ["t", "tts", "toggle"]:
            return "tts"
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

    def _toggle_tts(self) -> None:
        """Toggle text-to-speech functionality."""
        try:
            if not self.tts_enabled:
                # Try to initialize TTS
                self.console.print("[yellow]Initializing TTS engine...[/yellow]")
                self._init_tts()
                if self.tts_engine:
                    self.tts_enabled = True
                    self.console.print("[green]TTS enabled.[/green]")
                else:
                    self.console.print("[red]Failed to initialize TTS.[/red]")
            else:
                self.tts_enabled = False
                self.tts_engine = None
                self.console.print("[yellow]TTS disabled.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]TTS error: {e}[/red]")
            self.tts_enabled = False

    def _init_tts(self) -> None:
        """Initialize KittenTTS engine."""
        try:
            # Try to import and initialize KittenTTS
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load KittenTTS nano model (small enough for CPU)
            model_name = "KittenML/kitten-tts-nano-0.1"

            self.console.print(f"[dim]Loading {model_name}...[/dim]")

            # Use CPU for TTS to avoid GPU memory conflicts
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
            )

            self.tts_engine = {
                "tokenizer": tokenizer,
                "model": model,
            }

        except ImportError:
            self.console.print(
                "[red]TTS dependencies not available. Install transformers and torch.[/red]"
            )
            self.tts_engine = None
        except Exception as e:
            self.console.print(f"[red]Failed to load TTS model: {e}[/red]")
            self.tts_engine = None

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

                    # Use TTS for assistant messages if enabled
                    if self.tts_enabled and self.tts_engine:
                        self._speak_text(content)

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

                # Use TTS for assistant messages if enabled
                if self.tts_enabled and self.tts_engine:
                    self._speak_text(content)

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

    def _speak_text(self, text: str) -> None:
        """Convert text to speech using TTS engine."""
        if not self.tts_engine:
            return

        try:
            # Simple TTS implementation
            # Note: KittenTTS might have specific API requirements
            # This is a placeholder implementation

            # Clean text for TTS (remove markdown, thinking tags, etc.)
            clean_text = self._clean_text_for_tts(text)

            # Limit text length for TTS
            if len(clean_text) > 200:
                clean_text = clean_text[:197] + "..."

            if clean_text.strip():
                # Basic TTS placeholder - actual implementation would depend on KittenTTS API
                self.console.print(f"[dim]🔊 Speaking: {clean_text[:50]}...[/dim]")

                # Here you would implement actual TTS generation and playback
                # For now, just add a small delay to simulate speech time
                time.sleep(min(len(clean_text) * 0.05, 3.0))  # Max 3 seconds

        except Exception as e:
            logger.debug(f"TTS error: {e}")

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for text-to-speech processing."""
        import re

        # Remove markdown formatting
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Code
        text = re.sub(r"```[^`]*```", "", text)  # Code blocks
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Links

        # Remove special characters that don't read well
        text = re.sub(r"[#|>]", "", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()
