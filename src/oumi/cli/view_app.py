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

"""Textual TUI application for viewing JSONL conversations."""

import json
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Static,
)

from oumi.core.types.conversation import Conversation


class ImageWidget(Container):
    """Widget to display an image using rich-pixels with loading indicator."""

    DEFAULT_CSS = """
    ImageWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $surface;
    }

    ImageWidget .image-label {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }

    ImageWidget .loading-text {
        color: $text-muted;
        text-style: italic;
    }

    ImageWidget LoadingIndicator {
        width: auto;
        height: 1;
    }
    """

    def __init__(self, image_item, **kwargs):
        super().__init__(**kwargs)
        self.image_item = image_item
        self._pixels = None
        self._label = "Loading..."
        self._loaded = False

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()
        yield Label("Loading image...", classes="loading-text", id="loading-label")

    def on_mount(self):
        """Start loading image when mounted."""
        self._load_image_async()

    @work(thread=True)
    def _load_image_async(self):
        """Load image in background thread."""
        import shutil

        from oumi.core.types.conversation import Type

        # Get terminal width for sizing
        term_size = shutil.get_terminal_size((80, 24))
        # Use 70% of terminal width, max 80 chars
        max_width = min(int(term_size.columns * 0.7), 80)

        try:
            from rich_pixels import Pixels
        except ImportError:
            self._label = "🖼️ (Install rich-pixels: pip install rich-pixels)"
            self.app.call_from_thread(self._update_display)
            return

        item = self.image_item
        try:
            if item.type == Type.IMAGE_PATH:
                path = item.content or ""
                filename = Path(path).name if path else "unknown"
                from PIL import Image

                img = Image.open(path)
                w, h = img.size
                img = self._resize_for_terminal(img, max_width)
                self._pixels = Pixels.from_image(img)
                self._label = f"🖼️ {filename} ({w}x{h})"

            elif item.type == Type.IMAGE_URL:
                url = item.content or ""
                filename = url.split("/")[-1] if "/" in url else url[:30]
                import io

                import requests
                from PIL import Image

                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                w, h = img.size
                img = self._resize_for_terminal(img, max_width)
                self._pixels = Pixels.from_image(img)
                self._label = f"🖼️ {filename} ({w}x{h})"

            elif item.type == Type.IMAGE_BINARY:
                if item.binary:
                    import io

                    from PIL import Image

                    img = Image.open(io.BytesIO(item.binary))
                    w, h = img.size
                    size_kb = len(item.binary) / 1024
                    img = self._resize_for_terminal(img, max_width)
                    self._pixels = Pixels.from_image(img)
                    self._label = f"🖼️ embedded ({w}x{h}, {size_kb:.1f} KB)"
                else:
                    self._label = "🖼️ (no binary data)"
            else:
                self._label = "🖼️ (unknown image type)"

        except Exception as e:
            self._label = f"🖼️ (Failed to load: {e})"

        self._loaded = True
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        """Update the widget display after loading."""
        self.remove_children()
        if self._pixels:
            self.mount(Static(self._pixels))
        self.mount(Label(self._label, classes="image-label"))

    def _resize_for_terminal(self, img, max_width: int = 60):
        """Resize image to fit terminal width."""
        from PIL import Image

        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            new_h = int(h * ratio)
            img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
        return img


class MessageWidget(Container):
    """Widget to display a single message in a conversation."""

    DEFAULT_CSS = """
    MessageWidget {
        width: 100%;
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
        layout: vertical;
    }

    MessageWidget.system {
        background: $surface;
        border-left: thick $warning;
    }

    MessageWidget.user {
        background: $surface;
        border-left: thick $primary;
    }

    MessageWidget.assistant {
        background: $surface;
        border-left: thick $success;
    }

    MessageWidget.tool {
        background: $surface;
        border-left: thick $secondary;
    }

    MessageWidget .role-label {
        text-style: bold;
        margin-bottom: 1;
        height: auto;
    }

    MessageWidget .message-id {
        text-style: italic;
        color: $text-muted;
        margin-bottom: 1;
        height: auto;
    }

    MessageWidget.system .role-label {
        color: $warning;
    }

    MessageWidget.user .role-label {
        color: $primary;
    }

    MessageWidget.assistant .role-label {
        color: $success;
    }

    MessageWidget.tool .role-label {
        color: $secondary;
    }

    MessageWidget Static {
        width: 100%;
        height: auto;
    }

    MessageWidget Markdown {
        width: 100%;
        height: auto;
        margin: 0;
        padding: 0;
    }
    """

    def __init__(
        self,
        role: str,
        content: str,
        raw_mode: bool = False,
        search_term: str = "",
        message_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.role = role
        self.message_content = content
        self.raw_mode = raw_mode
        self.search_term = search_term
        self.message_id = message_id
        self.add_class(role.lower())

    def _highlight_search(self, text: str) -> str:
        """Highlight search terms in text."""
        if not self.search_term:
            return text
        # Escape special regex characters in search term
        escaped = re.escape(self.search_term)
        # Use case-insensitive replacement with markup
        pattern = re.compile(f"({escaped})", re.IGNORECASE)
        return pattern.sub(r"[reverse]\1[/reverse]", text)

    def compose(self) -> ComposeResult:
        # Display role label with tool-specific formatting
        if self.role.lower() == "tool" and self.message_id:
            yield Label(f"[{self.role.upper()}] 🔧", classes="role-label")
            yield Label(f"tool_call_id: {self.message_id}", classes="message-id")
        elif self.role.lower() == "tool":
            yield Label(f"[{self.role.upper()}] 🔧", classes="role-label")
        else:
            yield Label(f"[{self.role.upper()}]", classes="role-label")

        # Format and display content
        content = self._format_tool_content() if self.role.lower() == "tool" else self.message_content

        if self.raw_mode:
            highlighted = self._highlight_search(content)
            yield Static(highlighted, markup=True)
        else:
            # For markdown mode, highlight in the source
            highlighted = self._highlight_search(content)
            yield Markdown(highlighted)

    def _format_tool_content(self) -> str:
        """Format tool message content for better readability."""
        content = self.message_content

        # Try to pretty-print JSON content for tool responses
        try:
            # Check if content looks like JSON
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                parsed = json.loads(stripped)
                # Format as code block with pretty-printed JSON
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                return f"```json\n{formatted}\n```"
        except (json.JSONDecodeError, ValueError):
            pass

        return content


class ConversationPanel(VerticalScroll):
    """Panel displaying messages in a conversation."""

    DEFAULT_CSS = """
    ConversationPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation: Optional[Conversation] = None
        self.raw_mode: bool = False
        self.show_images: bool = False
        self.search_term: str = ""

    def set_conversation(
        self,
        conversation: Conversation,
        raw_mode: bool = False,
        search_term: str = "",
        show_images: bool = False,
    ):
        """Set the conversation to display."""
        self.conversation = conversation
        self.raw_mode = raw_mode
        self.show_images = show_images
        self.search_term = search_term
        self.refresh_messages()

    def set_raw_mode(self, raw_mode: bool):
        """Set raw mode and refresh display."""
        self.raw_mode = raw_mode
        self.refresh_messages()

    def set_show_images(self, show_images: bool):
        """Set show images mode and refresh display."""
        self.show_images = show_images
        self.refresh_messages()

    def set_search_term(self, search_term: str):
        """Set search term for highlighting."""
        self.search_term = search_term
        self.refresh_messages()

    def refresh_messages(self):
        """Refresh the displayed messages."""
        self.remove_children()
        if self.conversation is None:
            self.mount(Static("No conversation loaded"))
            return

        for msg in self.conversation.messages:
            # Extract images separately if show_images is enabled
            image_items = []
            if self.show_images and isinstance(msg.content, list):
                image_items = [item for item in msg.content if item.is_image()]

            content = self._get_message_content(msg)
            widget = MessageWidget(
                role=str(msg.role),
                content=content,
                raw_mode=self.raw_mode,
                search_term=self.search_term,
                message_id=msg.id,  # Pass message ID for tool_call_id display
            )
            self.mount(widget)

            # Mount image widgets for rendered images
            if self.show_images and image_items:
                for item in image_items:
                    self.mount(ImageWidget(item))

    def _get_message_content(self, msg) -> str:
        """Extract text content from a message."""
        if isinstance(msg.content, str):
            return msg.content
        elif isinstance(msg.content, list):
            # Separate images and text for cleaner display
            images = []
            texts = []
            for item in msg.content:
                if item.is_text():
                    texts.append(item.content or "")
                elif item.is_image():
                    # Skip images here if show_images is enabled (rendered separately)
                    if not self.show_images:
                        images.append(
                            self._format_image_content(item, self.raw_mode)
                        )

            parts = []
            # Show images first, then text
            if images:
                parts.extend(images)
            if texts:
                parts.extend(texts)
            return "\n\n".join(parts)
        return str(msg.content)

    def _format_image_content(self, item, raw_mode: bool = False) -> str:
        """Format image content item for display (text placeholder)."""
        from oumi.core.types.conversation import Type

        if item.type == Type.IMAGE_PATH:
            path = item.content or ""
            filename = Path(path).name if path else "unknown"
            if raw_mode:
                return f"🖼️ IMAGE: {path}"
            return f"```\n🖼️ IMAGE: {filename}\n```"
        elif item.type == Type.IMAGE_URL:
            url = item.content or ""
            try:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                filename = parsed.path.split("/")[-1] if parsed.path else ""
            except Exception:
                filename = ""
            if raw_mode:
                return f"🖼️ IMAGE: {url}"
            if filename:
                return f"```\n🖼️ IMAGE: {filename}\n```"
            return f"```\n🖼️ IMAGE: {url[:60]}{'...' if len(url) > 60 else ''}\n```"
        elif item.type == Type.IMAGE_BINARY:
            size_info = ""
            if item.binary:
                size_kb = len(item.binary) / 1024
                size_info = f" ({size_kb:.1f} KB)"
            if raw_mode:
                return f"🖼️ IMAGE: <binary data{size_info}>"
            return f"```\n🖼️ IMAGE: embedded{size_info}\n```"
        else:
            return f"```\n🖼️ IMAGE: {item.content or 'embedded'}\n```"


class MetadataPanel(Static):
    """Panel displaying conversation metadata."""

    DEFAULT_CSS = """
    MetadataPanel {
        width: 100%;
        height: auto;
        max-height: 8;
        padding: 1;
        background: $surface;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation: Optional[Conversation] = None

    def set_conversation(self, conversation: Conversation, index: int, total: int):
        """Set the conversation to display metadata for."""
        self.conversation = conversation
        self.refresh_metadata(index, total)

    def refresh_metadata(self, index: int, total: int):
        """Refresh the displayed metadata."""
        if self.conversation is None:
            self.update("No conversation loaded")
            return

        lines = [f"Conversation {index + 1} of {total}"]

        if self.conversation.conversation_id:
            lines.append(f"ID: {self.conversation.conversation_id}")

        lines.append(f"Messages: {len(self.conversation.messages)}")

        # Count by role and count images
        role_counts: dict[str, int] = {}
        image_count = 0
        for msg in self.conversation.messages:
            role = str(msg.role)
            role_counts[role] = role_counts.get(role, 0) + 1
            # Count images in this message
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.is_image():
                        image_count += 1
        role_str = " | ".join(f"{r}: {c}" for r, c in sorted(role_counts.items()))
        lines.append(f"Roles: {role_str}")

        # Show image count if there are images (vision-language dataset)
        if image_count > 0:
            lines.append(f"Images: {image_count} 🖼️")

        if self.conversation.metadata:
            # Skip synth_question/synth_answer as they're shown in conversation
            skip_keys = {"synth_question", "synth_answer"}
            filtered = {
                k: v
                for k, v in self.conversation.metadata.items()
                if k not in skip_keys
            }
            if filtered:
                lines.append("Metadata:")
                for k, v in filtered.items():
                    v_str = str(v)
                    # Only truncate very long values
                    if len(v_str) > 200:
                        v_str = v_str[:200] + "..."
                    lines.append(f"  {k}: {v_str}")

        self.update("\n".join(lines))


class SearchBar(Horizontal):
    """Search bar widget."""

    DEFAULT_CSS = """
    SearchBar {
        width: 100%;
        height: 3;
        padding: 0 1;
        background: $surface;
        display: none;
    }

    SearchBar.visible {
        display: block;
    }

    SearchBar Input {
        width: 1fr;
    }

    SearchBar Label {
        width: auto;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Search:")
        yield Input(placeholder="Enter search term...", id="search-input")


class GotoBar(Horizontal):
    """Go-to conversation bar widget."""

    DEFAULT_CSS = """
    GotoBar {
        width: 100%;
        height: 3;
        padding: 0 1;
        background: $surface;
        display: none;
    }

    GotoBar.visible {
        display: block;
    }

    GotoBar Input {
        width: 1fr;
    }

    GotoBar Label {
        width: auto;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Go to #:")
        yield Input(placeholder="Enter conversation number...", id="goto-input")


class HelpScreen(ModalScreen):
    """Modal screen showing keybindings help."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    HelpScreen .help-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $primary;
    }

    HelpScreen .help-section {
        text-style: bold;
        margin-top: 1;
        color: $primary;
    }

    HelpScreen .help-item {
        padding-left: 2;
    }

    HelpScreen Button {
        margin-top: 1;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("?", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Keyboard Shortcuts", classes="help-title")

            yield Label("Navigation", classes="help-section")
            yield Label("←/h      Previous conversation", classes="help-item")
            yield Label("→/l      Next conversation", classes="help-item")
            yield Label("g        First conversation", classes="help-item")
            yield Label("G        Last conversation", classes="help-item")
            yield Label(":        Go to conversation #", classes="help-item")

            yield Label("Scrolling", classes="help-section")
            yield Label("j/↓      Scroll down", classes="help-item")
            yield Label("k/↑      Scroll up", classes="help-item")
            yield Label("PgUp     Page up", classes="help-item")
            yield Label("PgDn     Page down", classes="help-item")

            yield Label("Search", classes="help-section")
            yield Label("/        Search conversations", classes="help-item")
            yield Label("n        Next match", classes="help-item")
            yield Label("N        Previous match", classes="help-item")
            yield Label("Esc      Clear search", classes="help-item")

            yield Label("Other", classes="help-section")
            yield Label("r        Toggle raw/markdown", classes="help-item")
            yield Label("i        Toggle image details", classes="help-item")
            yield Label("R        Random conversation", classes="help-item")
            yield Label("c/y      Copy conversation", classes="help-item")
            yield Label("s        Show statistics", classes="help-item")
            yield Label("?        Show this help", classes="help-item")
            yield Label("q/Esc    Quit", classes="help-item")

            yield Button("Close", variant="primary", id="help-close")

    @on(Button.Pressed, "#help-close")
    def on_close(self, event: Button.Pressed) -> None:
        self.dismiss()


class StatsScreen(ModalScreen):
    """Modal screen showing conversation statistics."""

    DEFAULT_CSS = """
    StatsScreen {
        align: center middle;
    }

    StatsScreen > Vertical {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    StatsScreen .stats-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $primary;
    }

    StatsScreen .stats-section {
        text-style: bold;
        margin-top: 1;
        color: $primary;
    }

    StatsScreen .stats-item {
        padding-left: 2;
    }

    StatsScreen .stats-grid {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        height: auto;
        padding: 1;
    }

    StatsScreen Button {
        margin-top: 1;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("s", "dismiss", "Close"),
    ]

    def __init__(self, conversations: list[Conversation], file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.conversations = conversations
        self.file_path = file_path

    def compose(self) -> ComposeResult:
        stats = self._compute_stats()

        with Vertical():
            yield Label("Conversation Statistics", classes="stats-title")
            yield Label(f"File: {self.file_path}", classes="stats-item")

            yield Label("Overview", classes="stats-section")
            yield Label(
                f"Total conversations: {stats['total_conversations']}",
                classes="stats-item",
            )
            yield Label(
                f"Total messages: {stats['total_messages']}", classes="stats-item"
            )
            yield Label(
                f"Avg messages/conversation: {stats['avg_messages']:.1f}",
                classes="stats-item",
            )
            yield Label(
                f"Min messages: {stats['min_messages']}", classes="stats-item"
            )
            yield Label(
                f"Max messages: {stats['max_messages']}", classes="stats-item"
            )

            yield Label("Messages by Role", classes="stats-section")
            for role, count in sorted(stats["role_counts"].items()):
                pct = (count / stats["total_messages"] * 100) if stats["total_messages"] > 0 else 0
                yield Label(f"{role}: {count} ({pct:.1f}%)", classes="stats-item")

            yield Label("Content Statistics", classes="stats-section")
            yield Label(
                f"Total characters: {stats['total_chars']:,}", classes="stats-item"
            )
            yield Label(
                f"Avg chars/message: {stats['avg_chars']:.0f}", classes="stats-item"
            )

            # Image statistics for vision-language datasets
            if stats["total_images"] > 0:
                yield Label("Vision-Language Statistics", classes="stats-section")
                yield Label(
                    f"Total images: {stats['total_images']}", classes="stats-item"
                )
                yield Label(
                    f"Conversations with images: {stats['conversations_with_images']}",
                    classes="stats-item",
                )
                avg_images = stats["total_images"] / stats["conversations_with_images"] if stats["conversations_with_images"] > 0 else 0
                yield Label(
                    f"Avg images/conversation: {avg_images:.1f}", classes="stats-item"
                )

            if stats["conversations_with_metadata"] > 0:
                yield Label("Metadata", classes="stats-section")
                yield Label(
                    f"Conversations with metadata: {stats['conversations_with_metadata']}",
                    classes="stats-item",
                )
                if stats["metadata_keys"]:
                    keys_str = ", ".join(list(stats["metadata_keys"])[:10])
                    if len(stats["metadata_keys"]) > 10:
                        keys_str += f" (+{len(stats['metadata_keys']) - 10} more)"
                    yield Label(f"Metadata keys: {keys_str}", classes="stats-item")

            yield Button("Close", variant="primary", id="stats-close")

    def _compute_stats(self) -> dict:
        """Compute statistics from conversations."""
        total_messages = 0
        total_chars = 0
        total_images = 0
        conversations_with_images = 0
        role_counts: Counter[str] = Counter()
        message_counts = []
        conversations_with_metadata = 0
        metadata_keys: set[str] = set()

        for conv in self.conversations:
            msg_count = len(conv.messages)
            message_counts.append(msg_count)
            total_messages += msg_count
            conv_has_images = False

            if conv.metadata:
                conversations_with_metadata += 1
                metadata_keys.update(conv.metadata.keys())

            for msg in conv.messages:
                role_counts[str(msg.role)] += 1
                if isinstance(msg.content, str):
                    total_chars += len(msg.content)
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if item.is_text() and item.content:
                            total_chars += len(item.content)
                        elif item.is_image():
                            total_images += 1
                            conv_has_images = True

            if conv_has_images:
                conversations_with_images += 1

        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "avg_messages": (
                total_messages / len(self.conversations)
                if self.conversations
                else 0
            ),
            "min_messages": min(message_counts) if message_counts else 0,
            "max_messages": max(message_counts) if message_counts else 0,
            "role_counts": dict(role_counts),
            "total_chars": total_chars,
            "avg_chars": total_chars / total_messages if total_messages else 0,
            "total_images": total_images,
            "conversations_with_images": conversations_with_images,
            "conversations_with_metadata": conversations_with_metadata,
            "metadata_keys": metadata_keys,
        }

    @on(Button.Pressed, "#stats-close")
    def on_close(self, event: Button.Pressed) -> None:
        self.dismiss()


class LoadingScreen(ModalScreen):
    """Modal screen shown while loading conversations."""

    BINDINGS = [
        Binding("escape", "quit_app", "Quit"),
        Binding("q", "quit_app", "Quit"),
    ]

    def action_quit_app(self) -> None:
        """Quit the application."""
        self.app.exit()

    DEFAULT_CSS = """
    LoadingScreen {
        align: center middle;
    }

    LoadingScreen > Vertical {
        width: 50;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 2;
        align: center middle;
    }

    LoadingScreen .loading-title {
        text-align: center;
        text-style: bold;
        padding: 1;
    }

    LoadingScreen .loading-status {
        text-align: center;
        padding: 1;
    }

    LoadingScreen LoadingIndicator {
        width: 100%;
        height: 3;
    }
    """

    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
        self._status = "Initializing..."

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Loading Conversations", classes="loading-title")
            yield LoadingIndicator()
            yield Label(self._status, id="loading-status", classes="loading-status")

    def update_status(self, status: str):
        """Update the loading status text."""
        self._status = status
        try:
            label = self.query_one("#loading-status", Label)
            label.update(status)
        except Exception:
            pass


class ConversationViewerApp(App):
    """A Textual app to browse JSONL conversation files."""

    TITLE = "Oumi Conversation Viewer"
    SUB_TITLE = "Browse conversations interactively"

    CSS = """
    #metadata-panel {
        dock: top;
    }

    #conversation-panel {
        width: 100%;
        height: 1fr;
    }

    #status-bar {
        dock: bottom;
        width: 100%;
        height: 1;
        background: $primary-background;
        padding: 0 1;
    }

    #search-bar {
        dock: top;
    }

    #goto-bar {
        dock: top;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("right", "next_conversation", "Next", show=True, priority=True),
        Binding("left", "prev_conversation", "Prev", show=True, priority=True),
        Binding("j", "scroll_down", "Scroll ↓", priority=True),
        Binding("k", "scroll_up", "Scroll ↑", priority=True),
        Binding("down", "scroll_down", "Scroll ↓", priority=True),
        Binding("up", "scroll_up", "Scroll ↑", priority=True),
        Binding("l", "next_conversation", "Next", priority=True),
        Binding("h", "prev_conversation", "Previous", priority=True),
        Binding("g", "first_conversation", "First", priority=True),
        Binding("G", "last_conversation", "Last", priority=True),
        Binding("pageup", "page_up", "Page Up", priority=True),
        Binding("pagedown", "page_down", "Page Down", priority=True),
        Binding("/", "search", "Search", show=True, priority=True),
        Binding("n", "search_next", "Next Match", priority=True),
        Binding("N", "search_prev", "Previous Match", priority=True),
        Binding("escape", "cancel_search", "Cancel", priority=True),
        Binding("r", "toggle_raw_mode", "Raw", show=True, priority=True),
        Binding("colon", "goto_conversation", "Go to #", show=True, priority=True),
        Binding("question_mark", "show_help", "Help", show=True, priority=True),
        Binding("c", "copy_conversation", "Copy", show=True, priority=True),
        Binding("y", "copy_conversation", "Copy", priority=True),
        Binding("s", "show_stats", "Stats", show=True, priority=True),
        Binding("R", "random_conversation", "Random", show=True, priority=True),
        Binding("i", "toggle_show_images", "Images", show=True, priority=True),
    ]

    def __init__(
        self,
        file_path: str,
        start_index: int = 0,
        from_stdin: bool = False,
    ):
        super().__init__()
        self.theme = "flexoki"
        self.file_path = file_path
        self.from_stdin = from_stdin
        self.conversations: list[Conversation] = []
        self.current_index = start_index
        self.search_term = ""
        self.search_matches: list[int] = []
        self.search_match_index = 0
        self.raw_mode = False
        self.show_images = False
        self._loading_screen: Optional[LoadingScreen] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield SearchBar(id="search-bar")
        yield GotoBar(id="goto-bar")
        yield MetadataPanel(id="metadata-panel")
        yield ConversationPanel(id="conversation-panel")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self):
        """Load conversations when the app mounts."""
        self._loading_screen = LoadingScreen(self.file_path)
        self.push_screen(self._loading_screen)
        self.load_conversations_async()

    @work(thread=True)
    def load_conversations_async(self):
        """Load conversations in a background thread."""
        self.load_conversations()
        self.call_from_thread(self._on_load_complete)

    def _on_load_complete(self):
        """Called when loading is complete."""
        if self._loading_screen:
            self.pop_screen()
            self._loading_screen = None

        if self.conversations:
            if self.current_index >= len(self.conversations):
                self.current_index = len(self.conversations) - 1
            self.show_conversation(self.current_index)
        self.update_status()

        # Set focus to conversation panel so key bindings work
        try:
            panel = self.query_one("#conversation-panel", ConversationPanel)
            panel.focus()
        except Exception:
            pass

    def _update_loading_status(self, status: str):
        """Update loading screen status from any thread."""
        if self._loading_screen:
            try:
                self.call_from_thread(self._loading_screen.update_status, status)
            except Exception:
                # App may not be running (e.g., in tests)
                pass

    def load_conversations(self):
        """Load conversations from the JSONL file."""
        path = Path(self.file_path)
        try:
            with open(path) as f:
                lines = f.readlines()

            total_lines = len(lines)
            for line_num, line in enumerate(lines, 1):
                # Update progress every 100 lines
                if line_num % 100 == 0 or line_num == total_lines:
                    self._update_loading_status(
                        f"Loading: {line_num}/{total_lines} lines"
                    )

                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    conv_data = self._extract_conversation_data(data)
                    if conv_data:
                        conv = Conversation.from_dict(conv_data)
                        self.conversations.append(conv)
                except Exception as e:
                    # Skip malformed lines but log them
                    try:
                        self.call_from_thread(
                            self.notify,
                            f"Line {line_num}: {e}",
                            severity="warning",
                            timeout=2,
                        )
                    except Exception:
                        # App may not be running (e.g., in tests)
                        pass

            self._update_loading_status(
                f"Loaded {len(self.conversations)} conversations"
            )
        except Exception as e:
            try:
                self.call_from_thread(
                    self.notify, f"Error loading file: {e}", severity="error"
                )
            except Exception:
                # App may not be running (e.g., in tests)
                pass

    def _extract_conversation_data(self, data: dict) -> Optional[dict]:
        """Extract conversation data from various JSONL formats.

        Supports:
        - Direct format: {"messages": [...], "metadata": {...}}
        - Synth format: {"synth_conversation": {"messages": [...]}, ...}
        - Nested conversation field: {"conversation": {"messages": [...]}}
        - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
        """
        # Direct format with messages at top level
        if "messages" in data:
            return data

        # Synth output format (synth_conversation field)
        if "synth_conversation" in data:
            conv_data = data["synth_conversation"]
            # Optionally add synth_question/synth_answer to metadata
            if isinstance(conv_data, dict):
                if "metadata" not in conv_data:
                    conv_data["metadata"] = {}
                if "synth_question" in data:
                    conv_data["metadata"]["synth_question"] = data["synth_question"]
                if "synth_answer" in data:
                    conv_data["metadata"]["synth_answer"] = data["synth_answer"]
            return conv_data

        # Generic nested conversation field
        if "conversation" in data:
            return data["conversation"]

        # If data has role/content, treat as single message
        if "role" in data and "content" in data:
            return {"messages": [data]}

        # Alpaca format: instruction, input, output
        if "instruction" in data and "output" in data:
            return self._convert_alpaca_to_conversation(data)

        # DPO format: prompt, chosen, rejected
        if "prompt" in data and "chosen" in data and "rejected" in data:
            return self._convert_dpo_to_conversation(data)

        return None

    def _convert_alpaca_to_conversation(self, data: dict) -> dict:
        """Convert Alpaca format to conversation format.

        Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
        Converts to: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        """
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output = data.get("output", "")

        # Combine instruction and input for user message
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

        # Preserve any additional fields as metadata
        metadata = {}
        for key, value in data.items():
            if key not in ("instruction", "input", "output"):
                metadata[key] = value

        result: dict = {"messages": messages}
        if metadata:
            result["metadata"] = metadata

        return result

    def _convert_dpo_to_conversation(self, data: dict) -> dict:
        """Convert DPO format to conversation format.

        DPO format: {"prompt": "...", "chosen": [...], "rejected": [...], "images": [...]}
        Converts to a conversation showing the prompt, chosen response, and rejected response.
        """
        prompt = data.get("prompt", "")
        chosen = data.get("chosen", [])
        rejected = data.get("rejected", [])
        images = data.get("images", [])

        messages = []

        # Build user message content
        user_content: list | str
        if images:
            # Vision-language DPO: include images in the user message
            content_items = []
            for img_url in images:
                content_items.append({"type": "image_url", "content": img_url})
            content_items.append({"type": "text", "content": prompt})
            user_content = content_items
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        # Add chosen response(s) with label
        for msg in chosen:
            content = msg.get("content", "")
            labeled_content = f"✅ **[CHOSEN]**\n\n{content}"
            messages.append({"role": "assistant", "content": labeled_content})

        # Add rejected response(s) with label
        for msg in rejected:
            content = msg.get("content", "")
            labeled_content = f"❌ **[REJECTED]**\n\n{content}"
            messages.append({"role": "assistant", "content": labeled_content})

        # Preserve any additional fields as metadata
        metadata = {"format": "dpo"}
        for key, value in data.items():
            if key not in ("prompt", "chosen", "rejected", "images"):
                metadata[key] = value

        return {"messages": messages, "metadata": metadata}

    def show_conversation(self, index: int):
        """Display a specific conversation."""
        if not self.conversations:
            return

        index = max(0, min(index, len(self.conversations) - 1))
        self.current_index = index

        conv = self.conversations[index]
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.set_conversation(
            conv,
            raw_mode=self.raw_mode,
            search_term=self.search_term,
            show_images=self.show_images,
        )

        metadata = self.query_one("#metadata-panel", MetadataPanel)
        metadata.set_conversation(conv, index, len(self.conversations))

        # Scroll to top
        panel.scroll_home()
        self.update_status()

    def update_status(self):
        """Update the status bar."""
        status = self.query_one("#status-bar", Static)
        if not self.conversations:
            display_path = "<stdin>" if self.from_stdin else self.file_path
            status.update(f"No conversations found in {display_path}")
            return

        mode = "RAW" if self.raw_mode else "MD"
        display_path = "<stdin>" if self.from_stdin else self.file_path
        msg = f"[{self.current_index + 1}/{len(self.conversations)}] [{mode}]"
        if self.show_images:
            msg += " [IMG]"
        msg += f" {display_path}"
        if self.search_term:
            msg += f" | Search: '{self.search_term}' ({len(self.search_matches)} matches)"
        status.update(msg)

    def action_next_conversation(self):
        """Go to the next conversation."""
        if self.current_index < len(self.conversations) - 1:
            self.show_conversation(self.current_index + 1)

    def action_prev_conversation(self):
        """Go to the previous conversation."""
        if self.current_index > 0:
            self.show_conversation(self.current_index - 1)

    def action_first_conversation(self):
        """Go to the first conversation."""
        self.show_conversation(0)

    def action_last_conversation(self):
        """Go to the last conversation."""
        self.show_conversation(len(self.conversations) - 1)

    def action_random_conversation(self):
        """Go to a random conversation."""
        if len(self.conversations) > 1:
            # Avoid selecting the same conversation
            new_index = self.current_index
            while new_index == self.current_index:
                new_index = random.randint(0, len(self.conversations) - 1)
            self.show_conversation(new_index)

    def action_scroll_down(self):
        """Scroll down in the conversation panel."""
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.scroll_down(animate=False)

    def action_scroll_up(self):
        """Scroll up in the conversation panel."""
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.scroll_up(animate=False)

    def action_page_up(self):
        """Page up in the conversation panel."""
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.scroll_page_up(animate=False)

    def action_page_down(self):
        """Page down in the conversation panel."""
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.scroll_page_down(animate=False)

    def action_toggle_raw_mode(self):
        """Toggle between markdown and raw text mode."""
        self.raw_mode = not self.raw_mode
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.set_raw_mode(self.raw_mode)
        self.update_status()

    def action_toggle_show_images(self):
        """Toggle image info display (loads images to show dimensions)."""
        self.show_images = not self.show_images
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.set_show_images(self.show_images)
        self.update_status()

    def action_search(self):
        """Show the search bar."""
        # Hide goto bar if visible
        goto_bar = self.query_one("#goto-bar", GotoBar)
        goto_bar.remove_class("visible")

        search_bar = self.query_one("#search-bar", SearchBar)
        search_bar.add_class("visible")
        search_input = self.query_one("#search-input", Input)
        search_input.focus()

    def action_cancel_search(self):
        """Hide the search bar and goto bar, or quit if nothing is active."""
        search_bar = self.query_one("#search-bar", SearchBar)
        goto_bar = self.query_one("#goto-bar", GotoBar)

        # If search or goto bar is visible, close them
        if search_bar.has_class("visible") or goto_bar.has_class("visible"):
            search_bar.remove_class("visible")
            goto_bar.remove_class("visible")
            self.search_term = ""
            self.search_matches = []
            # Clear highlighting
            panel = self.query_one("#conversation-panel", ConversationPanel)
            panel.set_search_term("")
            self.update_status()
        else:
            # Nothing active, quit the app
            self.exit()

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted):
        """Handle search submission."""
        self.search_term = event.value
        self.perform_search()
        search_bar = self.query_one("#search-bar", SearchBar)
        search_bar.remove_class("visible")

    def perform_search(self):
        """Search for conversations containing the search term."""
        self.search_matches = []
        if not self.search_term:
            return

        term_lower = self.search_term.lower()
        for i, conv in enumerate(self.conversations):
            for msg in conv.messages:
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    content = " ".join(
                        item.content or "" for item in msg.content if item.is_text()
                    )
                if term_lower in content.lower():
                    self.search_matches.append(i)
                    break

        self.search_match_index = 0
        if self.search_matches:
            self.show_conversation(self.search_matches[0])
            self.notify(f"Found {len(self.search_matches)} matching conversations")
        else:
            self.notify("No matches found", severity="warning")

        # Update highlighting in current conversation
        panel = self.query_one("#conversation-panel", ConversationPanel)
        panel.set_search_term(self.search_term)
        self.update_status()

    def action_search_next(self):
        """Go to the next search match."""
        if not self.search_matches:
            return
        self.search_match_index = (self.search_match_index + 1) % len(
            self.search_matches
        )
        self.show_conversation(self.search_matches[self.search_match_index])

    def action_search_prev(self):
        """Go to the previous search match."""
        if not self.search_matches:
            return
        self.search_match_index = (self.search_match_index - 1) % len(
            self.search_matches
        )
        self.show_conversation(self.search_matches[self.search_match_index])

    def action_goto_conversation(self):
        """Show the go-to bar."""
        # Hide search bar if visible
        search_bar = self.query_one("#search-bar", SearchBar)
        search_bar.remove_class("visible")

        goto_bar = self.query_one("#goto-bar", GotoBar)
        goto_bar.add_class("visible")
        goto_input = self.query_one("#goto-input", Input)
        goto_input.value = ""
        goto_input.focus()

    @on(Input.Submitted, "#goto-input")
    def on_goto_submitted(self, event: Input.Submitted):
        """Handle go-to submission."""
        goto_bar = self.query_one("#goto-bar", GotoBar)
        goto_bar.remove_class("visible")

        try:
            index = int(event.value) - 1  # Convert to 0-based
            if 0 <= index < len(self.conversations):
                self.show_conversation(index)
            else:
                self.notify(
                    f"Invalid index. Must be between 1 and {len(self.conversations)}",
                    severity="warning",
                )
        except ValueError:
            self.notify("Please enter a valid number", severity="warning")

    def action_show_help(self):
        """Show the help screen."""
        self.push_screen(HelpScreen())

    def action_show_stats(self):
        """Show the statistics screen."""
        if not self.conversations:
            self.notify("No conversations loaded", severity="warning")
            return
        self.push_screen(StatsScreen(self.conversations, self.file_path))

    def action_copy_conversation(self):
        """Copy the current conversation to clipboard in JSON format."""
        if not self.conversations:
            self.notify("No conversation to copy", severity="warning")
            return

        conv = self.conversations[self.current_index]

        # Convert to JSON format
        text = json.dumps(conv.to_dict(), indent=2, ensure_ascii=False)

        # Try to copy to clipboard using different methods
        try:
            import subprocess
            import sys

            if sys.platform == "darwin":
                # macOS
                process = subprocess.Popen(
                    ["pbcopy"], stdin=subprocess.PIPE, text=True
                )
                process.communicate(input=text)
                self.notify("Conversation copied to clipboard")
            elif sys.platform.startswith("linux"):
                # Linux - try xclip or xsel
                try:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                        text=True,
                    )
                    process.communicate(input=text)
                    self.notify("Conversation copied to clipboard")
                except FileNotFoundError:
                    try:
                        process = subprocess.Popen(
                            ["xsel", "--clipboard", "--input"],
                            stdin=subprocess.PIPE,
                            text=True,
                        )
                        process.communicate(input=text)
                        self.notify("Conversation copied to clipboard")
                    except FileNotFoundError:
                        # Fallback to temp file
                        self._save_to_temp_file(text)
            else:
                # Windows or other - try pyperclip or fallback
                try:
                    import pyperclip

                    pyperclip.copy(text)
                    self.notify("Conversation copied to clipboard")
                except ImportError:
                    self._save_to_temp_file(text)
        except Exception as e:
            self._save_to_temp_file(text)

    def _save_to_temp_file(self, text: str):
        """Save text to a temp file as fallback for clipboard."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(text)
            self.notify(f"Saved to {f.name}")
