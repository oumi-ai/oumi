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

"""Conversation renderer for creating asciinema recordings of chat playback."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class ConversationRenderer:
    """Renders conversation history as asciinema recording with step-by-step playback."""

    def __init__(
        self,
        conversation_history: list,
        console: Console,
        config: Any,
        thinking_processor: Optional[Any] = None,
    ):
        """Initialize the conversation renderer.

        Args:
            conversation_history: List of conversation messages.
            console: Rich console for output.
            config: Configuration object with style settings.
            thinking_processor: Optional thinking processor for rendering thinking content.
        """
        self.conversation_history = conversation_history
        self.console = console
        self.config = config
        self.thinking_processor = thinking_processor
        
        # Playback settings
        self.typing_delay = 0.05  # Delay between characters when "typing"
        self.message_delay = 2.0  # Delay between messages
        self.thinking_delay = 1.0  # Delay for thinking sections

    def render_to_asciinema(self, output_path: str) -> tuple[bool, str]:
        """Render the conversation to an asciinema recording.

        Args:
            output_path: Path where the .cast file should be saved.

        Returns:
            Tuple of (success, message).
        """
        try:
            # Create a temporary script that will play back the conversation
            temp_script = self._create_playback_script()
            
            # Start asciinema recording
            cmd = [
                "asciinema", "rec",
                "--overwrite",
                "--title", "Oumi Conversation Playback",
                "--command", f"python3 {temp_script}",
                output_path
            ]

            self.console.print(f"🎬 Starting asciinema recording: {output_path}")
            self.console.print("🎭 Playing back conversation step by step...")
            
            # Run asciinema recording
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up temporary script
            try:
                os.unlink(temp_script)
            except OSError:
                pass

            if result.returncode == 0:
                file_size = Path(output_path).stat().st_size
                return True, f"✅ Successfully recorded conversation to {output_path} ({file_size:,} bytes)"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return False, f"❌ Asciinema recording failed: {error_msg}"

        except subprocess.TimeoutExpired:
            return False, "❌ Recording timed out after 5 minutes"
        except Exception as e:
            return False, f"❌ Error creating recording: {str(e)}"

    def _create_playback_script(self) -> str:
        """Create a temporary Python script that plays back the conversation.

        Returns:
            Path to the temporary script file.
        """
        script_content = self._generate_playback_script_content()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            return f.name

    def _generate_playback_script_content(self) -> str:
        """Generate the Python script content for conversation playback."""
        
        # Extract conversation data for the script
        messages = []
        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Skip system and attachment messages for cleaner playback
            if role in ["user", "assistant"]:
                messages.append({
                    "role": role,
                    "content": content
                })

        # Create script content without problematic f-strings in template
        messages_json = json.dumps(messages, indent=2)
        
        script_content = f'''
import time
import sys
import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize console
console = Console()

# Conversation data
messages = {messages_json}

# Playback settings
TYPING_DELAY = {self.typing_delay}
MESSAGE_DELAY = {self.message_delay}
THINKING_DELAY = {self.thinking_delay}

def simulate_typing(text, delay=TYPING_DELAY):
    """Simulate typing by printing characters with delay."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # Final newline

def process_thinking_content(content):
    """Extract thinking content and final response."""
    import re
    
    # Look for common thinking patterns
    thinking_patterns = [
        (r'<thinking>(.*?)</thinking>', r'\\\\1'),
        (r'<think>(.*?)</think>', r'\\\\1'),
        (r'<reasoning>(.*?)</reasoning>', r'\\\\1'),
        (r'<reflection>(.*?)</reflection>', r'\\\\1'),
    ]
    
    thinking_content = ""
    final_content = content
    
    for pattern, replacement in thinking_patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            thinking_content = matches[0].strip()
            final_content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            break
    
    return thinking_content, final_content

def display_message(role, content, position=None):
    """Display a message with rich formatting."""
    if role == "user":
        title = "[bold blue]You[/bold blue]"
        border_style = "blue"
    else:
        pos_text = f" (#{{position}})" if position else ""
        title = f"[bold cyan]Assistant{{pos_text}}[/bold cyan]"
        border_style = "cyan"
        
        # Process thinking content for assistant messages
        thinking_content, final_content = process_thinking_content(content)
        
        if thinking_content:
            # Show thinking section
            thinking_panel = Panel(
                Text(thinking_content, style="dim white"),
                title="[dim yellow]🤔 Thinking[/dim yellow]",
                border_style="yellow",
                padding=(1, 2)
            )
            console.print(thinking_panel)
            console.print()
            time.sleep(THINKING_DELAY)
            content = final_content
    
    # Show main content
    panel = Panel(
        Text(content, style="white"),
        title=title,
        border_style=border_style,
        padding=(1, 2)
    )
    console.print(panel)

def main():
    """Main playback function."""
    # Clear screen and show header
    console.clear()
    
    header = Panel(
        Text("🎬 Oumi Conversation Playback", style="bold cyan", justify="center"),
        subtitle="[dim]Step-by-step conversation rendering[/dim]",
        border_style="cyan",
        padding=(1, 0)
    )
    console.print(header)
    console.print()
    
    # Count assistant messages for positioning
    assistant_count = 0
    
    # Play back each message
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        
        if role == "assistant":
            assistant_count += 1
            display_message(role, content, assistant_count)
        else:
            display_message(role, content)
        
        # Add delay between messages (except for last message)
        if i < len(messages) - 1:
            time.sleep(MESSAGE_DELAY)
    
    # Show completion message
    console.print()
    completion = Panel(
        Text("✅ Conversation playback complete!", style="bold green", justify="center"),
        border_style="green",
        padding=(0, 1)
    )
    console.print(completion)
    
    # Keep the recording open for a moment
    time.sleep(3)

if __name__ == "__main__":
    main()
'''.strip()

        return script_content

    def render_conversation_locally(self) -> None:
        """Render the conversation locally without asciinema (for testing)."""
        self.console.clear()
        
        # Show header
        header = Panel(
            Text("🎬 Oumi Conversation Playback", style="bold cyan", justify="center"),
            subtitle="[dim]Step-by-step conversation rendering[/dim]",
            border_style="cyan",
            padding=(1, 0),
            expand=getattr(self.config.style, "expand_panels", False) if self.config else False,
        )
        self.console.print(header)
        self.console.print()

        # Play back conversation
        assistant_count = 0
        for i, message in enumerate(self.conversation_history):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "user":
                self._display_user_message(content)
            elif role == "assistant":
                assistant_count += 1
                self._display_assistant_message(content, assistant_count)
            
            # Add delay between messages (except last)
            if i < len(self.conversation_history) - 1:
                time.sleep(self.message_delay)

        # Show completion
        self.console.print()
        completion = Panel(
            Text("✅ Conversation playback complete!", style="bold green", justify="center"),
            border_style="green",
            padding=(0, 1),
            expand=getattr(self.config.style, "expand_panels", False) if self.config else False,
        )
        self.console.print(completion)

    def _display_user_message(self, content: str) -> None:
        """Display a user message with formatting."""
        panel = Panel(
            Text(content, style="white"),
            title="[bold blue]You[/bold blue]",
            border_style="blue",
            padding=(1, 2),
            expand=getattr(self.config.style, "expand_panels", False) if self.config else False,
        )
        self.console.print(panel)

    def _display_assistant_message(self, content: str, position: int) -> None:
        """Display an assistant message with thinking processing."""
        # Process thinking content if processor is available
        if self.thinking_processor:
            thinking_result = self.thinking_processor.extract_thinking(content)
            
            if thinking_result.has_thinking:
                # Show thinking content
                self.thinking_processor.render_thinking(
                    thinking_result, self.console, self.config.style if self.config else None, compressed=False
                )
                self.console.print()
                time.sleep(self.thinking_delay)
                content = thinking_result.final_content

        # Show assistant response
        panel = Panel(
            Text(content, style="white"),
            title=f"[bold cyan]Assistant (#{position})[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=getattr(self.config.style, "expand_panels", False) if self.config else False,
        )
        self.console.print(panel)