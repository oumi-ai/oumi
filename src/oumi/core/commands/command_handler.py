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

"""Command handler for interactive Oumi inference commands."""

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from oumi.core.attachments import FileHandler, ContextWindowManager
from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.configs import InferenceConfig
from oumi.core.configs.params.style_params import StyleParams
from oumi.core.inference import BaseInferenceEngine


class CommandResult:
    """Result of executing a command.
    
    Attributes:
        success: Whether the command executed successfully.
        message: Optional message to display to the user.
        should_exit: Whether the chat session should exit.
        should_continue: Whether to continue processing (vs. skip inference).
    """
    
    def __init__(
        self,
        success: bool = True,
        message: Optional[str] = None,
        should_exit: bool = False,
        should_continue: bool = True,
    ):
        self.success = success
        self.message = message
        self.should_exit = should_exit
        self.should_continue = should_continue


class CommandHandler:
    """Handles execution of interactive commands in Oumi inference.
    
    This class processes parsed commands and executes the appropriate actions,
    such as displaying help, exiting the session, or managing conversation state.
    """
    
    def __init__(
        self,
        console: Console,
        config: InferenceConfig,
        conversation_history: list,
        inference_engine: BaseInferenceEngine,
    ):
        """Initialize the command handler.
        
        Args:
            console: Rich console for output.
            config: Inference configuration.
            conversation_history: List of conversation messages.
            inference_engine: The inference engine being used.
        """
        self.console = console
        self.config = config
        self.conversation_history = conversation_history
        self.inference_engine = inference_engine
        self._style = config.style
        
        # Initialize file attachment system
        # Try multiple possible attribute names for max context length
        max_context = None
        possible_context_attrs = [
            'model_max_length',     # Standard transformers
            'max_model_len',        # VLLM
            'max_tokens',           # Some configs
            'context_length',       # Alternative name
            'max_context_len',      # Another alternative
            'max_seq_len'          # Sequence length
        ]
        
        for attr in possible_context_attrs:
            max_context = getattr(config.model, attr, None)
            if max_context is not None:
                # Debug info - this will help us see which attribute was found
                print(f"[DEBUG] Found context length via '{attr}': {max_context}")
                break
        
        # Final fallback to 4096 if nothing found
        if max_context is None:
            max_context = 4096
            
        model_name = getattr(config.model, 'model_name', 'default')
            
        context_manager = ContextWindowManager(max_context_length=max_context, model_name=model_name)
        self.file_handler = FileHandler(context_manager)
    
    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a parsed command and return the result.
        
        Args:
            command: The parsed command to execute.
            
        Returns:
            CommandResult indicating the outcome.
        """
        try:
            # Route to specific handler based on command name
            if command.command == "help":
                return self._handle_help(command)
            elif command.command == "exit":
                return self._handle_exit(command)
            elif command.command == "attach":
                return self._handle_attach(command)
            elif command.command == "delete":
                return self._handle_delete_placeholder(command)
            elif command.command == "regen":
                return self._handle_regen_placeholder(command)
            elif command.command == "save":
                return self._handle_save_placeholder(command)
            elif command.command == "set":
                return self._handle_set_placeholder(command)
            else:
                return CommandResult(
                    success=False,
                    message=f"Unknown command: '{command.command}'",
                )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error executing command '{command.command}': {str(e)}",
            )
    
    def _handle_help(self, command: ParsedCommand) -> CommandResult:
        """Handle the /help() command."""
        help_content = self._generate_help_content()
        
        # Get style attributes with fallbacks
        title_style = getattr(self._style, 'assistant_title_style', 'bold cyan')
        border_style = getattr(self._style, 'assistant_border_style', 'cyan')
        padding = getattr(self._style, 'assistant_padding', (1, 2))
        expand = getattr(self._style, 'expand_panels', False)
        use_emoji = getattr(self._style, 'use_emoji', True)
        
        emoji = "📋 " if use_emoji else ""
        
        # Display help using styled panel
        self.console.print(Panel(
            Markdown(help_content),
            title=f"[{title_style}]{emoji}Oumi Interactive Commands[/{title_style}]",
            border_style=border_style,
            padding=padding,
            expand=expand
        ))
        
        return CommandResult(success=True, should_continue=False)
    
    def _handle_exit(self, command: ParsedCommand) -> CommandResult:
        """Handle the /exit() command."""
        # Get style attributes with fallbacks
        use_emoji = getattr(self._style, 'use_emoji', True)
        custom_theme = getattr(self._style, 'custom_theme', None)
        
        # Display goodbye message
        emoji = "👋 " if use_emoji else ""
        goodbye_style = (
            custom_theme.get("warning", "yellow")
            if custom_theme
            else "yellow"
        )
        
        self.console.print(f"\n[{goodbye_style}]{emoji}Goodbye![/{goodbye_style}]")
        
        return CommandResult(
            success=True,
            should_exit=True,
            should_continue=False
        )
    
    def _handle_attach(self, command: ParsedCommand) -> CommandResult:
        """Handle the /attach(path) command."""
        if not command.args:
            return CommandResult(
                success=False,
                message="attach command requires a file path argument",
                should_continue=False
            )
        
        file_path = command.args[0].strip()
        
        try:
            # Estimate current conversation tokens
            conversation_tokens = self._estimate_conversation_tokens()
            
            # Process the file
            attachment_result = self.file_handler.attach_file(file_path, conversation_tokens)
            
            if attachment_result.success:
                # Display attachment info
                self._display_attachment_result(attachment_result)
                
                # Add content to conversation (this will be used in the next inference)
                self._add_attachment_to_conversation(attachment_result)
                
                success_message = f"Attached {attachment_result.file_info.name}"
                if attachment_result.context_info:
                    success_message += f" - {attachment_result.context_info}"
                
                return CommandResult(
                    success=True,
                    message=success_message,
                    should_continue=False
                )
            else:
                return CommandResult(
                    success=False,
                    message=attachment_result.warning_message or f"Failed to attach {file_path}",
                    should_continue=False
                )
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return CommandResult(
                success=False,
                message=f"Error attaching file: {str(e)}\n\nTraceback:\n{error_details}",
                should_continue=False
            )
    
    def _handle_delete_placeholder(self, command: ParsedCommand) -> CommandResult:
        """Placeholder handler for /delete() command - Phase 3 implementation."""
        self._show_not_implemented_message("delete", "Phase 3")
        return CommandResult(success=False, should_continue=False)
    
    def _handle_regen_placeholder(self, command: ParsedCommand) -> CommandResult:
        """Placeholder handler for /regen() command - Phase 3 implementation."""
        self._show_not_implemented_message("regen", "Phase 3")
        return CommandResult(success=False, should_continue=False)
    
    def _handle_save_placeholder(self, command: ParsedCommand) -> CommandResult:
        """Placeholder handler for /save() command - Phase 5 implementation."""
        self._show_not_implemented_message("save", "Phase 5")
        return CommandResult(success=False, should_continue=False)
    
    def _handle_set_placeholder(self, command: ParsedCommand) -> CommandResult:
        """Placeholder handler for /set() command - Phase 4 implementation."""
        self._show_not_implemented_message("set", "Phase 4")
        return CommandResult(success=False, should_continue=False)
    
    def _show_not_implemented_message(self, command_name: str, phase: str):
        """Show a styled message for not-yet-implemented commands."""
        message = f"Command '/{command_name}()' is planned for {phase} implementation."
        
        self.console.print(Panel(
            Text(message, style="dim white"),
            title=f"[{self._style.error_title_style}]🚧 Coming Soon[/{self._style.error_title_style}]",
            border_style="yellow",
            padding=(0, 1)
        ))
    
    def _generate_help_content(self) -> str:
        """Generate the help content as Markdown."""
        help_content = f"""
## Available Commands

### Basic Commands
- **`/help()`** - Show this help message
- **`/exit()`** - Exit the interactive chat session

### Input Modes
- **`/ml`** - Switch to multi-line input mode
- **`/sl`** - Switch to single-line input mode

### File Operations *(Coming in Phase 2)*
- **`/attach(path)`** - Attach files to conversation
  - Supports: images (JPG, PNG, etc.), PDFs, text files, CSV, JSON, Markdown
  - Example: `/attach(document.pdf)` or `/attach(image.jpg)`

### Conversation Management *(Coming in Phase 3)*
- **`/delete()`** - Delete the previous conversation turn
- **`/regen()`** - Regenerate the last assistant response

### Export *(Coming in Phase 5)*  
- **`/save(path)`** - Save conversation to PDF
  - Example: `/save(chat_history.pdf)`

### Parameter Adjustment *(Coming in Phase 4)*
- **`/set(param=value)`** - Adjust generation parameters
  - Examples:
    - `/set(temperature=0.8)` - More creative responses
    - `/set(top_p=0.9)` - Nucleus sampling
    - `/set(max_tokens=2048)` - Longer responses
    - `/set(sampling=true)` - Enable sampling

## Input Modes

### Single-line Mode (Default)
- Press **Enter** to send your message
- Type `/ml` to switch to multi-line mode

### Multi-line Mode  
- Press **Enter** to add a new line
- Press **Enter** on an empty line to send
- Type `/sl` to switch back to single-line mode

## Usage Notes
- Commands must start with `/` and be the first thing in your message
- Use parentheses for arguments: `/command(arg1, arg2)`  
- Use `key=value` format for parameters: `/set(temperature=0.8)`
- Commands work in both input modes
- Commands are case-insensitive

## Examples
```
/help()
/ml                        # Switch to multi-line mode
/exit()  
/attach(my_document.pdf)
/set(temperature=0.7, top_p=0.9)
/save(conversation.pdf)
```

{f"🎨 **Tip**: You can customize the appearance with different style themes in your config!" if getattr(self._style, 'use_emoji', True) else "Tip: You can customize the appearance with different style themes in your config!"}
        """
        
        return help_content.strip()
    
    def display_command_error(self, error_message: str):
        """Display a command error message with styling."""
        error_style = getattr(self._style, 'error_style', 'red')
        error_title_style = getattr(self._style, 'error_title_style', 'bold red')
        error_border_style = getattr(self._style, 'error_border_style', 'red')
        expand = getattr(self._style, 'expand_panels', False)
        
        self.console.print(Panel(
            f"[{error_style}]{error_message}[/{error_style}]",
            title=f"[{error_title_style}]❌ Command Error[/{error_title_style}]",
            border_style=error_border_style,
            expand=expand
        ))
    
    def display_command_success(self, message: str):
        """Display a command success message with styling."""
        custom_theme = getattr(self._style, 'custom_theme', None)
        use_emoji = getattr(self._style, 'use_emoji', True)
        
        success_style = (
            custom_theme.get("success", "green")
            if custom_theme
            else "green"
        )
        
        emoji = "✅ " if use_emoji else ""
        
        self.console.print(Panel(
            f"[{success_style}]{emoji}{message}[/{success_style}]",
            title=f"[bold {success_style}]Command Success[/bold {success_style}]",
            border_style=success_style,
            padding=(0, 1)
        ))
    
    def _estimate_conversation_tokens(self) -> int:
        """Estimate the token count of the current conversation history."""
        if not self.conversation_history:
            return 0
        
        # Simple estimation: combine all conversation text
        text_content = ""
        for message in self.conversation_history:
            if isinstance(message, dict) and "content" in message:
                text_content += str(message["content"]) + "\n"
        
        # Use context manager to estimate tokens
        max_context = getattr(self.config.model, 'model_max_length', 4096)
        model_name = getattr(self.config.model, 'model_name', 'default')
        context_manager = ContextWindowManager(max_context, model_name)
        
        return context_manager.estimate_tokens(text_content)
    
    def _display_attachment_result(self, result):
        """Display the attachment result with rich formatting."""
        from oumi.core.attachments.file_handler import ProcessingStrategy
        
        # Choose appropriate emoji and color based on file type
        file_type = result.file_info.file_type.value
        emoji_map = {
            'image': '📷',
            'pdf': '📄', 
            'text': '📝',
            'csv': '📊',
            'json': '🔧',
            'markdown': '📝',
            'code': '💾',
            'unknown': '❓'
        }
        emoji = emoji_map.get(file_type, '📎')
        
        # Create display content
        size_mb = result.file_info.size_bytes / (1024 * 1024)
        content = f"{emoji} **{result.file_info.name}**\n"
        content += f"Type: {file_type.title()} ({size_mb:.2f} MB)\n"
        
        if result.context_info:
            content += f"Processing: {result.context_info}\n"
        
        if result.warning_message:
            content += f"\n⚠️  {result.warning_message}"
        
        # Choose border color based on success/warning
        border_color = "yellow" if result.warning_message else "green"
        title_style = f"bold {border_color}"
        
        self.console.print(Panel(
            content,
            title=f"[{title_style}]📎 File Attached[/{title_style}]",
            border_style=border_color,
            padding=(0, 1)
        ))
    
    def _add_attachment_to_conversation(self, result):
        """Add attachment content to conversation history for next inference."""
        # Simply add the text content to conversation history
        # This will be prepended to the next user message
        attachment_entry = {
            "role": "attachment",
            "file_name": result.file_info.name,
            "file_type": result.file_info.file_type.value,
            "text_content": result.text_content,
            "processing_strategy": result.file_info.processing_strategy.value
        }
        
        # Store in conversation history for reference
        self.conversation_history.append(attachment_entry)