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

"""Command router that coordinates all command handlers."""

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.commands.handlers.branch_operations_handler import (
    BranchOperationsHandler,
)
from oumi.core.commands.handlers.conversation_operations_handler import (
    ConversationOperationsHandler,
)
from oumi.core.commands.handlers.file_operations_handler import FileOperationsHandler
from oumi.core.commands.handlers.macro_operations_handler import MacroOperationsHandler
from oumi.core.commands.handlers.model_management_handler import ModelManagementHandler
from oumi.core.commands.handlers.parameter_management_handler import (
    ParameterManagementHandler,
)


class CommandRouter:
    """Central router for all command handling.

    This class maintains a registry of command handlers and routes
    parsed commands to the appropriate handler.
    """

    def __init__(self, context: CommandContext):
        """Initialize the command router.

        Args:
            context: Shared command context.
        """
        self.context = context
        self.console = context.console
        self._handlers: dict[str, BaseCommandHandler] = {}
        self._command_to_handler: dict[str, str] = {}

        # Initialize all handlers
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize and register all command handlers."""
        handlers = [
            ("file_operations", FileOperationsHandler(self.context)),
            ("model_management", ModelManagementHandler(self.context)),
            ("conversation_operations", ConversationOperationsHandler(self.context)),
            ("branch_operations", BranchOperationsHandler(self.context)),
            ("parameter_management", ParameterManagementHandler(self.context)),
            ("macro_operations", MacroOperationsHandler(self.context)),
        ]

        # Register handlers and build command mapping
        for handler_name, handler in handlers:
            self._handlers[handler_name] = handler

            # Map each command to its handler
            for command in handler.get_supported_commands():
                self._command_to_handler[command] = handler_name

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Route a command to the appropriate handler.

        Args:
            command: The parsed command to handle.

        Returns:
            CommandResult from the appropriate handler.
        """
        # Special handling for help command
        if command.command == "help":
            return self._handle_help(command)

        # Special handling for exit command
        if command.command == "exit":
            return self._handle_exit(command)

        # Route to appropriate handler
        handler_name = self._command_to_handler.get(command.command)
        if handler_name:
            handler = self._handlers[handler_name]
            return handler.handle_command(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unknown command: {command.command}. Type /help().",
                should_continue=False,
            )

    def _handle_help(self, command: ParsedCommand) -> CommandResult:
        """Handle the /help() command to show available commands."""
        help_content = self._generate_help_content()

        from rich.markdown import Markdown

        self.console.print(Markdown(help_content))

        return CommandResult(success=True, should_continue=False)

    def _handle_exit(self, command: ParsedCommand) -> CommandResult:
        """Handle the /exit() command to exit the chat session."""
        return CommandResult(
            success=True,
            message="Goodbye! 👋",
            should_exit=True,
            should_continue=False,
        )

    def _generate_help_content(self) -> str:
        """Generate comprehensive help content."""
        help_content = f"""
## Available Commands

### Basic Commands
- **`/help()`** - Show this help message
- **`/exit()`** - Exit the interactive chat session

### Input Modes
- **`/ml`** - Switch to multi-line input mode
- **`/sl`** - Switch to single-line input mode

### File Operations
- **`/attach(path)`** - Attach files to conversation
  - Supports: images (JPG, PNG, etc.), PDFs, text files, CSV, JSON, Markdown
  - Example: `/attach(document.pdf)` or `/attach(image.jpg)`
- **`/fetch(url)`** - Fetch web content and add to conversation context
  - Retrieves and parses HTML content from websites
  - Example: `/fetch(https://example.com)` or `/fetch(docs.python.org)`
  - Requires: `pip install 'oumi[interactive]'`
- **`/shell(command)`** - Execute safe shell commands and capture output
  - Runs commands in local environment with security restrictions
  - Example: `/shell(ls -la)` or `/shell(python --version)`
  - Output (stdout/stderr) added to conversation context
  - Security: Blocks dangerous operations, 30s timeout, 200 char limit

### Conversation Management
- **`/delete()`** - Delete the previous conversation turn
- **`/regen()`** - Regenerate the last assistant response
- **`/clear()`** - Clear entire conversation history and start fresh
- **`/show(pos)`** - View a specific conversation position
  - `/show()` - Show most recent assistant message
  - `/show(3)` - Show assistant message #3
  - Shows both user and assistant messages for the specified turn
- **`/render(path)`** - Record conversation playback as asciinema recording
  - Example: `/render(conversation.cast)` - Creates animated terminal recording
  - Plays back entire conversation step-by-step with realistic timing
  - Requires asciinema: `pip install asciinema`

### Parameter Adjustment
- **`/set(param=value)`** - Adjust generation parameters
  - Examples:
    - `/set(temperature=0.8)` - More creative responses
    - `/set(top_p=0.9)` - Nucleus sampling
    - `/set(max_tokens=2048)` - Longer responses
    - `/set(sampling=true)` - Enable sampling
  - Available parameters: temperature, top_p, top_k, max_tokens, sampling, seed,
    frequency_penalty, presence_penalty, min_p, num_beams
- **`/swap(model_name)`** - Switch to a different model while preserving conversation
  - Examples:
    - `/swap(llama-3.1-8b)` - Switch to Llama 3.1 8B model
    - `/swap(anthropic:claude-3-5-sonnet-20241022)` - Switch to Claude via API
  - Note: Requires infrastructure support for dynamic model loading
- **`/list_engines()`** - List available inference engines and their supported models
  - Shows local engines (NATIVE, VLLM, LLAMACPP) and API engines
    (ANTHROPIC, OPENAI, etc.)
  - Includes sample models and API key requirements for each engine

### Import/Export
- **`/save(path)`** - Save current conversation branch to various formats
  - Formats: PDF, TXT, MD, JSON, CSV, HTML (auto-detected from extension)
  - Examples:
    - `/save(chat.pdf)` - PDF with formatting
    - `/save(chat.txt)` - Plain text
    - `/save(chat.md)` - Markdown format
    - `/save(chat.json)` - Structured JSON
    - `/save(chat.csv)` - CSV for data analysis
    - `/save(chat.html)` - HTML with styling
  - Force format: `/save(myfile, format=json)`
- **`/save_history(path)`** - Save complete conversation state (all branches + config)
  - Saves: All branches, model config, generation params, attachments, statistics
  - Format: Comprehensive JSON with full conversation tree and metadata
  - Example: `/save_history(project_complete.json)`
  - Perfect for: Collaboration, full session backup, complex branching scenarios
- **`/import(path)`** - Import conversation data from supported formats
  - Formats: JSON, CSV, Excel (.xlsx/.xls), Markdown (.md), Text (.txt)
  - Examples:
    - `/import(chat.json)` - Import from JSON format
    - `/import(data.csv)` - Import from CSV with role/content columns
    - `/import(conversation.md)` - Import from Markdown with ## User/Assistant headers
  - Automatically detects format from file extension
- **`/import_history(path)`** - Restore complete conversation state
  - Restores: All branches, model config, current branch, full session state
  - Format: Oumi comprehensive history JSON (from `/save_history()`)
  - Example: `/import_history(project_complete.json)`
  - Perfect for: Resuming complex sessions, sharing complete conversation trees

### Context Management
- **`/compact()`** - Compress conversation history to save context window space
  - Summarizes older messages while preserving recent exchanges
  - Helps when approaching context window limits
  - Shows token savings after compaction

### Conversation Branching (TMux-style)
- **`/branch()`** - Create a new conversation branch from current point
  - Fork the conversation to explore different paths
  - Maximum of 5 branches allowed
- **`/branch_from(name,pos)`** - Create a branch from specific assistant message position
  - Example: `/branch_from(experiment,2)` - Branch from assistant message #2
  - Useful for exploring alternative paths from earlier in the conversation
- **`/switch(name)`** - Switch to a different conversation branch
  - Example: `/switch(main)` or `/switch(branch_1)`
- **`/branches()`** - List all conversation branches
  - Shows branch names, creation time, and message preview
- **`/branch_delete(name)`** - Delete a conversation branch
  - Example: `/branch_delete(branch_2)`
  - Cannot delete the main branch

### Thinking Display
- **`/full_thoughts()`** - Toggle between compressed and full thinking view
  - Compressed (default): Shows brief summaries of thinking content
  - Full mode: Shows complete thinking chains and reasoning
  - Works with multiple thinking formats: GPT-OSS, <think>, <reasoning>, etc.
- **`/clear_thoughts()`** - Remove thinking content from conversation history
  - Preserves the final responses while removing all thinking/reasoning sections
  - Useful for cleaning up conversation history while keeping the actual answers
  - Works across all supported thinking formats

### Macro System
- **`/macro(path)`** - Execute Jinja template-based conversation macros
  - Load and execute pre-defined conversation templates with customizable fields
  - Examples:
    - `/macro(judge.jinja)` - Load a judgment/evaluation macro
    - `/macro(code_repair.jinja)` - Load a code debugging assistance macro
    - `/macro(macros/creative_writing.jinja)` - Load with relative path
  - Validates template syntax and context window usage
  - Interactive field collection for template variables
  - Supports multi-turn conversation macros

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
/fetch(https://docs.python.org) # Fetch web content
/shell(ls -la)             # Execute shell command safely
/set(temperature=0.7, top_p=0.9)
/save(conversation.pdf)
/save_history(complete.json)   # Save complete conversation state with all branches
/import(data.json)             # Import conversation from JSON file
/import_history(complete.json) # Restore complete conversation state
/full_thoughts()           # Toggle thinking display mode
/clear_thoughts()          # Remove thinking content from history
/compact()                 # Compress conversation history
/show()                    # Show most recent assistant message
/show(2)                   # Show assistant message #2
/render(demo.cast)         # Record conversation playback as asciinema
/branch()                  # Create a new conversation branch
/branch_from(test,3)       # Create branch from assistant message #3
/list_engines()            # Show available inference engines
/swap(llama-3.1-8b)        # Switch to different model
/macro(judge.jinja)        # Execute a judgment/evaluation macro
```

{"🎨 **Tip**: Customize appearance with style themes!" if getattr(self.context._style, "use_emoji", True) else "Tip: Customize appearance with style themes!"}
        """

        return help_content.strip()

    def get_supported_commands(self) -> list[str]:
        """Get list of all supported commands.

        Returns:
            List of all command names supported by registered handlers.
        """
        commands = ["help", "exit"]  # Built-in commands
        commands.extend(self._command_to_handler.keys())
        return sorted(commands)

    def display_command_error(self, error_message: str):
        """Display a command error message with styling."""
        error_style = getattr(self.context._style, "error_style", "red")
        self.console.print(f"❌ {error_message}", style=error_style)

    def display_command_success(self, message: str):
        """Display a command success message with styling."""
        success_style = getattr(self.context._style, "success_style", "green")
        use_emoji = getattr(self.context._style, "use_emoji", True)

        if use_emoji:
            self.console.print(f"✅ {message}", style=success_style)
        else:
            self.console.print(f"Success: {message}", style=success_style)
