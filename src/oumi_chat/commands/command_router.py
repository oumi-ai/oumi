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

from oumi_chat.commands.base_handler import BaseCommandHandler, CommandResult
from oumi_chat.commands.command_context import CommandContext
from oumi_chat.commands.command_parser import ParsedCommand
from oumi_chat.commands.command_registry import COMMAND_REGISTRY
from oumi_chat.commands.handlers.branch_operations_handler import (
    BranchOperationsHandler,
)
from oumi_chat.commands.handlers.conversation_operations_handler import (
    ConversationOperationsHandler,
)
from oumi_chat.commands.handlers.file_operations_handler import FileOperationsHandler
from oumi_chat.commands.handlers.macro_operations_handler import MacroOperationsHandler
from oumi_chat.commands.handlers.model_management_handler import ModelManagementHandler
from oumi_chat.commands.handlers.parameter_management_handler import (
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

        self.context.console.print(Markdown(help_content))

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
        """Generate comprehensive help content using centralized command registry."""
        # Get all commands from the registry
        all_commands = COMMAND_REGISTRY.get_all_commands()
        total_commands = len(all_commands)

        # Build help sections dynamically from registry
        help_content = f"""
## Available Commands ({total_commands} total)

### Basic Commands
- **`/help()`** - {all_commands["help"].description}
- **`/exit()`** - {all_commands["exit"].description}

### Input Modes
- **`/ml`** - Switch to multi-line input mode
- **`/sl`** - Switch to single-line input mode

### File Operations
- **`/attach(path)`** - {all_commands["attach"].description}
  - Supports: images (JPG, PNG, etc.), PDFs, text files, CSV, JSON, Markdown
  - Example: `/attach(document.pdf)` or `/attach(image.jpg)`
- **`/fetch(url)`** - {all_commands["fetch"].description}
  - Retrieves and parses HTML content from websites
  - Example: `/fetch(https://example.com)` or `/fetch(docs.python.org)`
  - Requires: `pip install 'oumi[interactive]'`
- **`/shell(command)`** - {all_commands["shell"].description}
  - Runs commands in local environment with security restrictions
  - Example: `/shell(ls -la)` or `/shell(python --version)`
  - Output (stdout/stderr) added to conversation context
  - Security: Blocks dangerous operations, 30s timeout, 200 char limit
- **`/save(path)`** - {all_commands["save"].description}
  - Formats: PDF, TXT, MD, JSON, CSV, HTML (auto-detected from extension)
  - Examples: `/save(chat.pdf)`, `/save(chat.json)`, `/save(chat.md)`
  - Force format: `/save(myfile, format=json)`
- **`/import(path)`** - {all_commands["import"].description}
  - Formats: JSON, CSV, Excel (.xlsx/.xls), Markdown (.md), Text (.txt)
  - Examples: `/import(chat.json)`, `/import(data.csv)`, `/import(conversation.md)`
- **`/save_history(path)`** - {all_commands["save_history"].description}
  - Saves: All branches, model config, generation params, attachments, statistics
  - Format: Comprehensive JSON with full conversation tree and metadata
  - Perfect for: Collaboration, full session backup, complex branching scenarios
- **`/import_history(path)`** - {all_commands["import_history"].description}
  - Restores: All branches, model config, current branch, full session state
  - Perfect for: Resuming complex sessions, sharing complete conversation trees
- **`/load(chat_id)`** - {all_commands["load"].description}
  - Example: `/load(recent)` or `/load(session_20241201_143022)`

### Conversation Management
- **`/delete()`** - {all_commands["delete"].description}
- **`/regen()`** - {all_commands["regen"].description}
- **`/clear()`** - {all_commands["clear"].description}
- **`/show(pos)`** - {all_commands["show"].description}
  - `/show()` - Show most recent assistant message
  - `/show(3)` - Show assistant message #3
  - Shows both user and assistant messages for the specified turn
- **`/render(path)`** - {all_commands["render"].description}
  - Example: `/render(conversation.cast)` - Creates animated terminal recording
  - Plays back entire conversation step-by-step with realistic timing
  - Requires asciinema: `pip install asciinema`
- **`/compact()`** - {all_commands["compact"].description}
  - Summarizes older messages while preserving recent exchanges
  - Helps when approaching context window limits
  - Shows token savings after compaction

### Parameter Adjustment
- **`/set(param=value)`** - {all_commands["set"].description}
  - Examples: `/set(temperature=0.8)`, `/set(top_p=0.9)`, `/set(max_tokens=2048)`
  - Available parameters: temperature, top_p, top_k, max_tokens, sampling, seed,
    frequency_penalty, presence_penalty, min_p, num_beams

### Conversation Branching (TMux-style)
- **`/branch()`** - {all_commands["branch"].description}
  - Fork the conversation to explore different paths
  - Maximum of 5 branches allowed
- **`/branch_from(name,pos)`** - {all_commands["branch_from"].description}
  - Example: `/branch_from(experiment,2)` - Branch from assistant message #2
  - Useful for exploring alternative paths from earlier in the conversation
- **`/switch(name)`** - {all_commands["switch"].description}
  - Example: `/switch(main)` or `/switch(branch_1)`
- **`/branches()`** - {all_commands["branches"].description}
  - Shows branch names, creation time, and message preview
- **`/branch_delete(name)`** - {all_commands["branch_delete"].description}
  - Example: `/branch_delete(branch_2)`
  - Cannot delete the main branch

### Thinking Display
- **`/full_thoughts()`** - {all_commands["full_thoughts"].description}
  - Compressed (default): Shows brief summaries of thinking content
  - Full mode: Shows complete thinking chains and reasoning
  - Works with multiple thinking formats: GPT-OSS, <think>, <reasoning>, etc.
- **`/clear_thoughts()`** - {all_commands["clear_thoughts"].description}
  - Preserves the final responses while removing all thinking/reasoning sections
  - Useful for cleaning up conversation history while keeping the actual answers
  - Works across all supported thinking formats

### Model Management
- **`/swap(model_name)`** - {all_commands["swap"].description}
  - Examples: `/swap(llama-3.1-8b)`, `/swap(anthropic:claude-3-5-sonnet-20241022)`
  - `/swap(config:path/to/config.yaml)` - Switch using config file
  - Note: Requires infrastructure support for dynamic model loading
- **`/list_engines()`** - {all_commands["list_engines"].description}
  - Shows local engines (NATIVE, VLLM, LLAMACPP) and API engines
    (ANTHROPIC, OPENAI, etc.)
  - Includes sample models and API key requirements for each engine

### Macro System
- **`/macro(path)`** - {all_commands["macro"].description}
  - Examples: `/macro(judge.jinja)`, `/macro(code_repair.jinja)`
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

{
            "🎨 **Tip**: Customize appearance with style themes!"
            if getattr(self.context.style, "use_emoji", True)
            else "Tip: Customize appearance with style themes!"
        }
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
        error_style = getattr(self.context.style, "error_style", "red")
        self.context.console.print(f"❌ {error_message}", style=error_style)

    def display_command_success(self, message: str):
        """Display a command success message with styling."""
        success_style = getattr(self.context.style, "success_style", "green")
        use_emoji = getattr(self.context.style, "use_emoji", True)

        if use_emoji:
            self.context.console.print(f"✅ {message}", style=success_style)
        else:
            self.context.console.print(f"Success: {message}", style=success_style)
