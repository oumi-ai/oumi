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

import re
import threading
import time
from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.commands import CommandParser
from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_router import CommandRouter
from oumi.core.commands.utilities import make_safe
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.input import EnhancedInput, InputAction
from oumi.core.monitoring import SystemMonitor
from oumi.core.thinking import ThinkingProcessor
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.logging import logger


def _get_chars_per_token_ratio(model_name: str) -> float:
    """Get model-specific character-to-token ratio for estimation.

    Args:
        model_name: Name of the model.

    Returns:
        Estimated characters per token for the model.
    """
    model_name_lower = model_name.lower() if model_name else ""

    if "gpt" in model_name_lower or "claude" in model_name_lower:
        return 3.7  # GPT/Claude models typically have ~3.5-4 chars per token
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        return 4.0  # Llama/Mistral models typically have ~3.8-4.2 chars per token
    elif "qwen" in model_name_lower:
        return 3.5  # Qwen models are more efficient with multilingual text
    else:
        return 4.0  # Default conservative estimate


@contextmanager
def thinking_with_monitor(
    console: Console,
    system_monitor: SystemMonitor,
    status_message: str = "Thinking...",
    style: str = "cyan",
    update_interval: float = 2.0,
):
    """Context manager that shows thinking animation with live system monitor updates.

    Args:
        console: Rich console for output.
        system_monitor: System monitor to update during thinking.
        status_message: Message to display with spinner.
        style: Style for the status message.
        update_interval: Seconds between system monitor updates.
    """
    # Create layout with spinner and system monitor
    layout = Layout()
    layout.split_column(Layout(name="status", size=3), Layout(name="monitor", size=8))

    # Initialize spinner and system stats
    spinner = Spinner("dots", text=f"[{style}]{status_message}[/{style}]")

    # Flag to stop the update thread
    stop_updating = threading.Event()

    def update_display():
        """Update the system monitor display in a separate thread."""
        while not stop_updating.is_set():
            try:
                # Force update of system stats
                system_monitor.last_update_time = 0  # Force update
                stats = system_monitor.get_stats()
                monitor_panel = system_monitor.format_hud(stats)

                # Update layout
                layout["status"].update(spinner)
                layout["monitor"].update(monitor_panel)

                time.sleep(update_interval)
            except Exception as e:
                # Log the exception for debugging instead of silently passing
                logger.warning(f"Display update failed: {e}")
                # Continue anyway to avoid crashing the display thread
                pass

    # Start the update thread
    update_thread = threading.Thread(target=update_display, daemon=True)
    update_thread.start()

    # Use Live to display the animated layout
    with Live(layout, console=console, refresh_per_second=4):
        try:
            yield
        finally:
            # Stop the update thread
            stop_updating.set()
            # Wait briefly for thread to finish
            update_thread.join(timeout=0.5)


def _convert_to_harmony_format(content: str) -> dict:
    """Convert content with thinking tags to proper Harmony format structure.

    Args:
        content: Content that may contain thinking tags

    Returns:
        Dict with thinking and/or content fields structured for Harmony format
    """
    # Use the unified thinking processor for all formats
    processor = ThinkingProcessor()
    result = processor.convert_to_harmony_format(content)

    # Safety cleanup: remove any remaining harmony tags from final content
    if "content" in result:
        result["content"] = processor.clean_harmony_tags(result["content"])

    return result


def _process_thinking_tags(
    content: str, console: Console, style_params=None, command_context=None
) -> tuple[bool, str]:
    """Process and render thinking content using the unified ThinkingProcessor.

    Returns:
        Tuple of (has_thinking, final_content) where:
        - has_thinking: True if thinking content was found and processed
        - final_content: The content with thinking sections removed
    """
    # Use the unified thinking processor
    processor = ThinkingProcessor()
    thinking_result = processor.extract_thinking(content)

    if thinking_result.has_thinking:
        # Determine display mode from command context if available
        compressed = True  # Default to compressed
        if command_context and hasattr(command_context, "thinking_processor"):
            processor_instance = command_context.thinking_processor
            compressed = processor_instance.get_display_mode() == "compressed"

        # Render thinking content
        processor.render_thinking(
            thinking_result, console, style_params, compressed=compressed
        )
        return True, thinking_result.final_content

    return False, content  # No thinking content found, return original


def _process_latex_expressions(content: str) -> str:
    """Process LaTeX expressions in content and convert them to ASCII art using sympy."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy.printing.pretty import pretty
    except ImportError:
        # If sympy latex parsing is not available, return content as-is
        return content

    def latex_to_ascii(match):
        latex_expr = match.group(1).strip()
        is_display_math = match.group(0).startswith(
            r"\["
        )  # Display math vs inline math

        try:
            # Check if the expression contains \text{} commands
            if r"\text{" in latex_expr:
                # For expressions with \text{}, skip SymPy parsing and use manual processing
                raise Exception("Contains \\text{}, use manual processing")
            else:
                # Use SymPy's built-in LaTeX parser on expressions without \text{}
                parsed_expr = parse_latex(latex_expr)

                # For display math, use full pretty printing
                if is_display_math:
                    ascii_math = pretty(parsed_expr, use_unicode=True)
                else:
                    # For inline math, try compact representation first
                    ascii_math = pretty(parsed_expr, use_unicode=True, wrap_line=False)

                    # If it's still multi-line, try even more compact representation
                    if "\n" in ascii_math:
                        # Use the string representation with Unicode symbols
                        ascii_math = str(parsed_expr)
                        # Apply basic Unicode substitutions for better readability
                        unicode_replacements = [
                            ("**", "^"),
                            ("*", "⋅"),
                            ("sqrt", "√"),
                            ("pi", "π"),
                            ("infinity", "∞"),
                        ]
                        for pattern, replacement in unicode_replacements:
                            ascii_math = ascii_math.replace(pattern, replacement)

            # Return with appropriate formatting
            if is_display_math:
                return f"\n{ascii_math}\n"
            else:
                return ascii_math
        except Exception:
            # If parsing fails, try simple formatting with Unicode symbols
            try:
                # Enhanced fallback processing for expressions with \text{} commands
                simple_expr = latex_expr

                # Handle \text{} commands by extracting the text content
                simple_expr = re.sub(r"\\text\{([^}]+)\}", r"\1", simple_expr)

                # Handle fractions manually for better display
                def process_frac(match):
                    numerator = match.group(1).strip()
                    denominator = match.group(2).strip()

                    # For display math, create ASCII fraction
                    if is_display_math:
                        # Create a proper fraction display
                        max_width = max(len(numerator), len(denominator)) + 2
                        line = "─" * max_width
                        return f"\n{numerator.center(max_width)}\n{line}\n{denominator.center(max_width)}\n"
                    else:
                        # For inline, use simple division notation
                        return f"({numerator})/({denominator})"

                # Handle \frac{}{} commands
                simple_expr = re.sub(
                    r"\\frac\{([^}]+)\}\{([^}]+)\}", process_frac, simple_expr
                )

                # Basic symbol replacements
                simple_replacements = [
                    (r"\\times", "×"),
                    (r"\\cdot", "·"),
                    (r"\\div", "÷"),
                    (r"\\pi", "π"),
                    (r"\\infty", "∞"),
                    (r"\\geq|\\ge", "≥"),
                    (r"\\leq|\\le", "≤"),
                    (r"\\neq|\\ne", "≠"),
                    (r"\\pm", "±"),
                    (r"\\sqrt\{([^}]+)\}", r"√\1"),
                    (r"\^(\{[^}]+\}|\w)", r"^\1"),  # Keep exponents
                    (r"\_(\{[^}]+\}|\w)", r"_\1"),  # Keep subscripts
                ]

                for pattern, replacement in simple_replacements:
                    simple_expr = re.sub(pattern, replacement, simple_expr)

                # Remove remaining curly braces
                simple_expr = re.sub(r"\{([^}]+)\}", r"\1", simple_expr)

                # Clean up extra spaces
                simple_expr = re.sub(r"\s+", " ", simple_expr).strip()

                # Respect display vs inline math formatting
                if is_display_math:
                    return f"\n{simple_expr}\n"
                else:
                    return f" {simple_expr} "
            except Exception:
                # Ultimate fallback: return original LaTeX
                return match.group(0)

    # Process both display \[...\] and inline \(...\) LaTeX
    content = re.sub(r"\\\[(.*?)\\\]", latex_to_ascii, content, flags=re.DOTALL)
    content = re.sub(r"\\\((.*?)\\\)", latex_to_ascii, content, flags=re.DOTALL)

    return content


def _is_gpt_oss_model(model_name: str) -> bool:
    """Check if the model is a GPT-OSS model that requires Harmony format.

    Args:
        model_name: The model name to check

    Returns:
        True if this is a GPT-OSS model
    """
    # GPT-OSS models typically have "gpt-oss" in the name or path
    model_name_lower = model_name.lower()
    return (
        "gpt-oss" in model_name_lower
        or "gptoss" in model_name_lower
        or "openai/gpt-oss" in model_name_lower
    )


def _convert_conversation_for_harmony(
    conversation: Conversation, is_gpt_oss: bool
) -> Conversation:
    """Convert conversation messages to proper Harmony format if using GPT-OSS.

    Args:
        conversation: The conversation to convert
        is_gpt_oss: Whether this is a GPT-OSS model

    Returns:
        Converted conversation with proper Harmony format
    """
    if not is_gpt_oss:
        return conversation

    # Convert messages that might contain channel tags
    converted_messages = []

    for message in conversation.messages:
        if message.role == Role.ASSISTANT and isinstance(message.content, str):
            # Check if the content has channel tags
            if "<|channel|>" in message.content:
                # Convert to proper Harmony format
                harmony_fields = _convert_to_harmony_format(message.content)

                # Create new message with proper structure
                # For now, we'll use the content field and store thinking separately
                # The actual implementation might need to use a different approach
                # depending on how the inference engine expects the format

                if "content" in harmony_fields:
                    # Use the final content as the main message content
                    converted_message = Message(
                        role=message.role, content=harmony_fields["content"]
                    )

                    # If there's thinking content, we could store it as metadata
                    # or handle it according to the specific inference engine requirements
                    if "thinking" in harmony_fields:
                        # For display purposes, we might want to show both thinking and content
                        # but for the inference engine, we follow the Harmony format
                        pass

                    converted_messages.append(converted_message)
                else:
                    # Fallback to original message
                    converted_messages.append(message)
            else:
                converted_messages.append(message)
        else:
            converted_messages.append(message)

    return Conversation(messages=converted_messages)


def get_engine(config: InferenceConfig) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config."""
    if config.engine is None:
        logger.warning(
            "No inference engine specified. Using the default 'native' engine."
        )
    return build_inference_engine(
        engine_type=config.engine or InferenceEngineType.NATIVE,
        model_params=config.model,
        remote_params=config.remote_params,
    )


def _format_conversation_response(
    conversation: Conversation,
    console: Console,
    model_name: str = "Assistant",
    style_params=None,
    timing_info: Optional[dict] = None,
    command_context=None,
) -> None:
    """Format and display a conversation response with Rich formatting.

    Args:
        conversation: The conversation to format.
        console: Rich console for output.
        model_name: Name of the model.
        style_params: Style parameters.
        timing_info: Optional timing information (time_to_first_token, total_time).
        command_context: Optional command context for processing special commands.
    """
    for message in conversation.messages:
        if message.role == Role.USER:
            continue

        # Extract the text content from the message
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            content = ""
            for item in message.content:
                if hasattr(item, "content") and item.content:
                    # Handle both string and list content in ContentItems
                    if isinstance(item.content, str):
                        content += item.content
                    elif isinstance(item.content, list):
                        # If content is a list, join it as strings
                        content += " ".join(str(c) for c in item.content)
                    else:
                        content += str(item.content)
        else:
            content = str(message.content)

        # Check for thinking content first (all formats)
        has_thinking, final_content = _process_thinking_tags(
            content, console, style_params, command_context
        )
        if has_thinking:
            # Thinking content was processed and rendered, now use the cleaned final content
            content = final_content

        # Additional safety cleanup: remove any remaining harmony tags from display content
        if "<|" in content and "|>" in content:
            from oumi.core.thinking.thinking_processor import ThinkingProcessor

            processor = ThinkingProcessor()
            content = processor.clean_harmony_tags(content)

        # Process LaTeX expressions if no special tags
        content = _process_latex_expressions(content)

        # Get styling from params or use defaults
        if style_params:
            assistant_title_style = style_params.assistant_title_style
            assistant_border_style = style_params.assistant_border_style
            assistant_text_style = style_params.assistant_text_style
            assistant_padding = style_params.assistant_padding
            expand_panels = style_params.expand_panels
        else:
            assistant_title_style = "bold cyan"
            assistant_border_style = "cyan"
            assistant_text_style = "white"
            assistant_padding = (1, 2)
            expand_panels = False

        # Extract just the model name without organization/path
        display_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # Try to render as markdown if it looks like markdown, otherwise as plain text
        markdown_markers = [
            "```",
            "**",
            "*",
            "# ",
            "## ",
            "### ",
            "#### ",
            "`",
            "|",
            "- ",
            "1. ",
            "2. ",
            "3.",
            "---",
            "___",
            "[",
            "](",
        ]
        # Also check for patterns that indicate markdown structure
        has_markdown = any(marker in content for marker in markdown_markers)
        # Check for link patterns or emphasis patterns
        has_markdown = has_markdown or bool(
            re.search(r"\[.*?\]\(.*?\)", content)
        )  # Links
        has_markdown = has_markdown or bool(re.search(r"\*\*.*?\*\*", content))  # Bold
        has_markdown = has_markdown or bool(
            re.search(r"(?<!\*)\*(?!\*).*?(?<!\*)\*(?!\*)", content)
        )  # Italic (not bold)

        # Additional check for tables (more specific)
        has_markdown = has_markdown or bool(
            re.search(r"^\s*\|.*\|\s*$", content, re.MULTILINE)
        )  # Table rows
        has_markdown = has_markdown or bool(
            re.search(r"^\s*\|.*\|.*\|\s*$", content, re.MULTILINE)
        )  # Multi-column tables

        # Check for inline tables (single-line tables from GPT-OSS)
        has_markdown = has_markdown or bool(
            re.search(r"\|[^|]*\|[^|]*\|[^|]*\|", content)
        )  # At least 3 columns inline

        # Debug logging for markdown detection
        if has_markdown:
            logger.debug(f"Markdown detected in content: {content[:100]}...")
            try:
                # Normalize content for better markdown compatibility
                # Replace non-standard dash characters that might interfere with table parsing
                normalized_content = (
                    content.replace("‑", "-").replace("–", "-").replace("—", "-")
                )

                # Simple table formatting improvement for better readability
                def improve_table_readability(text):
                    """Add line breaks around table-like content for better markdown parsing."""
                    # Only add line breaks around obvious table patterns, don't try to parse them
                    lines = text.split("\n")
                    improved_lines = []

                    for line in lines:
                        # If line has lots of pipes and looks like a table row, add line breaks
                        if "|" in line and line.count("|") >= 4:
                            # Add the line as-is but with some spacing
                            improved_lines.append("")  # Empty line before
                            improved_lines.append(line.strip())  # The table line
                            improved_lines.append("")  # Empty line after
                        else:
                            improved_lines.append(line)

                    # Clean up excessive empty lines
                    result = "\n".join(improved_lines)
                    result = re.sub(
                        r"\n\n\n+", "\n\n", result
                    )  # Max 2 consecutive newlines

                    return result

                normalized_content = improve_table_readability(normalized_content)

                # Create markdown with explicit settings for better compatibility
                markdown_obj = Markdown(
                    normalized_content,
                    code_theme="monokai",  # Better code block rendering
                    justify="left",  # Left justify for tables
                    hyperlinks=True,  # Enable hyperlinks
                )
                console.print(
                    Panel(
                        markdown_obj,
                        title=f"[{assistant_title_style}]{display_name}[/{assistant_title_style}]",
                        border_style=assistant_border_style,
                        padding=assistant_padding,
                        expand=expand_panels,
                    )
                )
            except Exception as e:
                # Log the exception for debugging
                logger.warning(
                    f"Markdown rendering failed, falling back to plain text: {e}"
                )
                # Show original content in the fallback to help debug
                logger.debug(
                    f"Content that failed markdown rendering: {content[:200]}..."
                )
                # Fallback to plain text if markdown parsing fails
                console.print(
                    Panel(
                        Text(content, style=assistant_text_style),
                        title=f"[{assistant_title_style}]{display_name}[/{assistant_title_style}]",
                        border_style=assistant_border_style,
                        padding=assistant_padding,
                        expand=expand_panels,
                    )
                )
        else:
            console.print(
                Panel(
                    Text(content, style=assistant_text_style),
                    title=f"[{assistant_title_style}]{display_name}[/{assistant_title_style}]",
                    border_style=assistant_border_style,
                    padding=assistant_padding,
                    expand=expand_panels,
                )
            )

        # Display timing information if available
        if timing_info:
            timing_text = ""
            if "time_to_first_token" in timing_info:
                timing_text += f"First token: {timing_info['time_to_first_token']:.2f}s"
            if "total_time" in timing_info:
                if timing_text:
                    timing_text += " | "
                timing_text += f"Total: {timing_info['total_time']:.2f}s"
            if "tokens_per_second" in timing_info:
                if timing_text:
                    timing_text += " | "
                timing_text += f"Speed: {timing_info['tokens_per_second']:.1f} tokens/s"

            if timing_text:
                # Get style settings
                use_emoji = (
                    getattr(style_params, "use_emoji", True) if style_params else True
                )
                emoji = "⏱️ " if use_emoji else ""

                console.print(
                    Text(f"{emoji}{timing_text}", style="dim cyan"), justify="right"
                )


def _validate_context_usage(
    user_input: str,
    conversation_history: list,
    config: "InferenceConfig",
    system_monitor: Optional["SystemMonitor"] = None,
) -> tuple[bool, str]:
    """Validate that user input won't exceed the context window.

    Args:
        user_input: The user's input text to validate.
        conversation_history: Current conversation messages.
        config: Inference configuration.
        system_monitor: System monitor for current context tracking.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        from oumi.core.attachments.context_manager import ContextWindowManager

        # Get context window size
        max_context = getattr(config.model, "model_max_length", None) or 4096
        model_name = getattr(config.model, "model_name", "default")

        # Initialize context manager
        context_manager = ContextWindowManager(max_context, model_name)

        # Estimate current conversation tokens
        conversation_tokens = 0
        if conversation_history:
            text_content = ""
            for message in conversation_history:
                if isinstance(message, dict):
                    # Handle regular messages
                    if "content" in message:
                        text_content += str(message["content"]) + "\n"
                    # Handle attachment messages
                    elif (
                        message.get("role") == "attachment"
                        and "text_content" in message
                    ):
                        text_content += str(message["text_content"]) + "\n"
            conversation_tokens = context_manager.estimate_tokens(text_content)

        # Estimate tokens for user input
        user_input_tokens = context_manager.estimate_tokens(user_input)

        # Simple check: does current conversation + new input fit in total context?
        total_tokens_needed = conversation_tokens + user_input_tokens
        available_for_input = max_context - conversation_tokens

        # If user input exceeds available space
        if total_tokens_needed > max_context:
            # Calculate context usage for error message

            error_msg = (
                f"Input too large for context window!\n"
                f"  Input tokens: {user_input_tokens:,}\n"
                f"  Available space: {available_for_input:,}\n"
                f"  Current conversation: {conversation_tokens:,}\n"
                f"  Total needed: {total_tokens_needed:,}\n"
                f"  Total context limit: {max_context:,}\n\n"
                f"Suggestions:\n"
                f"  • Use /compact() to compress conversation history\n"
                f"  • Use /clear() to start fresh\n"
                f"  • Break your input into smaller parts"
            )
            return False, error_msg

        return True, ""

    except Exception as e:
        # If validation fails, log but allow input (fallback to existing behavior)
        import logging

        logging.warning(f"Context validation failed: {e}")
        return True, ""


def infer_interactive(
    config: InferenceConfig,
    *,
    input_image_bytes: Optional[list[bytes]] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Interactively provide the model response for a user-provided input."""
    # Create console with custom configuration
    console_kwargs = {}
    if config.style.force_terminal is not None:
        console_kwargs["force_terminal"] = config.style.force_terminal
    if config.style.force_jupyter is not None:
        console_kwargs["force_jupyter"] = config.style.force_jupyter
    if config.style.width is not None:
        console_kwargs["width"] = config.style.width
    if config.style.height is not None:
        console_kwargs["height"] = config.style.height
    if config.style.no_color:
        console_kwargs["no_color"] = True
    if config.style.legacy_windows:
        console_kwargs["legacy_windows"] = True

    console = Console(**console_kwargs)

    # Display welcome message
    emoji = "🤖 " if config.style.use_emoji else ""
    welcome_text = f"{emoji}Oumi Interactive Chat"
    console.print(
        Panel(
            Text(welcome_text, style=config.style.welcome_style),
            subtitle="[dim]Press Ctrl+C or Ctrl+D to exit[/dim]",
            border_style=config.style.welcome_border_style,
            expand=config.style.expand_panels,
        )
    )

    # Display model info
    model_name = getattr(config.model, "model_name", "Unknown Model")
    engine_type = config.engine.value if config.engine else "native"

    info_style = (
        config.style.custom_theme.get("info", "bold green")
        if config.style.custom_theme
        else "bold green"
    )
    console.print(f"[{info_style}]Model:[/{info_style}] {model_name}")
    console.print(f"[{info_style}]Engine:[/{info_style}] {engine_type}")
    if system_prompt:
        console.print(f"[{info_style}]System Prompt:[/{info_style}] {system_prompt}")
    console.print()

    # Create engine up front to avoid reinitializing it for each input.
    inference_engine = get_engine(config)

    conversation_history = []

    # Initialize system monitor for HUD
    max_context_tokens = getattr(config.model, "model_max_length", None) or 4096
    system_monitor = SystemMonitor(max_context_tokens=max_context_tokens)

    # Initialize command system and enhanced input handler
    command_parser = CommandParser()
    command_context = CommandContext(
        console, config, conversation_history, inference_engine, system_monitor
    )
    command_router = CommandRouter(command_context)
    command_context.set_command_router(command_router)

    # Note: After this point, use command_context.inference_engine and command_context.config
    # instead of local variables to ensure model swapping works properly
    input_handler = EnhancedInput(console, config.style.user_prompt_style)

    while True:
        # Display HUD if interval has passed
        system_monitor.display_hud(console, config.style)

        # Track if input came from command override (like /regen)
        is_from_override = False

        try:
            # Get input using enhanced input handler
            input_result = input_handler.get_input("You")

            # Handle input result actions
            if input_result.should_exit:
                return
            elif input_result.cancelled:
                continue
            elif input_result.multiline_toggled:
                # Mode was toggled, get new input
                continue
            elif input_result.action != InputAction.SUBMIT:
                # Some other action that doesn't require inference
                continue

            input_text = input_result.text
            if not input_text.strip():
                continue

            # Add all input to history for arrow key recall (commands and regular input)
            input_handler.add_to_history(input_text.strip())

            # Check for commands first - sanitize input to prevent false positives
            # from multi-line content or complex file paths
            safe_input = make_safe(input_text)
            if command_parser.is_command(safe_input):
                parsed_command = command_parser.parse_command(safe_input)

                if parsed_command is None:
                    command_router.display_command_error("Invalid command syntax")
                    continue

                # Validate the command
                is_valid, error_msg = command_parser.validate_command(parsed_command)
                if not is_valid:
                    command_router.display_command_error(error_msg)
                    continue

                # Execute the command
                command_result = command_router.handle_command(parsed_command)

                # Handle command result
                if not command_result.success and command_result.message:
                    command_router.display_command_error(command_result.message)
                elif command_result.success and command_result.message:
                    command_router.display_command_success(command_result.message)

                # Check if we should exit
                if command_result.should_exit:
                    return

                # Check if we should continue to next iteration (skip inference)
                if not command_result.should_continue:
                    console.print()  # Add spacing
                    continue

                # If command provided input override (e.g., from /regen or /macro), use that
                if (
                    hasattr(command_result, "user_input_override")
                    and command_result.user_input_override
                ):
                    input_text = command_result.user_input_override
                    # Only set is_from_override for actual regeneration operations
                    is_from_override = getattr(command_result, "is_regeneration", False)
                else:
                    # Skip inference since we don't have regular user input
                    console.print()  # Add spacing
                    continue
            else:
                # Not a command, proceed to inference
                pass

        except (EOFError, KeyboardInterrupt):  # Triggered by Ctrl+D/Ctrl+C
            emoji = "👋 " if config.style.use_emoji else ""
            goodbye_style = (
                config.style.custom_theme.get("warning", "yellow")
                if config.style.custom_theme
                else "yellow"
            )
            console.print(f"\n[{goodbye_style}]{emoji}Goodbye![/{goodbye_style}]")
            return

        # Validate context usage before inference
        is_valid, validation_error = _validate_context_usage(
            input_text, conversation_history, config, system_monitor
        )
        if not is_valid:
            command_router.display_command_error(validation_error)
            console.print()  # Add spacing
            continue

        try:
            # Track timing
            inference_start_time = time.time()
            first_token_time = None

            with thinking_with_monitor(
                console=console,
                system_monitor=system_monitor,
                status_message="Thinking...",
                style=config.style.status_style,
                update_interval=2.0,
            ):
                # Check if this is a NATIVE engine that supports conversation history
                from oumi.core.configs import InferenceEngineType

                # Initialize variables that will be used later regardless of engine type
                current_user_message = None
                current_user_content = []

                if config.engine == InferenceEngineType.NATIVE:
                    # Build the full conversation including history for NATIVE engine
                    system_messages = (
                        [Message(role=Role.SYSTEM, content=system_prompt)]
                        if system_prompt
                        else []
                    )

                    # Check if this is a GPT-OSS model
                    model_name = getattr(config.model, "model_name", "")
                    is_gpt_oss = _is_gpt_oss_model(model_name)

                    # Convert conversation history to Message objects
                    history_messages = []

                    for msg in conversation_history:
                        if msg["role"] == "user":
                            if msg.get("content_type") == "multimodal":
                                # This is a message with attachments - content is already ContentItems
                                history_messages.append(
                                    Message(role=Role.USER, content=msg["content"])
                                )
                            else:
                                # Regular text message
                                history_messages.append(
                                    Message(role=Role.USER, content=msg["content"])
                                )
                        elif msg["role"] == "assistant":
                            # For GPT-OSS models, clean up any channel tags from previous responses
                            content = msg["content"]
                            if is_gpt_oss and "<|channel|>" in content:
                                # Extract only the final content for conversation history
                                harmony_fields = _convert_to_harmony_format(content)
                                content = harmony_fields.get("content", content)

                            history_messages.append(
                                Message(role=Role.ASSISTANT, content=content)
                            )
                        elif msg["role"] == "attachment":
                            # Collect attachment text content for the current message
                            if "text_content" in msg:
                                # New simplified text format
                                current_user_content.append(msg["text_content"])
                            elif "content_items" in msg:
                                # Backward compatibility with old ContentItem format
                                for item in msg["content_items"]:
                                    if hasattr(item, "content") and item.content:
                                        current_user_content.append(str(item.content))

                    # Create the current user message
                    if current_user_content:
                        # We have pending attachments, combine them with the user input as text
                        full_content = (
                            "\n\n".join(current_user_content) + "\n\n" + input_text
                        )
                        current_user_message = Message(
                            role=Role.USER, content=full_content
                        )
                    else:
                        # No pending attachments, create a simple text message
                        current_user_message = Message(
                            role=Role.USER, content=input_text
                        )

                    # Create conversation with full history
                    # For regen operations, don't add current_user_message since it's already in history
                    if is_from_override:
                        user_messages = []
                    else:
                        user_messages = [current_user_message]

                    full_conversation = Conversation(
                        messages=system_messages + history_messages + user_messages
                    )

                    # Call inference engine directly with the full conversation
                    try:
                        # Use context references to ensure model swapping works
                        model_response = command_context.inference_engine.infer(
                            input=[full_conversation],
                            inference_config=command_context.config,
                        )
                    except Exception as e:
                        console.print(f"[red]Inference error: {str(e)}[/red]")
                        raise

                    # Record time to first token (approximation - when inference returns)
                    if first_token_time is None:
                        first_token_time = time.time()
                else:
                    # For VLLM and other engines, build full conversation including history
                    system_messages = (
                        [Message(role=Role.SYSTEM, content=system_prompt)]
                        if system_prompt
                        else []
                    )

                    # Check if this is a GPT-OSS model
                    model_name = getattr(config.model, "model_name", "")
                    is_gpt_oss = _is_gpt_oss_model(model_name)

                    # Convert conversation history to Message objects (same as NATIVE)
                    history_messages = []

                    for msg in conversation_history:
                        if msg["role"] == "user":
                            if msg.get("content_type") == "multimodal":
                                # This is a message with attachments - content is already ContentItems
                                history_messages.append(
                                    Message(role=Role.USER, content=msg["content"])
                                )
                            else:
                                # Regular text message
                                history_messages.append(
                                    Message(role=Role.USER, content=msg["content"])
                                )
                        elif msg["role"] == "assistant":
                            # For GPT-OSS models, clean up any channel tags from previous responses
                            content = msg["content"]
                            if is_gpt_oss and "<|channel|>" in content:
                                # Extract only the final content for conversation history
                                harmony_fields = _convert_to_harmony_format(content)
                                content = harmony_fields.get("content", content)

                            history_messages.append(
                                Message(role=Role.ASSISTANT, content=content)
                            )
                        elif msg["role"] == "attachment":
                            # Collect attachment text content for the current message
                            if "text_content" in msg:
                                # New simplified text format
                                current_user_content.append(msg["text_content"])
                            elif "content_items" in msg:
                                # Backward compatibility with old ContentItem format
                                for item in msg["content_items"]:
                                    if hasattr(item, "content") and item.content:
                                        current_user_content.append(str(item.content))

                    # Create the current user message
                    if current_user_content:
                        # We have pending attachments, combine them with the user input as text
                        full_content = (
                            "\n\n".join(current_user_content) + "\n\n" + input_text
                        )
                        current_user_message = Message(
                            role=Role.USER, content=full_content
                        )
                    else:
                        # No pending attachments, create a simple text message
                        current_user_message = Message(
                            role=Role.USER, content=input_text
                        )

                    # Create conversation with full history for VLLM
                    # For regen operations, don't add current_user_message since it's already in history
                    if is_from_override:
                        user_messages = []
                    else:
                        user_messages = [current_user_message]

                    full_conversation = Conversation(
                        messages=system_messages + history_messages + user_messages
                    )

                    # Call inference engine directly with the full conversation
                    try:
                        # Use context references to ensure model swapping works
                        model_response = command_context.inference_engine.infer(
                            input=[full_conversation],
                            inference_config=command_context.config,
                        )
                    except Exception as e:
                        console.print(f"[red]Inference error: {str(e)}[/red]")
                        raise

                    # Record time to first token (approximation - when inference returns)
                    if first_token_time is None:
                        first_token_time = time.time()

            # Calculate timing metrics
            total_inference_time = time.time() - inference_start_time
            time_to_first_token = (
                first_token_time - inference_start_time
                if first_token_time
                else total_inference_time
            )

            # Estimate tokens generated using model-aware ratios
            response_text = ""
            for conversation in model_response:
                for message in conversation.messages:
                    if message.role == Role.ASSISTANT and isinstance(
                        message.content, str
                    ):
                        response_text += message.content

            # Use model-specific character-to-token ratios
            chars_per_token = _get_chars_per_token_ratio(model_name)
            estimated_tokens = len(response_text) / chars_per_token
            tokens_per_second = (
                estimated_tokens / total_inference_time
                if total_inference_time > 0
                else 0
            )

            timing_info = {
                "time_to_first_token": time_to_first_token,
                "total_time": total_inference_time,
                "tokens_per_second": tokens_per_second,
            }

            # Format and display the response with timing
            # Get current model name from command context (handles model swaps)
            current_model_name = model_name  # Default fallback
            if (
                command_context
                and hasattr(command_context, "config")
                and hasattr(command_context.config, "model")
            ):
                # Use the model name from the current (potentially swapped) config
                swapped_model_name = getattr(
                    command_context.config.model, "model_name", None
                )
                if swapped_model_name:
                    current_model_name = swapped_model_name
            elif command_context and hasattr(command_context, "inference_engine"):
                # Fallback to inference engine model name if available
                engine_model_name = getattr(
                    command_context.inference_engine, "model_name", None
                )
                if engine_model_name:
                    current_model_name = engine_model_name
            for conversation in model_response:
                _format_conversation_response(
                    conversation,
                    console,
                    current_model_name,
                    config.style,
                    timing_info,
                    command_context,
                )

            # Store conversation history for all engines
            # Remove any pending attachment markers since they're now incorporated into the message
            # Use slice assignment to modify list in-place, preserving the reference
            conversation_history[:] = [
                msg for msg in conversation_history if msg.get("role") != "attachment"
            ]

            # Store the user message that was sent to the model (always string now)
            # Skip adding to history if this is from a regeneration operation
            # because the message should already be in the conversation history
            if not is_from_override:
                if current_user_message:
                    conversation_history.append(
                        {"role": "user", "content": current_user_message.content}
                    )
                else:
                    # Fallback - shouldn't happen but just in case
                    conversation_history.append({"role": "user", "content": input_text})
            else:
                # Skip user message addition during regeneration operations
                pass

            # Store assistant response in history
            # Check if this is a GPT-OSS model for response cleaning (use current model name)
            current_model_name_for_cleaning = getattr(config.model, "model_name", "")
            is_gpt_oss = _is_gpt_oss_model(current_model_name_for_cleaning)

            # Store only the latest assistant response to avoid duplicates and role alternation issues
            # Get the last conversation and its last assistant message
            if model_response:
                last_conversation = model_response[-1]  # Get most recent conversation
                last_assistant_message = None

                # Find the last assistant message in the conversation
                for message in reversed(last_conversation.messages):
                    if message.role == Role.ASSISTANT and isinstance(
                        message.content, str
                    ):
                        last_assistant_message = message
                        break

                if last_assistant_message:
                    content = last_assistant_message.content

                    # For GPT-OSS models, clean up channel tags when storing in history
                    # This prevents the raw tags from being sent back to the model
                    if is_gpt_oss and "<|channel|>" in content:
                        # Extract only the final content for conversation history
                        harmony_fields = _convert_to_harmony_format(content)
                        stored_content = harmony_fields.get("content", content)

                        # Store the cleaned content for conversation history
                        conversation_history.append(
                            {"role": "assistant", "content": stored_content}
                        )
                    else:
                        # Store original content for non-GPT-OSS models
                        conversation_history.append(
                            {"role": "assistant", "content": content}
                        )

            # Auto-save chat after each complete conversation turn
            try:
                file_operations_handler = command_router._handlers.get(
                    "file_operations"
                )
                if file_operations_handler and hasattr(
                    file_operations_handler, "auto_save_chat"
                ):
                    file_operations_handler.auto_save_chat()
            except Exception:
                # Silently fail auto-save to avoid interrupting user experience
                pass

            # Update context usage for HUD
            # Estimate total conversation tokens
            total_text = ""
            for msg in conversation_history:
                if isinstance(msg.get("content"), str):
                    total_text += msg["content"] + "\n"

            # Better token estimation based on model type
            chars_per_token = _get_chars_per_token_ratio(model_name)
            estimated_context_tokens = len(total_text) / chars_per_token
            system_monitor.update_context_usage(int(estimated_context_tokens))

            # Count conversation turns (each complete user+assistant exchange is one turn)
            assistant_messages = [
                msg for msg in conversation_history if msg.get("role") == "assistant"
            ]
            system_monitor.update_conversation_turns(len(assistant_messages))

            # Note: Input already added to history earlier in the main loop

        except Exception as e:
            console.print(
                Panel(
                    f"[{config.style.error_style}]Error: {str(e)}[/{config.style.error_style}]",
                    title=f"[{config.style.error_title_style}]Error[/{config.style.error_title_style}]",
                    border_style=config.style.error_border_style,
                    expand=config.style.expand_panels,
                )
            )

        console.print()  # Add spacing between exchanges


def infer(
    config: InferenceConfig,
    inputs: Optional[list[str]] = None,
    inference_engine: Optional[BaseInferenceEngine] = None,
    *,
    input_image_bytes: Optional[list[bytes]] = None,
    system_prompt: Optional[str] = None,
) -> list[Conversation]:
    """Infer using the given configuration and inputs.

    Args:
        config: The inference configuration.
        inputs: The inputs to use for inference. If not provided, then the
            input provided within the config will be used instead.
        inference_engine: The inference engine to use for inference. If not
            provided, then a new inference engine will be created based on the
            config.
        input_image_bytes: The image bytes to use for multimodal model
            inference.
        system_prompt: System prompt for task-specific instructions.

    Returns:
        A list of conversations with assistant responses.
    """
    if inference_engine is None:
        inference_engine = get_engine(config)

    if inputs is None:
        # Use the input_filepath from the configuration
        # For now, just return empty list
        return []

    conversations = []
    for input_text in inputs:
        # Create conversation with system message (if provided) and user message
        messages = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))

        # Handle multimodal input
        if input_image_bytes:
            content_items = []
            for image_bytes in input_image_bytes:
                content_items.append(
                    ContentItem(type=Type.IMAGE_URL, content=image_bytes)
                )
            content_items.append(ContentItem(type=Type.TEXT, content=input_text))
            messages.append(Message(role=Role.USER, content=content_items))
        else:
            messages.append(Message(role=Role.USER, content=input_text))

        conversation = Conversation(messages=messages)
        conversations.append(conversation)

    model_response = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )

    return model_response
