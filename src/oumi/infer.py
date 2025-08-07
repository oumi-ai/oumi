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
import time
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.commands import CommandHandler, CommandParser
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


def _convert_to_harmony_format(content: str) -> dict:
    """Convert content with thinking tags to proper Harmony format structure.

    Args:
        content: Content that may contain thinking tags

    Returns:
        Dict with thinking and/or content fields structured for Harmony format
    """
    # Use the unified thinking processor for all formats
    processor = ThinkingProcessor()
    return processor.convert_to_harmony_format(content)


def _process_thinking_tags(
    content: str, console: Console, style_params=None, command_handler=None
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
        # Determine display mode from command handler if available
        compressed = True  # Default to compressed
        if command_handler and hasattr(command_handler, 'show_full_thoughts'):
            compressed = not command_handler.show_full_thoughts
        
        # Render thinking content
        processor.render_thinking(
            thinking_result, 
            console, 
            style_params, 
            compressed=compressed
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
    command_handler=None,
) -> None:
    """Format and display a conversation response with Rich formatting.

    Args:
        conversation: The conversation to format.
        console: Rich console for output.
        model_name: Name of the model.
        style_params: Style parameters.
        timing_info: Optional timing information (time_to_first_token, total_time).
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
        has_thinking, final_content = _process_thinking_tags(content, console, style_params, command_handler)
        if has_thinking:
            # Thinking content was processed and rendered, now use the cleaned final content
            content = final_content

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

        # Debug logging for markdown detection
        if has_markdown:
            logger.debug(f"Markdown detected in content: {content[:100]}...")
            try:
                # Normalize content for better markdown compatibility
                # Replace non-standard dash characters that might interfere with table parsing
                normalized_content = (
                    content.replace("‑", "-").replace("–", "-").replace("—", "-")
                )

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

    # Initialize command system and enhanced input handler
    command_parser = CommandParser()
    command_handler = CommandHandler(
        console, config, conversation_history, inference_engine
    )
    input_handler = EnhancedInput(console, config.style.user_prompt_style)

    # Initialize system monitor for HUD
    max_context_tokens = getattr(config.model, "model_max_length", 4096)
    system_monitor = SystemMonitor(max_context_tokens=max_context_tokens)

    while True:
        # Display HUD if interval has passed
        system_monitor.display_hud(console, config.style)

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

            # Check for commands first
            if command_parser.is_command(input_text):
                parsed_command = command_parser.parse_command(input_text)

                if parsed_command is None:
                    command_handler.display_command_error("Invalid command syntax")
                    continue

                # Validate the command
                is_valid, error_msg = command_parser.validate_command(parsed_command)
                if not is_valid:
                    command_handler.display_command_error(error_msg)
                    continue

                # Execute the command
                command_result = command_handler.handle_command(parsed_command)

                # Handle command result
                if not command_result.success and command_result.message:
                    command_handler.display_command_error(command_result.message)
                elif command_result.success and command_result.message:
                    command_handler.display_command_success(command_result.message)

                # Check if we should exit
                if command_result.should_exit:
                    return

                # Check if we should continue to next iteration (skip inference)
                if not command_result.should_continue:
                    console.print()  # Add spacing
                    continue

                # If command provided input override (e.g., from /regen), use that
                if (
                    hasattr(command_result, "user_input_override")
                    and command_result.user_input_override
                ):
                    input_text = command_result.user_input_override
                else:
                    # Skip inference since we don't have regular user input
                    console.print()  # Add spacing
                    continue

        except (EOFError, KeyboardInterrupt):  # Triggered by Ctrl+D/Ctrl+C
            emoji = "👋 " if config.style.use_emoji else ""
            goodbye_style = (
                config.style.custom_theme.get("warning", "yellow")
                if config.style.custom_theme
                else "yellow"
            )
            console.print(f"\n[{goodbye_style}]{emoji}Goodbye![/{goodbye_style}]")
            return

        try:
            # Track timing
            inference_start_time = time.time()
            first_token_time = None

            with console.status(
                f"[{config.style.status_style}]Thinking...[/{config.style.status_style}]",
                spinner="dots",
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
                    full_conversation = Conversation(
                        messages=system_messages
                        + history_messages
                        + [current_user_message]
                    )

                    # Call inference engine directly with the full conversation
                    model_response = inference_engine.infer(
                        input=[full_conversation],
                        inference_config=config,
                    )

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
                    full_conversation = Conversation(
                        messages=system_messages
                        + history_messages
                        + [current_user_message]
                    )

                    # Call inference engine directly with the full conversation
                    model_response = inference_engine.infer(
                        input=[full_conversation],
                        inference_config=config,
                    )

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

            # Estimate tokens generated (rough approximation)
            response_text = ""
            for conversation in model_response:
                for message in conversation.messages:
                    if message.role == Role.ASSISTANT and isinstance(
                        message.content, str
                    ):
                        response_text += message.content

            # Rough token estimation (4 chars per token average)
            estimated_tokens = len(response_text) / 4
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
            for conversation in model_response:
                _format_conversation_response(
                    conversation, console, model_name, config.style, timing_info, command_handler
                )

            # Store conversation history for all engines
            # Remove any pending attachment markers since they're now incorporated into the message
            # Use slice assignment to modify list in-place, preserving the reference
            conversation_history[:] = [
                msg for msg in conversation_history if msg.get("role") != "attachment"
            ]

            # Store the user message that was sent to the model (always string now)
            if current_user_message:
                conversation_history.append(
                    {"role": "user", "content": current_user_message.content}
                )
            else:
                # Fallback - shouldn't happen but just in case
                conversation_history.append({"role": "user", "content": input_text})

            # Store assistant response in history
            # Check if this is a GPT-OSS model for response cleaning
            model_name = getattr(config.model, "model_name", "")
            is_gpt_oss = _is_gpt_oss_model(model_name)

            for conversation in model_response:
                for message in conversation.messages:
                    if message.role == Role.ASSISTANT and isinstance(
                        message.content, str
                    ):
                        content = message.content

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

            # Update context usage for HUD
            # Estimate total conversation tokens
            total_text = ""
            for msg in conversation_history:
                if isinstance(msg.get("content"), str):
                    total_text += msg["content"] + "\n"

            # Rough token estimation (4 chars per token average)
            estimated_context_tokens = len(total_text) / 4
            system_monitor.update_context_usage(int(estimated_context_tokens))

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
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: A list of input PNG image bytes to be used with `image+text`
            VLMs. Only used in interactive mode.
        system_prompt: System prompt for task-specific instructions.

    Returns:
        object: A list of model responses.
    """
    if not inference_engine:
        inference_engine = get_engine(config)

    # Pass None if no conversations are provided.
    conversations = None
    if inputs is not None and len(inputs) > 0:
        system_messages = (
            [Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []
        )
        if input_image_bytes is None or len(input_image_bytes) == 0:
            conversations = [
                Conversation(
                    messages=(
                        system_messages + [Message(role=Role.USER, content=content)]
                    )
                )
                for content in inputs
            ]
        else:
            conversations = [
                Conversation(
                    messages=(
                        system_messages
                        + [
                            Message(
                                role=Role.USER,
                                content=(
                                    [
                                        ContentItem(
                                            type=Type.IMAGE_BINARY, binary=image_bytes
                                        )
                                        for image_bytes in input_image_bytes
                                    ]
                                    + [ContentItem(type=Type.TEXT, content=content)]
                                ),
                            )
                        ]
                    )
                )
                for content in inputs
            ]

    generations = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )
    return generations
