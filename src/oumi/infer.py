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
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.logging import logger


def _process_gpt_oss_tags(content: str, console: Console, style_params=None) -> None:
    """Process and render GPT-OSS reasoning tags with nice formatting."""
    # Pattern to match GPT-OSS reasoning blocks
    # <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
    pattern = r"<\|channel\|>(\w+)<\|message\|>(.*?)(?:<\|end\|><\|start\|>assistant<\|channel\|>(\w+)<\|message\|>(.*?))?(?:<\|end\|>|$)"

    matches = list(re.finditer(pattern, content, re.DOTALL))

    if matches:
        for match in matches:
            channel1 = match.group(1)  # e.g., "analysis"
            content1 = match.group(2).strip()  # analysis content
            channel2 = match.group(3)  # e.g., "final"
            content2 = match.group(4).strip() if match.group(4) else ""  # final content

            # Get styles from params or use defaults
            if style_params:
                analysis_text_style = style_params.analysis_text_style
                analysis_title_style = style_params.analysis_title_style
                analysis_border_style = style_params.analysis_border_style
                use_emoji = style_params.use_emoji
                expand_panels = style_params.expand_panels
            else:
                analysis_text_style = "dim cyan"
                analysis_title_style = "bold yellow"
                analysis_border_style = "yellow"
                use_emoji = True
                expand_panels = False

            # Render analysis section
            if channel1 == "analysis":
                emoji = "🧠 " if use_emoji else ""
                console.print(
                    Panel(
                        Text(content1, style=analysis_text_style),
                        title=f"[{analysis_title_style}]{emoji}Analysis[/{analysis_title_style}]",
                        border_style=analysis_border_style,
                        padding=(0, 1),
                        expand=expand_panels,
                    )
                )
            else:
                console.print(
                    Panel(
                        Text(content1, style="white"),
                        title=f"[bold magenta]{channel1.title()}[/bold magenta]",
                        border_style="magenta",
                        padding=(0, 1),
                        expand=expand_panels,
                    )
                )

            # Render final response section
            if channel2 and content2:
                if channel2 == "final":
                    if style_params:
                        response_text_style = style_params.response_text_style
                        response_title_style = style_params.response_title_style
                        response_border_style = style_params.response_border_style
                    else:
                        response_text_style = "bright_white"
                        response_title_style = "bold green"
                        response_border_style = "green"

                    emoji = "💬 " if use_emoji else ""
                    console.print(
                        Panel(
                            Text(content2, style=response_text_style),
                            title=f"[{response_title_style}]{emoji}Response[/{response_title_style}]",
                            border_style=response_border_style,
                            padding=(0, 1),
                            expand=expand_panels,
                        )
                    )
                else:
                    console.print(
                        Panel(
                            Text(content2, style="white"),
                            title=f"[bold blue]{channel2.title()}[/bold blue]",
                            border_style="blue",
                            padding=(0, 1),
                            expand=expand_panels,
                        )
                    )
        return True  # Indicates we processed special tags

    return False  # No special tags found


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
) -> None:
    """Format and display a conversation response with Rich formatting."""
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
                    content += str(item.content)
        else:
            content = str(message.content)

        # Check for GPT-OSS reasoning tags first
        if _process_gpt_oss_tags(content, console, style_params):
            # Special tags were processed, we're done
            return

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
        if any(marker in content for marker in ["```", "**", "*", "#", "`"]):
            try:
                console.print(
                    Panel(
                        Markdown(content),
                        title=f"[{assistant_title_style}]{display_name}[/{assistant_title_style}]",
                        border_style=assistant_border_style,
                        padding=assistant_padding,
                        expand=expand_panels,
                    )
                )
            except Exception:
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

    while True:
        try:
            # Use Rich prompt for better UX
            input_text = Prompt.ask(
                f"[{config.style.user_prompt_style}]You[/{config.style.user_prompt_style}]",
                console=console,
            )
            if not input_text.strip():
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
            with console.status(
                f"[{config.style.status_style}]Thinking...[/{config.style.status_style}]",
                spinner="dots",
            ):
                # Check if this is a NATIVE engine that supports conversation history
                from oumi.core.configs import InferenceEngineType

                if config.engine == InferenceEngineType.NATIVE:
                    # Build the full conversation including history for NATIVE engine
                    system_messages = (
                        [Message(role=Role.SYSTEM, content=system_prompt)]
                        if system_prompt
                        else []
                    )

                    # Convert conversation history to Message objects
                    history_messages = []
                    for msg in conversation_history:
                        if msg["role"] == "user":
                            history_messages.append(
                                Message(role=Role.USER, content=msg["content"])
                            )
                        elif msg["role"] == "assistant":
                            history_messages.append(
                                Message(role=Role.ASSISTANT, content=msg["content"])
                            )

                    # Add the current user input
                    current_user_message = Message(role=Role.USER, content=input_text)

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
                else:
                    # For VLLM and other engines, use the original single-input approach
                    model_response = infer(
                        config=config,
                        inputs=[input_text],
                        system_prompt=system_prompt,
                        input_image_bytes=input_image_bytes,
                        inference_engine=inference_engine,
                    )

            # Format and display the response
            for conversation in model_response:
                _format_conversation_response(
                    conversation, console, model_name, config.style
                )

            # Store conversation history (only for NATIVE engine which supports it)
            if config.engine == InferenceEngineType.NATIVE:
                # For NATIVE engine, store both user and assistant messages in history after successful inference
                conversation_history.append({"role": "user", "content": input_text})

                # Store assistant response in history
                for conversation in model_response:
                    for message in conversation.messages:
                        if message.role == Role.ASSISTANT and isinstance(
                            message.content, str
                        ):
                            conversation_history.append(
                                {"role": "assistant", "content": message.content}
                            )
            else:
                # For other engines like VLLM, conversation history is handled by the engine itself
                # so we don't manually track it here
                pass

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
