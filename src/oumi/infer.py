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
from rich.syntax import Syntax
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
        is_display_math = match.group(0).startswith(r'\[')  # Display math vs inline math
        
        try:
            # Check if the expression contains \text{} commands
            if r'\text{' in latex_expr:
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
                    if '\n' in ascii_math:
                        # Use the string representation with Unicode symbols
                        ascii_math = str(parsed_expr)
                        # Apply basic Unicode substitutions for better readability
                        unicode_replacements = [
                            ('**', '^'),
                            ('*', '⋅'),
                            ('sqrt', '√'),
                            ('pi', 'π'),
                            ('infinity', '∞'),
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
                simple_expr = re.sub(r'\\text\{([^}]+)\}', r'\1', simple_expr)
                
                # Handle fractions manually for better display
                def process_frac(match):
                    numerator = match.group(1).strip()
                    denominator = match.group(2).strip()
                    
                    # For display math, create ASCII fraction
                    if is_display_math:
                        # Create a proper fraction display
                        max_width = max(len(numerator), len(denominator)) + 2
                        line = '─' * max_width
                        return f"\n{numerator.center(max_width)}\n{line}\n{denominator.center(max_width)}\n"
                    else:
                        # For inline, use simple division notation
                        return f"({numerator})/({denominator})"
                
                # Handle \frac{}{} commands
                simple_expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', process_frac, simple_expr)
                
                # Basic symbol replacements
                simple_replacements = [
                    (r'\\times', '×'),
                    (r'\\cdot', '·'),
                    (r'\\div', '÷'),
                    (r'\\pi', 'π'),
                    (r'\\infty', '∞'),
                    (r'\\geq|\\ge', '≥'),
                    (r'\\leq|\\le', '≤'),
                    (r'\\neq|\\ne', '≠'),
                    (r'\\pm', '±'),
                    (r'\\sqrt\{([^}]+)\}', r'√\1'),
                    (r'\^(\{[^}]+\}|\w)', r'^\1'),  # Keep exponents
                    (r'\_(\{[^}]+\}|\w)', r'_\1'),  # Keep subscripts
                ]
                
                for pattern, replacement in simple_replacements:
                    simple_expr = re.sub(pattern, replacement, simple_expr)
                
                # Remove remaining curly braces
                simple_expr = re.sub(r'\{([^}]+)\}', r'\1', simple_expr)
                
                # Clean up extra spaces
                simple_expr = re.sub(r'\s+', ' ', simple_expr).strip()
                
                # Respect display vs inline math formatting
                if is_display_math:
                    return f"\n{simple_expr}\n"
                else:
                    return f" {simple_expr} "
            except Exception:
                # Ultimate fallback: return original LaTeX
                return match.group(0)
    
    # Process both display \[...\] and inline \(...\) LaTeX
    content = re.sub(r'\\\[(.*?)\\\]', latex_to_ascii, content, flags=re.DOTALL)
    content = re.sub(r'\\\((.*?)\\\)', latex_to_ascii, content, flags=re.DOTALL)
    
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


def _format_conversation_response(conversation: Conversation, console: Console, model_name: str = "Assistant") -> None:
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
                if hasattr(item, 'content') and item.content:
                    content += str(item.content)
        else:
            content = str(message.content)
        
        # Process LaTeX expressions first
        content = _process_latex_expressions(content)
        
        # Extract just the model name without organization/path
        display_name = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Try to render as markdown if it looks like markdown, otherwise as plain text
        if any(marker in content for marker in ['```', '**', '*', '#', '`']):
            try:
                console.print(Panel(
                    Markdown(content),
                    title=f"[bold cyan]{display_name}[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                ))
            except Exception:
                # Fallback to plain text if markdown parsing fails
                console.print(Panel(
                    Text(content, style="white"),
                    title=f"[bold cyan]{display_name}[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                ))
        else:
            console.print(Panel(
                Text(content, style="white"),
                title=f"[bold cyan]{display_name}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))


def infer_interactive(
    config: InferenceConfig,
    *,
    input_image_bytes: Optional[list[bytes]] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Interactively provide the model response for a user-provided input."""
    console = Console()
    
    # Display welcome message
    console.print(Panel(
        Text("🤖 Oumi Interactive Chat", style="bold magenta"),
        subtitle="[dim]Press Ctrl+C or Ctrl+D to exit[/dim]",
        border_style="magenta"
    ))
    
    # Display model info
    model_name = getattr(config.model, 'model_name', 'Unknown Model')
    engine_type = config.engine.value if config.engine else "native"
    
    console.print(f"[bold green]Model:[/bold green] {model_name}")
    console.print(f"[bold green]Engine:[/bold green] {engine_type}")
    if system_prompt:
        console.print(f"[bold green]System Prompt:[/bold green] {system_prompt}")
    console.print()
    
    # Create engine up front to avoid reinitializing it for each input.
    inference_engine = get_engine(config)
    
    conversation_history = []
    
    while True:
        try:
            # Use Rich prompt for better UX
            input_text = Prompt.ask(
                "[bold blue]You[/bold blue]",
                console=console
            )
            if not input_text.strip():
                continue
                
        except (EOFError, KeyboardInterrupt):  # Triggered by Ctrl+D/Ctrl+C
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            return
            
        try:
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                # Build the full conversation including history
                system_messages = (
                    [Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []
                )
                
                # Convert conversation history to Message objects
                history_messages = []
                for msg in conversation_history:
                    if msg["role"] == "user":
                        history_messages.append(Message(role=Role.USER, content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history_messages.append(Message(role=Role.ASSISTANT, content=msg["content"]))
                
                # Add the current user input
                current_user_message = Message(role=Role.USER, content=input_text)
                
                # Create conversation with full history
                full_conversation = Conversation(
                    messages=system_messages + history_messages + [current_user_message]
                )
                
                # Call inference engine directly with the full conversation
                model_response = inference_engine.infer(
                    input=[full_conversation],
                    inference_config=config,
                )
            
            # Format and display the response
            for conversation in model_response:
                _format_conversation_response(conversation, console, model_name)
                
            # Store both user and assistant messages in history after successful inference
            conversation_history.append({"role": "user", "content": input_text})
            
            # Store assistant response in history
            for conversation in model_response:
                for message in conversation.messages:
                    if message.role == Role.ASSISTANT and isinstance(message.content, str):
                        conversation_history.append({"role": "assistant", "content": message.content})
                        
        except Exception as e:
            console.print(Panel(
                f"[red]Error: {str(e)}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
        
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
