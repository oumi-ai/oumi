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

import os
from typing import Annotated, Final, Optional, Dict, Tuple, List

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger

_DEFAULT_CLI_PDF_DPI: Final[int] = 200


def infer(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ] = "",
    interactive: Annotated[
        bool,
        typer.Option("-i", "--interactive", help="Run in an interactive session."),
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check", 
            help="Check which inference engines are compatible with the current environment.",
        ),
    ] = False,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help=(
                "File path or URL of an input image to be used with image+text VLLMs. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help=(
                "System prompt for task-specific instructions. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Run inference on a model.

    If `input_filepath` is provided in the configuration file, inference will run on
    those input examples. Otherwise, inference will run interactively with user-provided
    inputs.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        output_dir: Directory to save configs
        (defaults to OUMI_DIR env var or ~/.oumi/fetch).
        interactive: Whether to run in an interactive session.
        check: Whether to check inference engine compatibility.
        image: Path to the input image for `image+text` VLLMs.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    # If --check flag is used, run the engine compatibility check
    if check:
        # Import here to avoid circular imports
        from oumi.core.configs import InferenceEngineType
        from oumi.builders.inference_engines import ENGINE_MAP
        
        # Use rich for pretty formatting
        console = Console()
        
        console.print("Checking inference engine compatibility...", style="bold blue")
        
        # Create table for the results
        table = Table(
            title="Inference Engine Compatibility",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        
        table.add_column("Engine", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Compatible", style="green")
        table.add_column("Details", style="yellow")
        
        # Check each engine
        all_engines = list(InferenceEngineType)
        all_engines.sort(key=lambda x: x.value)
        
        for engine_type in all_engines:
            engine_name = engine_type.value
            
            # Skip REMOTE type as it's a generic type and not directly used
            if engine_type == InferenceEngineType.REMOTE:
                continue
                
            if engine_type in ENGINE_MAP:
                engine_class = ENGINE_MAP[engine_type]
                
                # Categorize engines
                if any(remote_name in engine_class.__name__ for remote_name in ["OpenAI", "Anthropic", "Google", "Remote"]):
                    engine_category = "Remote API"
                elif engine_class.__name__ in ["VLLMInferenceEngine", "SGLangInferenceEngine"]:
                    engine_category = "GPU-based"
                elif engine_class.__name__ == "LlamaCppInferenceEngine":
                    engine_category = "CPU/GPU"
                else:
                    engine_category = "Local"
                
                # Check compatibility
                try:
                    is_compatible, message = engine_class.check()
                except Exception as e:
                    is_compatible = False
                    message = f"Error checking compatibility: {str(e)}"
                
                # Format compatibility status
                status_text = Text("✓ Yes") if is_compatible else Text("✗ No")
                status_text.stylize("bold green" if is_compatible else "bold red")
                
                table.add_row(
                    engine_name,
                    engine_category,
                    status_text,
                    message
                )
            else:
                table.add_row(
                    engine_name, 
                    "Unknown",
                    Text("? Unknown", style="bold yellow"),
                    "Engine implementation not found in ENGINE_MAP"
                )
        
        console.print(table)
        return
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.INFER),
        )
    )

    # Delayed imports
    from oumi import infer as oumi_infer
    from oumi import infer_interactive as oumi_infer_interactive
    from oumi.core.configs import InferenceConfig
    from oumi.utils.image_utils import (
        create_png_bytes_from_image_list,
        load_image_png_bytes_from_path,
        load_image_png_bytes_from_url,
        load_pdf_pages_from_path,
        load_pdf_pages_from_url,
    )
    # End imports

    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_image_png_bytes: Optional[list[bytes]] = None
    if image:
        image_lower = image.lower()
        if image_lower.startswith("http://") or image_lower.startswith("https://"):
            if image_lower.endswith(".pdf"):
                input_image_png_bytes = create_png_bytes_from_image_list(
                    load_pdf_pages_from_url(image, dpi=_DEFAULT_CLI_PDF_DPI)
                )
            else:
                input_image_png_bytes = [load_image_png_bytes_from_url(image)]
        else:
            if image_lower.endswith(".pdf"):
                input_image_png_bytes = create_png_bytes_from_image_list(
                    load_pdf_pages_from_path(image, dpi=_DEFAULT_CLI_PDF_DPI)
                )
            else:
                input_image_png_bytes = [load_image_png_bytes_from_path(image)]
    if parsed_config.input_path:
        if interactive:
            logger.warning(
                "Input path provided, skipping interactive inference. "
                "To run in interactive mode, do not provide an input path."
            )
        generations = oumi_infer(parsed_config)
        # Don't print results if output_filepath is provided.
        if parsed_config.output_path:
            return
        table = Table(
            title="Inference Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        table.add_column("Conversation", style="green")
        for generation in generations:
            table.add_row(repr(generation))
        cli_utils.CONSOLE.print(table)
        return
    if not interactive:
        logger.warning(
            "No input path provided, running in interactive mode. "
            "To run with an input path, provide one in the configuration file."
        )
    return oumi_infer_interactive(
        parsed_config,
        input_image_bytes=input_image_png_bytes,
        system_prompt=system_prompt,
    )
