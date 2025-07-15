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

from typing import Annotated, Optional

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


def quantize(
    ctx: typer.Context,
    config: Annotated[
        Optional[str],
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help=(
                "Path to the configuration file for quantization. "
                "Can be a local path or an Oumi registry URI (oumi://...). "
                "The config file should be in YAML format and define a "
                "QuantizationConfig. "
                "If not provided, will create a default config from CLI arguments."
            ),
        ),
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help=(
                "Quantization method to use. "
                "AWQ methods (recommended): awq_q4_0 (default), awq_q4_1, awq_q8_0, awq_f16. "
                "Direct GGUF: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32. "
                "AWQ provides better quality through activation-aware quantization."
            ),
        ),
    ] = "awq_q4_0",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Path or identifier of the model to quantize. "
                "Can be a HuggingFace model ID (e.g., 'oumi-ai/HallOumi-8B' this is the model from Oumi), "
                "a local directory path, or an Oumi model registry identifier. "
                "If not specified, uses the model defined in the config file."
            ),
        ),
    ] = "",
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help=(
                "Output path for the quantized model. "
                "For GGUF format, use .gguf extension. "
                "For other formats, can be a directory or file path. "
                "Default creates 'quantized_model.gguf' in current directory."
            ),
        ),
    ] = "quantized_model.gguf",
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    r"""🚧 DEVELOPMENT: Quantize a model to reduce its size and memory requirements.

    Example:
        oumi quantize --model "oumi-ai/HallOumi-8B" --method awq_q4_0 --output "halloumi-8b-q4.gguf"

    Note:
        The quantization process may require significant memory and time,
        especially for large models. Ensure sufficient disk space for both
        the original and quantized models during processing.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Delayed imports
    from oumi import quantize as oumi_quantize
    from oumi.core.configs import ModelParams, QuantizationConfig
    # End imports

    if config is not None:
        # Use provided config file
        config_path = str(
            cli_utils.resolve_and_fetch_config(
                try_get_config_name_for_alias(config, AliasType.QUANTIZE),
            )
        )
        parsed_config: QuantizationConfig = QuantizationConfig.from_yaml_and_arg_list(
            config_path, extra_args, logger=logger
        )

        # Override config with CLI arguments if provided
        if model:
            parsed_config.model.model_name = model
        if method != "awq_q4_0":  # Only override if not default
            parsed_config.method = method
        if output != "quantized_model.gguf":  # Only override if not default
            parsed_config.output_path = output
    else:
        # Create config from CLI arguments
        if not model:
            raise typer.BadParameter(
                "Either --config must be provided or --model must be specified"
            )

        parsed_config = QuantizationConfig(
            model=ModelParams(model_name=model),
            method=method,
            output_path=output,
            output_format="gguf",  # Default format
        )

    parsed_config.finalize_and_validate()

    with cli_utils.CONSOLE.status("Quantizing model...", spinner="dots"):
        result = oumi_quantize(parsed_config)

    # Check for GGUF conversion failure
    if result and result.get("fallback_mode"):
        raise RuntimeError(
            "AWQ quantization fallback mode is no longer supported.\n"
            "Please try using a different quantization method (e.g., --method bnb_4bit or --method q4_0) "
            "or ensure that the required dependencies for AWQ are installed."
        )
    elif result and result.get("gguf_conversion_failed"):
        cli_utils.CONSOLE.print("✅ AWQ quantization completed successfully!")
        cli_utils.CONSOLE.print(
            "⚠️  GGUF conversion failed - saved as PyTorch format instead"
        )
        cli_utils.CONSOLE.print(
            "💡 For GGUF output, install: pip install llama-cpp-python"
        )
    else:
        cli_utils.CONSOLE.print("✅ Model quantized successfully!")

    # Display output path (might have changed due to fallback)
    actual_output_path = (
        result.get("output_path", parsed_config.output_path)
        if result
        else parsed_config.output_path
    )
    cli_utils.CONSOLE.print(f"📁 Output saved to: {actual_output_path}")

    if result:
        if result.get("simulation_mode"):
            cli_utils.CONSOLE.print("🎭 Mode: Simulation")
            cli_utils.CONSOLE.print(
                f"📦 Method: {result.get('quantization_method', 'Unknown')}"
            )
        else:
            cli_utils.CONSOLE.print(
                f"📊 Original size: {result.get('original_size', 'Unknown')}"
            )
        cli_utils.CONSOLE.print(
            f"📉 Output size: {result.get('quantized_size', 'Unknown')}"
        )
        if not result.get("simulation_mode"):
            cli_utils.CONSOLE.print(
                f"🗜️  Compression ratio: {result.get('compression_ratio', 'Unknown')}"
            )
