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
                "AWQ methods: awq_q4_0 (default), "
                "awq_q4_1, awq_q8_0, awq_f16. "
                "BitsAndBytes methods: bnb_4bit, bnb_8bit. "
            ),
        ),
    ] = "awq_q4_0",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Path or identifier of the model to quantize. "
                "Can be a HuggingFace model ID (e.g., 'oumi-ai/HallOumi-8B' this "
                "is the model from Oumi), "
                "a local directory path, or an Oumi model registry identifier. "
                "If not specified, uses the model defined in the config file."
            ),
        ),
    ] = "",
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output path for the quantized model.",
        ),
    ] = "quantized_model",
):
    r"""🚧 DEVELOPMENT: Quantize a model to reduce its size and memory requirements.

    Example:
        oumi quantize --model "oumi-ai/HallOumi-8B" --method awq_q4_0 \
            --output "halloumi-8b-q4.gguf"

    Note:
        The quantization process may require significant memory and time,
        especially for large models. Ensure sufficient disk space for both
        the original and quantized models during processing.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Delayed imports
    from oumi import quantize as oumi_quantize
    from oumi.core.configs import ModelParams, QuantizationConfig
    from oumi.quantize.constants import SUPPORTED_METHODS
    from oumi.utils.torch_utils import device_cleanup

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
        if model:  # Non-empty string
            parsed_config.model.model_name = model
        if method != "awq_q4_0":  # Only override if not default
            parsed_config.method = method
        if output != "quantized_model.gguf":  # Only override if not default
            parsed_config.output_path = output

        # Only safetensors is supported for now
        parsed_config.output_format = "safetensors"
    else:
        # Create config from CLI arguments
        if not model:  # Empty string or Nones
            raise typer.BadParameter(
                "Either --config must be provided or --model must be specified"
            )

        # Determine appropriate output format based on method
        if method.startswith("awq_"):
            output_format = "pytorch"
        elif method.startswith("bnb_"):
            output_format = "pytorch"  # or "safetensors" depending on preference
        else:
            raise ValueError(
                f"Unsupported quantization method: {method}. "
                f"Must be one of: {SUPPORTED_METHODS}."
            )

        parsed_config = QuantizationConfig(
            model=ModelParams(model_name=model),
            method=method,
            output_path=output,
            output_format=output_format,
        )

    parsed_config.finalize_and_validate()

    # Configure environment and cleanup
    cli_utils.configure_common_env_vars()
    device_cleanup()

    try:
        with cli_utils.CONSOLE.status("Quantizing model...", spinner="dots"):
            result = oumi_quantize(parsed_config)
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        device_cleanup()
        raise

    # Check if quantization was successful
    if not result or not result.output_path:
        logger.error("❌ Model quantization failed!")
        device_cleanup()
        return

    # Display results using QuantizationResult attributes
    cli_utils.CONSOLE.print("✅ Model quantized successfully!")
    cli_utils.CONSOLE.print(f"📁 Output saved to: {result.output_path}")
    cli_utils.CONSOLE.print(f"🔧 Method: {result.quantization_method}")
    cli_utils.CONSOLE.print(f"📋 Format: {result.format_type}")
    cli_utils.CONSOLE.print(
        f"📊 Quantized size: {result.quantized_size_bytes / (1024**3):.2f} GB"
    )

    # Display additional info if available
    if result.additional_info:
        for key, value in result.additional_info.items():
            cli_utils.CONSOLE.print(f"ℹ️  {key}: {value}")

    device_cleanup()
