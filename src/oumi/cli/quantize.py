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
                "Quantization method to use. Supported methods: "
                "q4_0 (4-bit, default), q4_1 (4-bit improved), "
                "q5_0 (5-bit), q5_1 (5-bit improved), q8_0 (8-bit), "
                "f16 (16-bit float), f32 (32-bit float). "
                "Choose q4_0 for good compression, q8_0 for minimal quality loss."
            ),
        ),
    ] = "q4_0",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Path or identifier of the model to quantize. "
                "Can be a HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf'), "
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

    ⚠️  This feature is currently in development and runs in simulation mode.
        It validates inputs and configuration but does not perform actual quantization.

    This command will quantize machine learning models to reduce their file size and
    memory footprint while maintaining inference performance. Quantization converts
    model weights from higher precision (e.g., float32) to lower precision
    (e.g., 4-bit, 8-bit) representations.

    **Current Status: Interface Testing**
    - ✅ CLI argument validation
    - ✅ Configuration file support
    - ✅ Model identifier validation
    - 🚧 Core quantization implementation (in development)

    **Planned Quantization Methods:**
    - q4_0: 4-bit quantization (4x compression, good quality)
    - q8_0: 8-bit quantization (2x compression, minimal quality loss)
    - f16: 16-bit float (2x compression, GPU-friendly)

    **Planned Output Formats:**
    - GGUF: Compatible with llama.cpp (best for CPU inference)
    - Safetensors: Compatible with HuggingFace transformers
    - PyTorch: Native PyTorch format

    **Testing Examples:**

    Test interface with config file:

        $ oumi quantize --config quantize_config.yaml

    Test with CLI arguments:

        $ oumi quantize --method q4_0 --model meta-llama/Llama-2-7b-hf \\
            --output test.gguf

    Test with local model:

        $ oumi quantize --method q8_0 --model ./my_model --output ./test/model.gguf

    Args:
        ctx: The Typer context object containing extra CLI arguments.
        config: Path to the configuration file for quantization. Can override
            individual settings with additional CLI arguments.
        method: Quantization method to use. Determines the precision and
            compression level of the quantized model.
        model: Path or identifier of the model to quantize. Overrides the
            model specified in the config file if provided.
        output: Output path for the quantized model. The file extension
            should match the desired output format.
        level: The logging level for the command execution.

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
        if method != "q4_0":  # Only override if not default
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

    cli_utils.CONSOLE.print("✅ Model quantized successfully!")
    cli_utils.CONSOLE.print(f"📁 Output saved to: {parsed_config.output_path}")
    if result:
        cli_utils.CONSOLE.print(
            f"📊 Original size: {result.get('original_size', 'Unknown')}"
        )
        cli_utils.CONSOLE.print(
            f"📉 Quantized size: {result.get('quantized_size', 'Unknown')}"
        )
        cli_utils.CONSOLE.print(
            f"🗜️  Compression ratio: {result.get('compression_ratio', 'Unknown')}"
        )
