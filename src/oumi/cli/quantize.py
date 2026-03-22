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

from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_quantize_config
from oumi.utils.logging import logger

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.QUANTIZE, "Available Quantization Configs", "quantize"
)


def quantize(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str | None,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help=(
                "Path to the configuration file for quantization. "
                "Can be a local path or an Oumi registry URI (oumi://...). "
                "The config file should be in YAML format and define a "
                "QuantizationConfig. "
                "If not provided, will create a default config from CLI arguments."
            ),
            rich_help_panel="Options",
            autocompletion=complete_quantize_config,
        ),
    ] = None,
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available quantization configs.",
            callback=_list_configs_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    # Model options
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Path or identifier of the model to quantize. "
                "Can be a HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B-Instruct'), "
                "a local directory path, or an Oumi model registry identifier. "
                "If not specified, uses the model defined in the config file."
            ),
            rich_help_panel="Model",
        ),
    ] = "",
    # Quantization options
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help=(
                "Quantization method to use. "
                "LLM Compressor: fp8_dynamic, fp8_block, "
                "w4a16, w4a16_asym, w8a16. "
                "BitsAndBytes: bnb_4bit, bnb_8bit."
            ),
            rich_help_panel="Quantization",
        ),
    ] = "",
    algorithm: Annotated[
        str,
        typer.Option(
            "--algorithm",
            help=(
                "Quantization algorithm: auto, rtn, gptq, awq. "
                "'auto' selects the best algorithm for the chosen method."
            ),
            rich_help_panel="Quantization",
        ),
    ] = "",
    # Output options
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output directory for the quantized model.",
            rich_help_panel="Output",
        ),
    ] = "quantized_model",
):
    r"""Quantize a model to reduce its size and memory requirements.

    Supports LLM Compressor (FP8, GPTQ, AWQ) and BitsAndBytes (NF4, INT8)
    quantization methods. Produces models in compressed-tensors format
    optimized for vLLM serving.

    Example:
        oumi quantize --model "meta-llama/Llama-3.1-8B-Instruct" \
            --method fp8_dynamic --output "llama3-8b-fp8"

    Note:
        The quantization process may require significant memory and time,
        especially for large models. Ensure sufficient disk space for both
        the original and quantized models during processing.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Delayed imports
    from oumi import quantize as oumi_quantize
    from oumi.core.configs import ModelParams, QuantizationConfig
    from oumi.quantize.constants import QuantizationAlgorithm, QuantizationMethod
    from oumi.utils.torch_utils import device_cleanup

    parsed_method: QuantizationMethod | None = None
    if method:
        try:
            parsed_method = QuantizationMethod(method)
        except ValueError:
            raise typer.BadParameter(
                f"Unsupported quantization method: {method}. "
                f"Must be one of: {[m.value for m in QuantizationMethod]}."
            )

    parsed_algorithm: QuantizationAlgorithm | None = None
    if algorithm:
        try:
            parsed_algorithm = QuantizationAlgorithm(algorithm)
        except ValueError:
            raise typer.BadParameter(
                f"Unsupported algorithm: {algorithm}. "
                f"Must be one of: {[a.value for a in QuantizationAlgorithm]}."
            )

    if config is not None:
        config_path = str(
            cli_utils.resolve_and_fetch_config(
                try_get_config_name_for_alias(config, AliasType.QUANTIZE),
            )
        )
        parsed_config: QuantizationConfig = QuantizationConfig.from_yaml_and_arg_list(
            config_path, extra_args, logger=logger
        )

        if model:
            parsed_config.model.model_name = model
        if parsed_method is not None:
            parsed_config.method = parsed_method
        if parsed_algorithm is not None:
            parsed_config.algorithm = parsed_algorithm
        if output != "quantized_model":
            parsed_config.output_path = output
    else:
        if not model:
            raise typer.BadParameter(
                "Either --config must be provided or --model must be specified."
            )
        if parsed_method is None:
            raise typer.BadParameter(
                "--method is required when not using a config file. "
                f"Must be one of: {[m.value for m in QuantizationMethod]}."
            )

        kwargs: dict = {
            "model": ModelParams(model_name=model),
            "method": parsed_method,
            "output_path": output,
        }
        if parsed_algorithm is not None:
            kwargs["algorithm"] = parsed_algorithm

        parsed_config = QuantizationConfig(**kwargs)

    parsed_config.finalize_and_validate()

    # Configure environment and cleanup
    cli_utils.configure_common_env_vars()
    device_cleanup()

    try:
        with cli_utils.CONSOLE.status("Quantizing model...", spinner="dots"):
            result = oumi_quantize(parsed_config)
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        device_cleanup()
        raise

    if not result or not result.output_path:
        logger.error("Model quantization failed!")
        device_cleanup()
        return

    cli_utils.CONSOLE.print("Model quantized successfully!")
    cli_utils.CONSOLE.print(f"Output saved to: {result.output_path}")
    cli_utils.CONSOLE.print(f"Method: {result.quantization_method}")
    cli_utils.CONSOLE.print(f"Format: {result.format_type}")
    cli_utils.CONSOLE.print(
        f"Quantized size: {result.quantized_size_bytes / (1024**3):.2f} GB"
    )

    if result.additional_info:
        for key, value in result.additional_info.items():
            cli_utils.CONSOLE.print(f"  {key}: {value}")

    device_cleanup()
