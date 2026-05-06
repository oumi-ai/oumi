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
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_quantize_config
from oumi.utils.logging import logger


def _list_schemes_callback(value: bool) -> None:
    if not value:
        return

    from oumi.core.configs.quantization_config import QuantizationBackend
    from oumi.quantize.constants import SCHEME_REGISTRY

    _BACKEND_LABELS = {
        QuantizationBackend.LLM_COMPRESSOR: "LLM Compressor",
        QuantizationBackend.BNB: "BitsAndBytes",
    }

    table = Table(
        title="Available Quantization Schemes",
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Scheme", style="green")
    table.add_column("Backend", style="yellow")
    table.add_column("Default Algorithm", style="magenta")
    table.add_column("Calibration", style="cyan")
    table.add_column("Min GPU", style="dim")
    table.add_column("Description", style="white")

    for scheme, info in SCHEME_REGISTRY.items():
        calib = (
            "[yellow]yes[/yellow]" if info.needs_calibration else "[green]no[/green]"
        )
        table.add_row(
            scheme.value,
            _BACKEND_LABELS.get(info.backend, info.backend.value),
            info.default_algorithm.value,
            calib,
            f"SM {info.min_compute_capability}",
            info.description,
        )

    cli_utils.CONSOLE.print()
    cli_utils.CONSOLE.print(table)
    cli_utils.CONSOLE.print()
    cli_utils.CONSOLE.print(
        "[dim]Use --scheme <name> to select a scheme, "
        "e.g.: oumi quantize -c config.yaml --scheme fp8_dynamic[/dim]"
    )
    cli_utils.CONSOLE.print()
    raise typer.Exit(code=0)


def _list_algorithms_callback(value: bool) -> None:
    if not value:
        return

    from oumi.core.configs.quantization_config import QuantizationAlgorithm
    from oumi.quantize.constants import (
        ALGORITHM_REGISTRY,
        get_default_schemes_by_algorithm,
    )

    schemes_by_algo = get_default_schemes_by_algorithm()

    table = Table(
        title="Available Quantization Algorithms",
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Algorithm", style="green")
    table.add_column("Calibration", style="cyan")
    table.add_column("Default For Schemes", style="yellow")
    table.add_column("Description", style="white")

    for algo in QuantizationAlgorithm:
        # 'bnb' is auto-selected for bnb_* schemes and rejected for LLM
        # Compressor schemes; it is never user-selectable.
        if algo == QuantizationAlgorithm.BNB:
            continue
        info = ALGORITHM_REGISTRY[algo]
        if info.needs_calibration is None:
            calib_display = "[dim]depends on scheme[/dim]"
        elif info.needs_calibration:
            calib_display = "[yellow]yes[/yellow]"
        else:
            calib_display = "[green]no[/green]"
        default_schemes = (
            ", ".join(s.value for s in schemes_by_algo.get(algo, [])) or "-"
        )
        table.add_row(algo.value, calib_display, default_schemes, info.description)

    cli_utils.CONSOLE.print()
    cli_utils.CONSOLE.print(table)
    cli_utils.CONSOLE.print()
    cli_utils.CONSOLE.print(
        "[dim]Use --algorithm <name> to override, "
        "e.g.: oumi quantize -c config.yaml --algorithm gptq[/dim]"
    )
    cli_utils.CONSOLE.print()
    raise typer.Exit(code=0)


def quantize(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name (e.g. llama3.1-8b-fp8).",
            rich_help_panel="Options",
            autocompletion=complete_quantize_config,
        ),
    ],
    list_schemes: Annotated[
        bool,
        typer.Option(
            "--list-schemes",
            help="List all available quantization schemes and exit.",
            callback=_list_schemes_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    list_algorithms: Annotated[
        bool,
        typer.Option(
            "--list-algorithms",
            help="List all available quantization algorithms and exit.",
            callback=_list_algorithms_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    level: Annotated[
        cli_utils.LogLevel | None,
        typer.Option(
            "--log-level",
            "-log",
            help="Logging level.",
            show_default=False,
            show_choices=True,
            case_sensitive=False,
            callback=cli_utils.set_log_level,
            rich_help_panel="Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
            rich_help_panel="Options",
        ),
    ] = False,
    # Model overrides
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model.model_name",
            help="Model name or HuggingFace path.",
            rich_help_panel="Model",
        ),
    ] = None,
    # Quantization overrides
    scheme: Annotated[
        str | None,
        typer.Option(
            "--scheme",
            help=(
                "Compression scheme. "
                "LLM Compressor: fp8_dynamic, fp8_block, "
                "w4a16, w4a16_asym, w8a16. "
                "BitsAndBytes: bnb_nf4, bnb_fp4, bnb_int8."
            ),
            rich_help_panel="Quantization",
        ),
    ] = None,
    algorithm: Annotated[
        str | None,
        typer.Option(
            "--algorithm",
            help=(
                "Quantization algorithm: auto, rtn, gptq, awq. "
                "'auto' selects the best algorithm for the chosen scheme. "
                "Not applicable to bnb_* schemes (the BitsAndBytes algorithm "
                "is auto-selected)."
            ),
            rich_help_panel="Quantization",
        ),
    ] = None,
    # Output overrides
    output_path: Annotated[
        str | None,
        typer.Option(
            "--output_path",
            help="Output directory for the quantized model.",
            rich_help_panel="Output",
        ),
    ] = None,
):
    """Quantize a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for quantization.
        list_schemes: List all available quantization schemes.
        list_algorithms: List all available quantization algorithms.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        model_name: Model name or HuggingFace path.
        scheme: Compression scheme.
        algorithm: Quantization algorithm.
        output_path: Output directory for the quantized model.
    """
    # Auto-collect overrides from dot-notation options (e.g., --model.model_name)
    option_overrides = cli_utils.collect_config_overrides(ctx)
    # Parse any additional extra args from command line
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    # Combine: explicit options take precedence (added last)
    all_overrides = extra_args + option_overrides

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.QUANTIZE),
        )
    )

    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        from oumi import quantize as oumi_quantize
        from oumi.core.configs import QuantizationConfig
        from oumi.quantize.utils import format_size
        from oumi.utils.torch_utils import device_cleanup

        cli_utils.configure_common_env_vars()
        parsed_config: QuantizationConfig = QuantizationConfig.from_yaml_and_arg_list(
            config, all_overrides, logger=logger
        )

        # Apply non-dot-notation overrides (kept as direct assignments because
        # they are advertised as separate CLI flags rather than dot paths).
        if scheme is not None:
            parsed_config.scheme = scheme
        if algorithm is not None:
            parsed_config.algorithm = algorithm
        if output_path is not None:
            parsed_config.output_path = output_path

        parsed_config.finalize_and_validate()

    if verbose:
        parsed_config.print_config(logger)

    from oumi.telemetry import TelemetryManager

    TelemetryManager.get_instance().tags(
        model_name=parsed_config.model.model_name,
        quantization_scheme=parsed_config.scheme,
        quantization_algorithm=parsed_config.algorithm,
    )

    device_cleanup()
    try:
        with cli_utils.CONSOLE.status(
            "[green]Quantizing model...[/green]", spinner="dots"
        ):
            result = oumi_quantize(parsed_config)

        table = Table(
            title="Quantization Results",
            title_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Output Path", str(result.output_path))
        table.add_row("Backend", result.backend.value)
        table.add_row("Scheme", result.scheme.value)
        table.add_row("Format", result.format_type)
        table.add_row("Quantized Size", format_size(result.quantized_size_bytes))
        if result.additional_info:
            for key, value in result.additional_info.items():
                table.add_row(key, str(value))

        cli_utils.CONSOLE.print(table)
    finally:
        device_cleanup()
