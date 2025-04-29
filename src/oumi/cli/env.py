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

import importlib.metadata
import importlib.util
import logging
import os
import platform
from typing import Annotated, Optional

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.utils.cli_styling import StyleLevel, get_style_level, print_result
from oumi.utils.logging import logger


def _get_package_version(package_name: str, version_fallback: str) -> str:
    """Gets the version of the specified package.

    Args:
        package_name: The name of the package.
        version_fallback: The fallback version string.

    Returns:
        str: The version of the package, or a fallback string if the package is not
            installed.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return version_fallback


def env(
    styling: Annotated[
        Optional[str],
        typer.Option(
            "--styling",
            help="Set the CLI styling level (none, full)",
        ),
    ] = None,
    test_logs: Annotated[
        bool,
        typer.Option(
            "--test-logs",
            help="Display sample log messages at each level to test formatting",
        ),
    ] = False,
):
    """Prints information about the current environment and controls CLI styling."""

    # Handle styling option if provided
    if styling is not None:
        valid_levels = [level.value for level in StyleLevel]
        styling = styling.lower()

        if styling in valid_levels:
            os.environ["OUMI_STYLE_LEVEL"] = styling
            print_result(f"CLI styling set to '{styling}' for this session", True)
        else:
            print_result(
                f"Invalid styling level '{styling}'. Must be one of: {', '.join(valid_levels)}",
                False,
            )
    # Delayed imports
    from oumi.utils.torch_utils import format_cudnn_version
    # End imports

    version_fallback = "<not installed>"
    env_var_fallback = "<not set>"

    # All relevant environment vars.
    env_vars = sorted(
        [
            "ACCELERATE_DYNAMO_BACKEND",
            "ACCELERATE_DYNAMO_MODE",
            "ACCELERATE_DYNAMO_USE_FULLGRAPH",
            "ACCELERATE_DYNAMO_USE_DYNAMIC",
            "ACCELERATE_USE_FSDP",
            "CUDA_VISIBLE_DEVICES",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "OUMI_EXTRA_DEPS_FILE",
            "OUMI_NO_STYLE",
            "OUMI_SLURM_CONNECTIONS",
            "OUMI_STYLE_LEVEL",
            "OUMI_FORCE_EDITABLE_INSTALL",
            "OUMI_USE_SPOT_VM",
            "RANK",
            "WORLD_SIZE",
        ]
    )

    # All deps, excluding dev, docs, and gcp.
    core_packages = sorted(
        [
            "accelerate",
            "aiohttp",
            "bitsandbytes",
            "datasets",
            "diffusers",
            "einops",
            "jsonlines",
            "llama-cpp-python",
            "liger-kernel",
            "lm-eval",
            "numpy",
            "nvidia-ml-py",
            "omegaconf",
            "open_clip_torch",
            "pandas",
            "peft",
            "pexpect",
            "pillow",
            "pydantic",
            "responses",
            "sglang",
            "skypilot",
            "tensorboard",
            "timm",
            "torch",
            "torchdata",
            "torchvision",
            "tqdm",
            "transformers",
            "trl",
            "typer",
            "vllm",
            "wandb",
            "mlflow",
        ]
    )
    package_versions = {
        package: _get_package_version(package, version_fallback)
        for package in core_packages
    }
    env_values = {env_var: os.getenv(env_var, env_var_fallback) for env_var in env_vars}
    cli_utils.section_header("Oumi environment information:")
    env_table = Table(show_header=False, show_lines=False)
    env_table.add_row("Oumi version", _get_package_version("oumi", version_fallback))
    env_table.add_row("Python version", platform.python_version())
    env_table.add_row("Platform", platform.platform())

    # Add CLI styling information
    current_style = get_style_level()
    env_table.add_row("CLI styling", current_style.value)

    cli_utils.CONSOLE.print(env_table)
    cli_utils.section_header("Installed dependencies:")
    deps_table = Table(show_header=True, show_lines=False)
    deps_table.add_column("PACKAGE", justify="left")
    deps_table.add_column("VERSION", justify="left")
    for package, version in package_versions.items():
        deps_table.add_row(package, version)
    cli_utils.CONSOLE.print(deps_table)

    if env_vars:
        cli_utils.section_header("Environment variables:")
        env_var_table = Table(show_header=True, show_lines=False)
        env_var_table.add_column("VARIABLE", justify="left")
        env_var_table.add_column("VALUE", justify="left")
        for var in env_vars:
            env_var_table.add_row(var, env_values[var])
        cli_utils.CONSOLE.print(env_var_table)

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        cli_utils.section_header("PyTorch information:")
        cuda_table = Table(show_header=False, show_lines=False)
        cuda_table.add_row("CUDA available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            cuda_table.add_row("CUDA version", str(torch.version.cuda))
            cuda_table.add_row(
                "cuDNN version", format_cudnn_version(torch.backends.cudnn.version())
            )
            cuda_table.add_row("Number of GPUs", str(torch.cuda.device_count()))
            cuda_table.add_row("GPU type", torch.cuda.get_device_name())
            total_memory_gb = float(torch.cuda.mem_get_info()[1]) / float(
                1024 * 1024 * 1024
            )
            cuda_table.add_row("GPU memory", f"{total_memory_gb:.1f}GB")
        cli_utils.CONSOLE.print(cuda_table)

    # Test log formatting if requested
    if test_logs:
        from oumi.utils.logging import show_log_formatting

        cli_utils.section_header("Log Formatting Test:")

        # Temporarily set to DEBUG level to show all message types
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        show_log_formatting()
        logger.setLevel(original_level)
