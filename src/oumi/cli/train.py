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
import shutil
from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


def _handle_distributed_training(
    distributed_flag: bool,
    config_path: str,
    extra_args: list[str],
    level: str | None,
    verbose: bool,
) -> None:
    """Handle distributed training launch logic.

    Behavior:
    - If already under launcher: log debug, return (no-op)
    - If --distributed and multi-GPU: re-exec with torchrun
    - If --distributed and single-GPU: log info, return
    - If NOT --distributed and multi-GPU: log WARNING about unused GPUs
    - If NOT --distributed and single-GPU: return silently

    Args:
        distributed_flag: Whether --distributed was passed.
        config_path: Path to the configuration file.
        extra_args: Additional CLI arguments to pass through.
        level: The logging level.
        verbose: Whether verbose mode is enabled.
    """
    from oumi.utils.distributed_utils import is_under_distributed_launcher

    # Check if already under a launcher (torchrun or accelerate)
    if is_under_distributed_launcher():
        logger.debug("Already under distributed launcher, proceeding normally")
        return

    # Lazy import torch to avoid slow startup
    import torch

    # Detect available GPUs
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if distributed_flag:
        if gpu_count <= 1:
            logger.info("Single GPU detected, running without distributed launcher.")
            return

        # Re-exec with torchrun
        from oumi.core.cluster import detect_cluster_info

        run_info = detect_cluster_info()

        # Build the command
        # Check if torchrun is available, fallback to python -m torch.distributed.run
        torchrun_available = shutil.which("torchrun") is not None
        if torchrun_available:
            cmd = ["torchrun"]
        else:
            cmd = ["python", "-m", "torch.distributed.run"]

        cmd.extend([
            f"--nproc-per-node={run_info.gpus_per_node}",
            f"--nnodes={run_info.num_nodes}",
            f"--node-rank={run_info.node_rank}",
            f"--master-addr={run_info.master_address}",
            f"--master-port={run_info.master_port}",
            "-m", "oumi", "train",
            "-c", config_path,
        ])

        # Add level and verbose flags if set
        if level:
            cmd.extend(["--level", level])
        if verbose:
            cmd.append("--verbose")

        # Add extra args
        cmd.extend(extra_args)

        logger.info(f"Launching distributed training: {' '.join(cmd)}")
        os.execvp(cmd[0], cmd)  # Replaces process, never returns
    else:
        # No --distributed flag
        if gpu_count > 1:
            logger.warning(
                f"Multiple GPUs detected ({gpu_count}) but --distributed not set. "
                "Running on single GPU. Use 'oumi train --distributed -c config.yaml' "
                "for multi-GPU training."
            )


def train(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training."
        ),
    ],
    distributed: Annotated[
        bool,
        typer.Option(
            "--distributed", "-d",
            help="Auto-launch with torchrun for multi-GPU training. "
                 "Detects available GPUs and cluster environment automatically. "
                 "Safe to use even when already under a launcher (becomes no-op)."
        ),
    ] = False,
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Train a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        distributed: Auto-launch with torchrun for multi-GPU training.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Resolve config path first (before potential re-exec)
    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.TRAIN),
        )
    )

    # Handle distributed training (may re-exec with torchrun)
    _handle_distributed_training(distributed, config, extra_args, level, verbose)

    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import train as oumi_train
        from oumi.core.configs import TrainingConfig
        from oumi.core.distributed import set_random_seeds
        from oumi.utils.torch_utils import (
            device_cleanup,
            limit_per_process_memory,
        )
        # End imports

    cli_utils.configure_common_env_vars()

    parsed_config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    limit_per_process_memory()
    device_cleanup()
    set_random_seeds(
        parsed_config.training.seed, parsed_config.training.use_deterministic
    )

    # Run training
    oumi_train(parsed_config, verbose=verbose)

    device_cleanup()
