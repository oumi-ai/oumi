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

"""Deprecated distributed training launcher commands.

This module is deprecated. Use `oumi train --distributed` instead, or run
`torchrun -m oumi train -c config.yaml` directly.
"""

import copy
import os
import shutil
import sys
import time
from subprocess import Popen
from sys import stderr, stdout

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.utils.logging import logger


#
# Commands
#
def torchrun(
    ctx: typer.Context,
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Starts `torchrun` sub-process w/ automatically configured common params.

    .. deprecated::
        This command is deprecated. Use `oumi train --distributed -c config.yaml`
        instead, or run `torchrun -m oumi train -c config.yaml` directly.

    Args:
        ctx: The Typer context object.
        level: The logging level for the specified command.
    """
    logger.warning(
        "DEPRECATED: 'oumi distributed torchrun' is deprecated. "
        "Use 'oumi train --distributed -c config.yaml' instead, or run "
        "'torchrun -m oumi train -c config.yaml' directly."
    )

    # Lazy import to avoid importing oumi.core at CLI startup
    from oumi.core.cluster import ClusterInfo, detect_cluster_info

    try:
        run_info: ClusterInfo = detect_cluster_info()
    except (ValueError, RuntimeError):
        logger.exception("Failed to detect cluster info!")
        raise

    # In some environments (e.g., OLCF Frontier) the "torchrun" command isn't available.
    # In that case, use "python -m torch.distributed.run" instead,
    # which should be equivalent:
    # https://docs.pytorch.org/docs/stable/elastic/run.html#module-torch.distributed.run
    torchrun_available = shutil.which("torchrun") is not None

    try:
        cmds: list[str] = []
        args = copy.deepcopy(ctx.args)
        if (  # Fallback to `oumi train -c ...` for single-node with 1 GPU (OPE-1315).
            (run_info.num_nodes == 1 and run_info.gpus_per_node == 1)
            and len(args) >= 3
            and args[0] == "-m"
            and args[1] == "oumi"
            and args[2] == "train"
        ):
            logger.info(
                "Single GPU detected (1 node, 1 GPU). "
                "Bypassing torchrun for direct execution."
            )
            args.pop(0)  # Remove leading "-m".
            cmds = []
        else:
            cmds = (
                ["torchrun"]
                if torchrun_available
                else ["python", "-m", "torch.distributed.run"]
            ) + [
                f"--nnodes={run_info.num_nodes}",
                f"--node-rank={run_info.node_rank}",
                f"--nproc-per-node={run_info.gpus_per_node}",
                f"--master-addr={run_info.master_address}",
                f"--master-port={run_info.master_port}",
            ]
        cmds.extend(args)

        _run_subprocess(cmds, rank=run_info.node_rank)
    except Exception:
        logger.exception(
            f"`torchrun` failed (Rank: {run_info.node_rank})!\nCommands: {cmds}"
        )
        raise


def accelerate(
    ctx: typer.Context,
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Starts `accelerate` sub-process w/ automatically configured common params.

    .. deprecated::
        This command is deprecated. Use `oumi train --distributed -c config.yaml`
        instead, or run `accelerate launch -m oumi train -c config.yaml` directly.

    Args:
        ctx: The Typer context object.
        level: The logging level for the specified command.
    """
    logger.warning(
        "DEPRECATED: 'oumi distributed accelerate' is deprecated. "
        "Use 'oumi train --distributed -c config.yaml' instead, or run "
        "'accelerate launch -m oumi train -c config.yaml' directly."
    )

    # Lazy import to avoid importing oumi.core at CLI startup
    from oumi.core.cluster import ClusterInfo, detect_cluster_info

    try:
        run_info: ClusterInfo = detect_cluster_info()
    except (ValueError, RuntimeError):
        logger.exception("Failed to detect cluster info!")
        raise

    try:
        accelerate_subcommand: str | None = None
        extra_args = copy.deepcopy(ctx.args)
        if (
            len(extra_args) > 0
            and len(extra_args[0]) > 0
            and not extra_args[0].startswith("-")
        ):
            # Copy sub-commands like "launch" to insert them right after `accelerate`
            # ("accelerate launch ...")
            accelerate_subcommand = extra_args.pop(0)

        cmds: list[str] = (
            ["accelerate"]
            + ([accelerate_subcommand] if accelerate_subcommand is not None else [])
            + [
                f"--num_machines={run_info.num_nodes}",
                f"--machine_rank={run_info.node_rank}",
                f"--num_processes={run_info.total_gpus}",
                f"--main_process_ip={run_info.master_address}",
                f"--main_process_port={run_info.master_port}",
            ]
        )
        cmds.extend(extra_args)

        _run_subprocess(cmds, rank=run_info.node_rank)
    except Exception:
        logger.exception(f"`accelerate` failed (Rank: {run_info.node_rank})!")
        raise


#
# Helper functions
#
def _run_subprocess(cmds: list[str], *, rank: int) -> None:
    env_copy = os.environ.copy()

    start_time = time.perf_counter()
    logger.info(f"Running the command: {cmds}")

    p = Popen(
        cmds,
        env=env_copy,
        stdout=stdout,
        stderr=stderr,
        bufsize=1,
        universal_newlines=True,
    )
    rc = p.wait()
    duration_sec = time.perf_counter() - start_time
    duration_str = f"Duration: {duration_sec:.1f} sec"
    if rc != 0:
        logger.error(
            f"{cmds[0]} failed with exit code: {rc} ({duration_str}). Command: {cmds}"
        )
        sys.exit(rc)

    logger.info(f"Successfully completed! (Rank: {rank}. {duration_str})")
