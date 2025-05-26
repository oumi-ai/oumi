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

import copy
import enum
import os
import shutil
import sys
import time
from subprocess import Popen
from sys import stderr, stdout
from typing import Any, Final, NamedTuple, Optional

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.utils.logging import logger

# Port range [1024, 65535] is generally available
# for application use w/o root permissions (non-privileged)
_MASTER_PORT_MIN_VALID_VALUE: Final[int] = 1024
_MASTER_PORT_MAX_VALID_VALUE: Final[int] = 65535

_SKY_ENV_VARS = {
    "SKYPILOT_NODE_RANK",
    "SKYPILOT_NODE_IPS",
    "SKYPILOT_NUM_GPUS_PER_NODE",
}

_POLARIS_ENV_VARS = {
    "PBS_NODEFILE",
    "PBS_JOBID",
}

_SLURM_ENV_VARS = {
    "SLURM_NODELIST",
    "SLURM_JOBID",
}

_MASTER_ADDR_ENV = "MASTER_ADDRESS"
_MASTER_PORT_ENV = "MASTER_PORT"

_DEFAULT_MASTER_ADDR = "127.0.0.1"
_DEFAULT_MASTER_PORT = 8007


class _RunBackend(str, enum.Enum):
    SKYPILOT = "SkyPilot"
    POLARIS = "Polaris"
    LOCAL_MACHINE = "LocalMachine"


class _WorldInfo(NamedTuple):
    num_nodes: int
    """Total number of nodes (machines)."""
    gpus_per_node: int
    """Number of GPU-s per node."""


class _ProcessRunInfo:
    def __init__(
        self,
        node_rank: int,
        world_info: _WorldInfo,
        master_address: str,
        master_port: int,
        node_ips: list[str],
    ):
        """Initializes run info, and validates arguments."""
        if not (world_info.num_nodes > 0 and world_info.gpus_per_node > 0):
            raise ValueError(
                f"Non-positive number of nodes or GPUs per node: {world_info}"
            )
        elif not (node_rank >= 0 and node_rank < world_info.num_nodes):
            raise ValueError(
                f"Node rank {node_rank} is out of range: [0, {world_info.num_nodes})."
            )
        elif len(master_address) == 0:
            raise ValueError(f"Empty master address: {master_address}.")
        elif not (
            master_port >= _MASTER_PORT_MIN_VALID_VALUE
            and master_port <= _MASTER_PORT_MAX_VALID_VALUE
        ):
            raise ValueError(
                f"Master port: {master_port} is outside of valid range: "
                f"[{_MASTER_PORT_MIN_VALID_VALUE}, {_MASTER_PORT_MAX_VALID_VALUE}]."
            )

        self._world_info = world_info
        self._node_rank = int(node_rank)
        self._master_address = master_address
        self._master_port = master_port
        self._node_ips = node_ips

    @property
    def node_rank(self) -> int:
        """Node rank in the [0, num_nodes) range."""
        return self._node_rank

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (machines)."""
        return self._world_info.num_nodes

    @property
    def gpus_per_node(self) -> int:
        """Number of GPU-s per node."""
        return self._world_info.gpus_per_node

    @property
    def total_gpus(self) -> int:
        """Total number of nodes (machines)."""
        return self._world_info.num_nodes * self._world_info.gpus_per_node

    @property
    def master_address(self) -> str:
        """Master address."""
        return self._master_address

    @property
    def node_ips(self) -> list[str]:
        """List of node IPs."""
        return self._node_ips

    @property
    def master_port(self) -> int:
        """Master port."""
        return self._master_port

    def __repr__(self) -> str:
        """Defines how this class is properly printed."""
        fields_dict: dict[str, Any] = {
            "node_rank": self.node_rank,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "total_gpus": self.total_gpus,
            "master_address": self.master_address,
            "master_port": self.master_port,
            "node_ips": self.node_ips,
        }
        return repr(fields_dict)


#
# Comamnds
#
def torchrun(
    ctx: typer.Context,
    level: cli_utils.LOG_LEVEL_TYPE = None,
) -> None:
    """Starts `torchrun` sub-process w/ automatically configured common params.

    Args:
        ctx: The Typer context object.
        level: The logging level for the specified command.
    """
    try:
        run_info: _ProcessRunInfo = _detect_process_run_info(os.environ.copy())
    except (ValueError, RuntimeError):
        logger.exception("Failed to detect process run info!")
        raise

    # In some environments (e.g., OLCF Frontier) the "torchrun" command isn't available.
    # In that case, use "python -m torch.distributed.run" instead,
    # which should be equivalent:
    # https://docs.pytorch.org/docs/stable/elastic/run.html#module-torch.distributed.run
    torchrun_available = shutil.which("torchrun") is not None

    try:
        cmds: list[str] = (
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
        cmds.extend(ctx.args)

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

    Args:
        ctx: The Typer context object.
        level: The logging level for the specified command.
    """
    try:
        run_info: _ProcessRunInfo = _detect_process_run_info(os.environ.copy())
    except (ValueError, RuntimeError):
        logger.exception("Failed to detect process run info!")
        raise

    try:
        accelerate_subcommand: Optional[str] = None
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
def _detect_process_run_info(env: dict[str, str]) -> _ProcessRunInfo:
    """Detects process run info.

    Uses known environment variables to detect common runtime parameters.

    Args:
        env: All environment variables.

    Returns:
        Process run info.

    Raises:
        ValueError: If any of the required environment variables are missing or invalid.
        RuntimeError: If the node list is empty, or there are issues with backend
            detection.
    """
    # Detect the process run info depending on the runtime environment.
    # Each runtime environment is checked in the order of priority.
    process_run_info = _detect_skypilot_process_run_info(env)

    if process_run_info is None:
        process_run_info = _detect_polaris_process_run_info(env)

    if process_run_info is None:
        process_run_info = _detect_slurm_process_run_info(env)

    if process_run_info is None:
        process_run_info = _detect_local_machine_process_run_info(env)

    if process_run_info is None:
        raise RuntimeError("Failed to detect process run info!")

    # Extra verification logic to make sure that the detected process run info is
    # consistent with the environment variables.
    # Will raise an exception if the detected process run info is not consistent.
    _verify_process_run_info(process_run_info, env)

    return process_run_info


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


def _verify_process_run_info(run_info: _ProcessRunInfo, env: dict[str, str]) -> None:
    oumi_total_gpus: Optional[int] = _get_optional_int_env_var(
        "OUMI_TOTAL_NUM_GPUS", env
    )
    oumi_num_nodes: Optional[int] = _get_optional_int_env_var("OUMI_NUM_NODES", env)
    oumi_master_address: Optional[str] = env.get("OUMI_MASTER_ADDR", None)
    if oumi_master_address is not None and len(oumi_master_address) == 0:
        raise ValueError("Empty master address in 'OUMI_MASTER_ADDR'!")

    assert len(run_info.node_ips) > 0, "Empty list of nodes!"
    assert run_info.node_rank is not None

    if oumi_num_nodes is not None and oumi_num_nodes != run_info.num_nodes:
        raise ValueError(
            "Inconsistent number of nodes: "
            f"{run_info.num_nodes} vs {oumi_num_nodes} in 'OUMI_NUM_NODES'."
        )
    elif oumi_total_gpus is not None and (oumi_total_gpus != run_info.total_gpus):
        raise ValueError(
            "Inconsistent total number of GPUs: "
            f"{run_info.total_gpus} vs {oumi_total_gpus} "
            "in 'OUMI_TOTAL_NUM_GPUS'. "
            f"Nodes: {run_info.num_nodes}. GPU-s per node: {run_info.gpus_per_node}."
        )
    elif oumi_master_address and oumi_master_address not in run_info.node_ips:
        raise ValueError(
            f"Master address '{oumi_master_address}' not found in the list of nodes."
        )


#
# Parse environment variables
#
def _detect_polaris_process_run_info(env: dict[str, str]) -> Optional[_ProcessRunInfo]:
    polaris_node_file = env.get("PBS_NODEFILE", None)
    if polaris_node_file is None:
        return None

    logger.debug("Running in Polaris environment!")
    for env_var_name in _POLARIS_ENV_VARS:
        if env.get(env_var_name, None) is None:
            raise ValueError(
                f"Polaris environment variable '{env_var_name}' is not defined!"
            )
    if not polaris_node_file:
        raise ValueError("Empty value in the 'PBS_NODEFILE' environment variable!")
    with open(polaris_node_file) as f:
        nodes_str = f.read()
    node_ips = _parse_nodes_str(nodes_str)
    if len(node_ips) == 0:
        raise RuntimeError("Empty list of nodes in 'PBS_NODEFILE'!")
    gpus_per_node = 4  # Per Polaris spec.
    node_rank = _get_optional_int_env_var("PMI_RANK", env)
    if node_rank is None:
        node_rank = 0

    return _ProcessRunInfo(
        node_rank=node_rank,
        world_info=_WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=node_ips[0],
        master_port=_DEFAULT_MASTER_PORT,
        node_ips=node_ips,
    )


def _detect_slurm_process_run_info(env: dict[str, str]) -> Optional[_ProcessRunInfo]:
    nodes_str = env.get("SLURM_NODELIST", None)
    if nodes_str is None:
        return None
    logger.debug("Running in Slurm environment!")
    for env_var_name in _SLURM_ENV_VARS:
        if env.get(env_var_name, None) is None:
            raise ValueError(
                f"Slurm environment variable '{env_var_name}' is not defined!"
            )
    if not nodes_str:
        raise ValueError("Empty value in the 'SLURM_NODELIST' environment variable!")
    node_ips = _parse_nodes_str(nodes_str)
    if len(node_ips) == 0:
        raise RuntimeError("Empty list of nodes in 'PBS_NODEFILE'!")
    gpus_per_node = 8  # Per Frontier spec.
    node_rank = _get_optional_int_env_var("PMI_RANK", env)
    if node_rank is None:
        node_rank = 0

    return _ProcessRunInfo(
        node_rank=node_rank,
        world_info=_WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=node_ips[0],
        master_port=_DEFAULT_MASTER_PORT,
        node_ips=node_ips,
    )


def _detect_skypilot_process_run_info(env: dict[str, str]) -> Optional[_ProcessRunInfo]:
    node_rank: Optional[int] = _get_optional_int_env_var("SKYPILOT_NODE_RANK", env)
    if node_rank is None:
        return None

    logger.debug("Running in SkyPilot environment!")
    for env_var_name in _SKY_ENV_VARS:
        if env.get(env_var_name, None) is None:
            raise ValueError(
                f"SkyPilot environment variable '{env_var_name}' is not defined!"
            )
    node_ips = _parse_nodes_str(env.get("SKYPILOT_NODE_IPS", ""))
    if len(node_ips) == 0:
        raise RuntimeError("Empty list of nodes in 'SKYPILOT_NODE_IPS'!")
    gpus_per_node = _get_positive_int_env_var("SKYPILOT_NUM_GPUS_PER_NODE", env)

    return _ProcessRunInfo(
        node_rank=node_rank,
        world_info=_WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=node_ips[0],
        master_port=_DEFAULT_MASTER_PORT,
        node_ips=node_ips,
    )


def _detect_local_machine_process_run_info(env: dict[str, str]) -> _ProcessRunInfo:
    import torch  # Importing torch takes time so only load it in this scenario.

    # Attempt to produce a local configuration
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No supported distributed backends found and no GPUs on local machine!"
        )

    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available > 0:
        oumi_num_nodes = 1
        oumi_master_address = env.get(_MASTER_ADDR_ENV, _DEFAULT_MASTER_ADDR)
        oumi_master_port = int(env.get(_MASTER_PORT_ENV, _DEFAULT_MASTER_PORT))
        node_rank = 0
        gpus_per_node = num_gpus_available
        node_ips = [oumi_master_address]
        cli_utils.configure_common_env_vars()
    else:
        raise RuntimeError("CUDA available but no GPUs found on local machine!")

    return _ProcessRunInfo(
        node_rank=node_rank,
        world_info=_WorldInfo(num_nodes=oumi_num_nodes, gpus_per_node=gpus_per_node),
        master_address=oumi_master_address,
        master_port=oumi_master_port,
        node_ips=node_ips,
    )


#
# Private helper functions to parse environment variables
#
def _get_optional_int_env_var(var_name: str, env: dict[str, str]) -> Optional[int]:
    str_value = env.get(var_name, None)
    if str_value is None:
        return None

    try:
        int_value = int(str_value)
    except ValueError as e:
        raise ValueError(f"Environment variable '{var_name}' is not an integer!") from e
    return int_value


def _get_int_env_var(var_name: str, env: dict[str, str]) -> int:
    int_value = _get_optional_int_env_var(var_name, env)
    if int_value is None:
        raise ValueError(f"Environment variable '{var_name}' is not defined!")
    return int_value


def _get_positive_int_env_var(var_name: str, env: dict[str, str]) -> int:
    int_value = _get_int_env_var(var_name, env)
    if not (int_value > 0):
        raise ValueError(
            f"Environment variable '{var_name}' is not positive: {int_value}!"
        )
    return int_value


def _parse_nodes_str(nodes_str: str) -> list[str]:
    node_ips = [x.strip() for line in nodes_str.split("\n") for x in line.split(",")]
    node_ips = [x for x in node_ips if len(x) > 0]
    return node_ips
