import copy
import enum
import os
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


class _RunBackend(str, enum.Enum):
    SKYPILOT = "SkyPilot"
    POLARIS = "Polaris"


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
        }
        return repr(fields_dict)


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
    node_ips = [x.strip() for x in nodes_str.split("\n")]
    node_ips = [x for x in node_ips if len(x) > 0]
    return node_ips


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
    oumi_total_gpus: Optional[int] = _get_optional_int_env_var(
        "OUMI_TOTAL_NUM_GPUS", env
    )
    oumi_num_nodes: Optional[int] = _get_optional_int_env_var("OUMI_NUM_NODES", env)
    oumi_master_address: Optional[str] = env.get("OUMI_MASTER_ADDR", None)
    if oumi_master_address is not None and len(oumi_master_address) == 0:
        raise ValueError("Empty master address in 'OUMI_MASTER_ADDR'!")

    backend: Optional[_RunBackend] = None

    node_rank: Optional[int] = _get_optional_int_env_var("SKYPILOT_NODE_RANK", env)
    if node_rank is not None:
        backend = _RunBackend.SKYPILOT
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

    polaris_node_file = env.get("PBS_NODEFILE", None)
    if polaris_node_file is not None:
        if backend is not None:
            raise RuntimeError(
                f"Multiple backends detected: {_RunBackend.POLARIS} and {backend}!"
            )
        backend = _RunBackend.POLARIS
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

    if backend is None:
        raise RuntimeError("None of supported distributed backends found!")

    assert len(node_ips) > 0, "Empty list of nodes!"
    assert node_rank is not None

    if oumi_num_nodes is not None and oumi_num_nodes != len(node_ips):
        raise ValueError(
            "Inconsistent number of nodes: "
            f"{len(node_ips)} vs {oumi_num_nodes} in 'OUMI_NUM_NODES'."
        )
    elif oumi_total_gpus is not None and (
        oumi_total_gpus != len(node_ips) * gpus_per_node
    ):
        raise ValueError(
            "Inconsistent total number of GPUs: "
            f"{len(node_ips) * gpus_per_node} vs {oumi_total_gpus} "
            "in 'OUMI_TOTAL_NUM_GPUS'. "
            f"Nodes: {len(node_ips)}. GPU-s per node: {gpus_per_node}."
        )
    elif oumi_master_address and oumi_master_address not in node_ips:
        raise ValueError(
            f"Master address '{oumi_master_address}' "
            f"not found in the list of nodes."
        )

    result = _ProcessRunInfo(
        node_rank=node_rank,
        world_info=_WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=(oumi_master_address or node_ips[0]),
        master_port=8007,
    )
    return result


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
            f"{cmds[0]} failed with exit code: {rc} ({duration_str}). "
            f"Command: {cmds}"
        )
        sys.exit(rc)

    logger.info(f"Successfully completed! (Rank: {rank}. {duration_str})")


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

    try:
        cmds: list[str] = [
            "torchrun",
            f"--nnodes={run_info.num_nodes}",
            f"--node-rank={run_info.node_rank}",
            f"--nproc-per-node={run_info.gpus_per_node}",
            f"--master-addr={run_info.master_address}",
            f"--master-port={run_info.master_port}",
        ]
        cmds.extend(ctx.args)

        _run_subprocess(cmds, rank=run_info.node_rank)
    except Exception:
        logger.exception(f"`torchrun` failed (Rank: {run_info.node_rank})!")
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
