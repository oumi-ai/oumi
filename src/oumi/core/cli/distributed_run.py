import enum
import os
from typing import Annotated, Final, NamedTuple, Optional

import typer

import oumi.core.cli.cli_utils as cli_utils
from oumi.utils.logging import logger

_MASTER_PORT_MIN_VALID_VALUE: Final[int] = 1024
_MASTER_PORT_MAX_VALID_VALUE: Final[int] = 65535

_SKY_ENV_VARS = {
    "SKYPILOT_NODE_RANK",
    "SKYPILOT_NODE_IPS",
    "SKYPILOT_NUM_GPUS_PER_NODE",
}


class _RunBackend(str, enum.Enum):
    SKY = "SkyPilot"
    POLARIS = "Polaris"


class WorldInfo(NamedTuple):
    num_nodes: int
    """Total number of nodes (machines)."""
    gpus_per_node: int
    """Number of GPU-s per node."""


class ProcessRunInfo:
    def __init__(
        self,
        node_rank: int,
        world_info: WorldInfo,
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


def detect_process_run_info(env: dict[str, str]) -> ProcessRunInfo:
    """Detects process run info.

    Uses known environment variables to detect common runtime parameters.

    Args:
        env: All environment variables.

    Returns:
        Process run info.
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
        backend = _RunBackend.SKY
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
            f"not found in teh list of nodes."
        )

    return ProcessRunInfo(
        node_rank=node_rank,
        world_info=WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=(oumi_master_address or node_ips[0]),
        master_port=8007,
    )


def torchrun(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training."
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Train a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        level: The logging level for the specified command.
    """
    run_info: ProcessRunInfo = detect_process_run_info(os.environ.copy())
    logger.info(f"run_info: {run_info}")
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    logger.info(f"extra_args: {extra_args}")

    raise NotImplementedError


def accelerate(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training."
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Train a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        level: The logging level for the specified command.
    """
    raise NotImplementedError
