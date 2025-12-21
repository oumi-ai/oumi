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

"""Cluster detection and distributed training environment utilities.

This module provides utilities for detecting the cluster environment
(SkyPilot, SLURM, Polaris, local machine) and retrieving information
needed for distributed training.
"""

import enum
import os
import subprocess
from typing import Any, Final, NamedTuple

from oumi.utils.logging import logger

# Port range [1024, 65535] is generally available
# for application use w/o root permissions (non-privileged)
MASTER_PORT_MIN_VALID_VALUE: Final[int] = 1024
MASTER_PORT_MAX_VALID_VALUE: Final[int] = 65535

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

DEFAULT_MASTER_ADDR: Final[str] = "127.0.0.1"
DEFAULT_MASTER_PORT: Final[int] = 8007


class ClusterBackend(str, enum.Enum):
    """Detected cluster backend type."""

    SKYPILOT = "SkyPilot"
    POLARIS = "Polaris"
    SLURM = "Slurm"
    LOCAL_MACHINE = "LocalMachine"


class WorldInfo(NamedTuple):
    """Information about the distributed training world."""

    num_nodes: int
    """Total number of nodes (machines)."""
    gpus_per_node: int
    """Number of GPUs per node."""


class ClusterInfo:
    """Information about the cluster environment for distributed training.

    This class contains all the information needed to launch distributed
    training with torchrun or accelerate.
    """

    def __init__(
        self,
        node_rank: int,
        world_info: WorldInfo,
        master_address: str,
        master_port: int,
        node_ips: list[str],
    ):
        """Initializes cluster info, and validates arguments.

        Args:
            node_rank: The rank of the current node.
            world_info: Information about the distributed world.
            master_address: The address of the master node.
            master_port: The port for distributed communication.
            node_ips: List of IP addresses of all nodes.

        Raises:
            ValueError: If any of the arguments are invalid.
        """
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
            master_port >= MASTER_PORT_MIN_VALID_VALUE
            and master_port <= MASTER_PORT_MAX_VALID_VALUE
        ):
            raise ValueError(
                f"Master port: {master_port} is outside of valid range: "
                f"[{MASTER_PORT_MIN_VALID_VALUE}, {MASTER_PORT_MAX_VALID_VALUE}]."
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
        """Number of GPUs per node."""
        return self._world_info.gpus_per_node

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs across all nodes."""
        return self._world_info.num_nodes * self._world_info.gpus_per_node

    @property
    def master_address(self) -> str:
        """Master address for distributed communication."""
        return self._master_address

    @property
    def node_ips(self) -> list[str]:
        """List of node IPs."""
        return self._node_ips

    @property
    def master_port(self) -> int:
        """Master port for distributed communication."""
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


def detect_cluster_info(env: dict[str, str] | None = None) -> ClusterInfo:
    """Detects cluster information from the environment.

    Automatically detects the cluster backend (SkyPilot, Polaris, SLURM, or
    local machine) and returns information needed for distributed training.

    Args:
        env: Environment variables dict. If None, uses os.environ.

    Returns:
        ClusterInfo with detected cluster configuration.

    Raises:
        ValueError: If required environment variables are missing or invalid.
        RuntimeError: If cluster detection fails or no GPUs are available.
    """
    if env is None:
        env = os.environ.copy()

    # Detect the cluster info depending on the runtime environment.
    # Each runtime environment is checked in order of priority.
    cluster_info = _detect_skypilot_cluster_info(env)

    if cluster_info is None:
        cluster_info = _detect_polaris_cluster_info(env)

    if cluster_info is None:
        cluster_info = _detect_slurm_cluster_info(env)

    if cluster_info is None:
        cluster_info = _detect_local_machine_cluster_info(env)

    if cluster_info is None:
        raise RuntimeError("Failed to detect cluster info!")

    # Extra verification logic to make sure that the detected cluster info is
    # consistent with the environment variables.
    _verify_cluster_info(cluster_info, env)

    logger.debug(f"Cluster info: {cluster_info}")

    return cluster_info


def _verify_cluster_info(cluster_info: ClusterInfo, env: dict[str, str]) -> None:
    """Verify detected cluster info against OUMI_* environment variables."""
    oumi_total_gpus: int | None = _get_optional_int_env_var("OUMI_TOTAL_NUM_GPUS", env)
    oumi_num_nodes: int | None = _get_optional_int_env_var("OUMI_NUM_NODES", env)
    oumi_master_address: str | None = env.get("OUMI_MASTER_ADDR", None)
    if oumi_master_address is not None and len(oumi_master_address) == 0:
        raise ValueError("Empty master address in 'OUMI_MASTER_ADDR'!")

    if len(cluster_info.node_ips) == 0:
        raise ValueError("Empty list of nodes!")

    if oumi_num_nodes is not None and oumi_num_nodes != cluster_info.num_nodes:
        raise ValueError(
            "Inconsistent number of nodes: "
            f"{cluster_info.num_nodes} vs {oumi_num_nodes} in 'OUMI_NUM_NODES'."
        )
    elif oumi_total_gpus is not None and (oumi_total_gpus != cluster_info.total_gpus):
        raise ValueError(
            "Inconsistent total number of GPUs: "
            f"{cluster_info.total_gpus} vs {oumi_total_gpus} "
            "in 'OUMI_TOTAL_NUM_GPUS'. "
            f"Nodes: {cluster_info.num_nodes}. GPUs per node: {cluster_info.gpus_per_node}."
        )
    elif oumi_master_address and oumi_master_address not in cluster_info.node_ips:
        raise ValueError(
            f"Master address '{oumi_master_address}' not found in the list of nodes."
        )


#
# Backend-specific detection functions
#
def _detect_polaris_cluster_info(env: dict[str, str]) -> ClusterInfo | None:
    """Detect cluster info in Polaris (PBS) environment."""
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

    return ClusterInfo(
        node_rank=node_rank,
        world_info=WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=node_ips[0],
        master_port=int(env.get(_MASTER_PORT_ENV, DEFAULT_MASTER_PORT)),
        node_ips=node_ips,
    )


def _detect_slurm_cluster_info(env: dict[str, str]) -> ClusterInfo | None:
    """Detect cluster info in SLURM environment."""
    import torch  # Importing torch takes time so only load it in this scenario.

    slurm_nodes_str = env.get("SLURM_NODELIST", None)
    if slurm_nodes_str is None:
        return None
    logger.debug("Running in Slurm environment!")
    for env_var_name in _SLURM_ENV_VARS:
        if env.get(env_var_name, None) is None:
            raise ValueError(
                f"Slurm environment variable '{env_var_name}' is not defined!"
            )
    if not slurm_nodes_str:
        raise ValueError("Empty value in the 'SLURM_NODELIST' environment variable!")
    # Parse slurm nodelist string (ex. "nid[001240-001241]") into a newline-separated
    # list of node IPs.
    nodes_str = subprocess.run(
        ["scontrol", "show", "hostnames", slurm_nodes_str],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout
    node_ips = _parse_nodes_str(nodes_str)
    if len(node_ips) == 0:
        raise RuntimeError("Empty list of nodes in 'SLURM_NODELIST'!")
    gpus_per_node = torch.cuda.device_count()

    node_rank = _get_optional_int_env_var("SLURM_NODEID", env)
    # If running on a single node, default to 0.
    if node_rank is None and len(node_ips) == 1:
        node_rank = 0
    if node_rank is None:
        raise ValueError(
            "Unable to determine node rank on a multi-node setup. "
            "'SLURM_NODEID' is not set."
        )

    return ClusterInfo(
        node_rank=node_rank,
        world_info=WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=node_ips[0],
        master_port=int(env.get(_MASTER_PORT_ENV, DEFAULT_MASTER_PORT)),
        node_ips=node_ips,
    )


def _detect_skypilot_cluster_info(env: dict[str, str]) -> ClusterInfo | None:
    """Detect cluster info in SkyPilot environment."""
    node_rank: int | None = _get_optional_int_env_var("SKYPILOT_NODE_RANK", env)
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

    return ClusterInfo(
        node_rank=node_rank,
        world_info=WorldInfo(num_nodes=len(node_ips), gpus_per_node=gpus_per_node),
        master_address=node_ips[0],
        master_port=int(env.get(_MASTER_PORT_ENV, DEFAULT_MASTER_PORT)),
        node_ips=node_ips,
    )


def _detect_local_machine_cluster_info(env: dict[str, str]) -> ClusterInfo:
    """Detect cluster info on local machine."""
    import torch  # Importing torch takes time so only load it in this scenario.

    # Attempt to produce a local configuration
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No supported distributed backends found and no GPUs on local machine!\n"
            "Possible solutions:\n"
            "  1. Ensure NVIDIA drivers are installed: nvidia-smi\n"
            "  2. Install PyTorch with CUDA support\n"
            "  3. For CPU-only training, use: oumi train -c <config> (without --distributed)"
        )

    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available > 0:
        oumi_num_nodes = 1
        oumi_master_address = env.get(_MASTER_ADDR_ENV, DEFAULT_MASTER_ADDR)
        oumi_master_port = int(env.get(_MASTER_PORT_ENV, DEFAULT_MASTER_PORT))
        node_rank = 0
        gpus_per_node = num_gpus_available
        node_ips = [oumi_master_address]
    else:
        raise RuntimeError("CUDA available but no GPUs found on local machine!")

    return ClusterInfo(
        node_rank=node_rank,
        world_info=WorldInfo(num_nodes=oumi_num_nodes, gpus_per_node=gpus_per_node),
        master_address=oumi_master_address,
        master_port=oumi_master_port,
        node_ips=node_ips,
    )


#
# Private helper functions to parse environment variables
#
def _get_optional_int_env_var(var_name: str, env: dict[str, str]) -> int | None:
    """Get an optional integer environment variable."""
    str_value = env.get(var_name, None)
    if str_value is None:
        return None

    try:
        int_value = int(str_value)
    except ValueError as e:
        raise ValueError(f"Environment variable '{var_name}' is not an integer!") from e
    return int_value


def _get_int_env_var(var_name: str, env: dict[str, str]) -> int:
    """Get a required integer environment variable."""
    int_value = _get_optional_int_env_var(var_name, env)
    if int_value is None:
        raise ValueError(f"Environment variable '{var_name}' is not defined!")
    return int_value


def _get_positive_int_env_var(var_name: str, env: dict[str, str]) -> int:
    """Get a required positive integer environment variable."""
    int_value = _get_int_env_var(var_name, env)
    if not (int_value > 0):
        raise ValueError(
            f"Environment variable '{var_name}' is not positive: {int_value}!"
        )
    return int_value


def _parse_nodes_str(nodes_str: str) -> list[str]:
    """Parse a newline/comma-separated string of node IPs."""
    node_ips = [x.strip() for line in nodes_str.split("\n") for x in line.split(",")]
    node_ips = [x for x in node_ips if len(x) > 0]
    return node_ips
