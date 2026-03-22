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

"""JAX model utilities for checkpoint management and distributed setup.

Provides checkpoint loading/saving via orbax, weight conversion from PyTorch,
and multi-host JAX initialization following jax-llm-examples patterns.
"""

from pathlib import Path
from typing import Any

from oumi.utils.logging import logger

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    Mesh = None  # type: ignore[assignment]


def setup_tensor_parallelism(num_devices: int) -> Any | None:
    """Creates a JAX mesh for tensor parallelism.

    Creates a 3-axis mesh ("x", "y", "z") following jax-llm-examples conventions,
    with ``num_devices`` on the "x" axis.

    Args:
        num_devices: Number of devices for parallelism.

    Returns:
        JAX Mesh object, or None if JAX is not available.
    """
    if jax is None:
        logger.warning("JAX not available for tensor parallelism")
        return None

    import numpy as np

    if Mesh is None:
        raise RuntimeError("JAX Mesh not available. Reinstall JAX.")
    devices = jax.devices()[:num_devices]
    try:
        mesh = jax.make_mesh((num_devices, 1, 1), ("x", "y", "z"))
    except AttributeError:
        mesh = Mesh(np.array(devices).reshape(num_devices, 1, 1), ("x", "y", "z"))
    logger.info(f"Created JAX mesh with {num_devices} devices")
    return mesh


def load_checkpoint_orbax(
    checkpoint_path: str | Path,
    abstract_params: Any,
    shardings: Any,
) -> Any:
    """Loads a JAX checkpoint using orbax.

    Args:
        checkpoint_path: Path to the orbax checkpoint directory.
        abstract_params: Abstract parameter structure (from model.Weights.abstract).
        shardings: Sharding specification (from model.Weights.shardings).

    Returns:
        Loaded parameter pytree.

    Raises:
        ImportError: If orbax-checkpoint is not installed.
    """
    try:
        import orbax.checkpoint as ocp  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "orbax-checkpoint is required for JAX checkpoint loading. "
            "Install with: pip install orbax-checkpoint"
        ) from e

    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(
        str(checkpoint_path),
        target=abstract_params,
    )
    logger.info(f"Loaded JAX checkpoint from {checkpoint_path}")
    return params


def save_checkpoint_orbax(
    params: Any,
    checkpoint_path: str | Path,
) -> None:
    """Saves JAX model parameters using orbax.

    Args:
        params: Model parameters pytree to save.
        checkpoint_path: Path to save the checkpoint.

    Raises:
        ImportError: If orbax-checkpoint is not installed.
    """
    try:
        import orbax.checkpoint as ocp  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "orbax-checkpoint is required for JAX checkpoint saving. "
            "Install with: pip install orbax-checkpoint"
        ) from e

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(checkpoint_path), params)
    logger.info(f"Saved JAX checkpoint to {checkpoint_path}")


def convert_pytorch_to_jax_weights(
    pytorch_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Converts a PyTorch state dict to JAX-compatible parameters.

    Args:
        pytorch_state_dict: PyTorch model state dictionary.

    Returns:
        JAX-compatible parameters dictionary.
    """
    if jnp is None:
        raise RuntimeError("JAX is not available for weight conversion")

    jax_params = {}
    for key, tensor in pytorch_state_dict.items():
        if hasattr(tensor, "detach"):
            numpy_array = tensor.detach().cpu().numpy()
        else:
            numpy_array = tensor
        jax_params[key] = jnp.array(numpy_array)

    logger.info(f"Converted {len(jax_params)} PyTorch weights to JAX")
    return jax_params


def setup_multi_host_jax() -> bool:
    """Initializes JAX for multi-host distributed execution.

    Returns:
        True if setup successful, False otherwise.
    """
    if jax is None:
        logger.warning("JAX not available for multi-host setup")
        return False

    try:
        jax.distributed.initialize()
        logger.info(
            f"JAX multi-host setup: process {jax.process_index()}/{jax.process_count()}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to setup JAX multi-host: {e}")
        return False
