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

"""JAX utilities for device inspection and tensor conversion."""

from typing import Any

import numpy as np

from oumi.utils.logging import logger

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


def torch_to_jax(tensor: Any) -> Any:
    """Converts a PyTorch tensor to a JAX array.

    Args:
        tensor: PyTorch tensor to convert.

    Returns:
        JAX array.

    Raises:
        RuntimeError: If JAX is not available.
    """
    if jnp is None:
        raise RuntimeError("JAX not available for tensor conversion")

    numpy_array = tensor.detach().cpu().numpy()
    return jnp.array(numpy_array)


def jax_to_torch(array: Any, device: str = "cpu") -> Any:
    """Converts a JAX array to a PyTorch tensor.

    Args:
        array: JAX array to convert.
        device: Target PyTorch device.

    Returns:
        PyTorch tensor.

    Raises:
        RuntimeError: If conversion fails.
    """
    import torch

    numpy_array = np.array(array)
    return torch.from_numpy(numpy_array).to(device)


def check_jax_devices() -> dict[str, Any]:
    """Returns information about available JAX devices.

    Returns:
        Dictionary with device counts, types, and TPU/GPU availability.
    """
    if jax is None:
        return {"error": "JAX not available"}

    devices = jax.devices()
    tpu_devices = [d for d in devices if "tpu" in str(d).lower()]
    gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

    return {
        "num_devices": len(devices),
        "device_types": [str(device.device_kind) for device in devices],
        "devices": [str(device) for device in devices],
        "has_tpus": len(tpu_devices) > 0,
        "num_tpus": len(tpu_devices),
        "has_gpus": len(gpu_devices) > 0,
        "num_gpus": len(gpu_devices),
    }


def setup_jax_for_performance() -> None:
    """Configures JAX for optimal inference performance."""
    if jax is None:
        logger.warning("JAX not available, cannot configure")
        return

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
    logger.info("JAX configured for performance")


def memory_usage_mb() -> float:
    """Returns current JAX device memory usage in MB.

    Returns:
        Memory usage in megabytes, or 0.0 if unavailable.
    """
    if jax is None:
        return 0.0

    try:
        devices = jax.devices()
        if devices:
            stats = devices[0].memory_stats()
            if stats and "bytes_in_use" in stats:
                return stats["bytes_in_use"] / (1024 * 1024)
    except Exception:
        pass

    return 0.0
