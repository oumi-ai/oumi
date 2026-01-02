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

"""Shared system information utilities.

This module provides functions for collecting system information
used by both telemetry and the CLI `oumi env` command.
"""

import importlib.metadata
import importlib.util
import os
import platform
from typing import Any

# Core packages to collect versions for (used by both telemetry and CLI)
CORE_PACKAGES = frozenset(
    [
        "accelerate",
        "bitsandbytes",
        "datasets",
        "deepspeed",
        "flash-attn",
        "liger-kernel",
        "llama-cpp-python",
        "lm-eval",
        "numpy",
        "omegaconf",
        "oumi",
        "peft",
        "sglang",
        "skypilot",
        "torch",
        "torchdata",
        "torchvision",
        "transformers",
        "triton",
        "trl",
        "vllm",
    ]
)


def get_package_version(package_name: str, fallback: str = "") -> str:
    """Get the version of an installed package.

    Args:
        package_name: The name of the package.
        fallback: The fallback version string if not installed.

    Returns:
        The version string, or fallback if not installed.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def get_package_versions(packages: frozenset[str] | None = None) -> dict[str, str]:
    """Get versions of relevant packages.

    Args:
        packages: Set of package names to check. Defaults to CORE_PACKAGES.

    Returns:
        Dictionary mapping package names to version strings.
    """
    packages = packages or CORE_PACKAGES
    versions = {}
    for pkg in packages:
        version = get_package_version(pkg)
        if version:
            versions[pkg] = version
    return versions


def get_platform_info() -> dict[str, str]:
    """Get OS and platform information.

    Returns:
        Dictionary containing OS, version, platform, Python version, and architecture.
    """
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
    }


def get_cpu_info() -> dict[str, Any]:
    """Get CPU information.

    Returns:
        Dictionary containing CPU count.
    """
    return {
        "cpu_count": os.cpu_count(),
    }


def get_gpu_info() -> dict[str, Any]:
    """Get GPU/accelerator information.

    Returns:
        Dictionary containing accelerator type, count, and details.
    """
    if importlib.util.find_spec("torch") is None:
        return {"accelerator_type": "none", "accelerator_count": 0}

    import torch

    accelerator_info: list[dict[str, Any]] = []
    accelerator_type = "none"

    # NVIDIA GPUs (CUDA)
    if torch.cuda.is_available():
        accelerator_type = "cuda"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accelerator_info.append(
                {
                    "name": props.name,
                    "memory_bytes": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    # Apple Silicon (MPS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator_type = "mps"
        accelerator_info.append({"name": "Apple Silicon", "memory_bytes": None})

    result: dict[str, Any] = {
        "accelerator_type": accelerator_type,
        "accelerator_count": len(accelerator_info),
    }

    if accelerator_info:
        result["accelerators"] = accelerator_info
        # Add convenience fields for CUDA
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            result["gpu_name"] = torch.cuda.get_device_name()
            result["gpu_memory_bytes"] = torch.cuda.mem_get_info()[1]

    return result


def get_system_info() -> dict[str, Any]:
    """Get comprehensive system information.

    Returns:
        Dictionary containing platform, CPU, GPU, and package version information.
    """
    info: dict[str, Any] = dict(get_platform_info())
    info.update(get_cpu_info())
    info.update(get_gpu_info())
    info["package_versions"] = get_package_versions()
    return info
