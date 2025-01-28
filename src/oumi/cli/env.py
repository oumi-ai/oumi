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
import os
import platform


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


def _get_padded_table(
    kv: dict, key_title: str, value_title: str, padding: int = 5
) -> str:
    """Formats a key-value pair as a table with padding.

    Args:
        kv: The key-value pair to format.
        key_title: The title for the key column.
        value_title: The title for the value column.
        padding: The padding to use.

    Returns:
        str: The formatted table.
    """
    max_length = max(len(key) for key in kv.keys())
    formatted_kv = []
    for key, value in kv.items():
        k = "{0:{space}}".format(key, space=max_length + padding)
        formatted_kv.append(k + value)
    title_row = (
        "{0:{space}}".format(key_title, space=max_length + padding) + value_title + "\n"
    )
    return title_row + "\n".join(formatted_kv)


def env():
    """Prints information about the current environment."""
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
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "OUMI_EXTRA_DEPS_FILE",
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
        ]
    )
    package_versions = {
        package: _get_package_version(package, version_fallback)
        for package in core_packages
    }
    env_values = {env_var: os.getenv(env_var, env_var_fallback) for env_var in env_vars}
    print("----------Oumi environment information:----------\n")
    print(f"Oumi version: {_get_package_version('oumi', version_fallback)}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("\nInstalled dependencies:")
    print(_get_padded_table(package_versions, "PACKAGE", "VERSION"))

    if env_vars:
        print("\nEnvironment variables:")
        print(_get_padded_table(env_values, "VARIABLE", "VALUE"))

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        print("\nPyTorch information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(
                f"cuDNN version: {format_cudnn_version(torch.backends.cudnn.version())}"
            )
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"GPU type: {torch.cuda.get_device_name()}")
            total_memory_gb = float(torch.cuda.mem_get_info()[1]) / float(
                1024 * 1024 * 1024
            )
            print(f"GPU memory: {total_memory_gb:.1f}GB")
