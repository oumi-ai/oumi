import importlib.metadata
import importlib.util
import platform
from typing import Optional


def _format_cudnn_version(v: Optional[int]) -> str:
    if v is None:
        return ""
    return ".".join(map(str, (v // 1000, v // 100 % 10, v % 100)))


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


def env():
    """Prints information about the current environment."""
    version_fallback = "<not installed>"
    core_packages = [
        "accelerate",
        "aiohttp",
        "bitsandbytes",
        "datasets",
        "diffusers",
        "einops",
        "jsonlines",
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
        "tqdm",
        "transformers",
        "trl",
        "typer",
        "vllm",
        "wandb",
    ]
    package_versions = {
        package: _get_package_version(package, version_fallback)
        for package in core_packages
    }
    padding = 5
    max_length = max(len(package) for package in package_versions.keys())
    formatted_versions = []
    for package, version in package_versions.items():
        k = "{0:{space}}".format(package, space=max_length + padding)
        formatted_versions.append(k + version)
    print("----------Oumi environment information:----------\n")
    print(f"Oumi version: {_get_package_version('oumi', version_fallback)}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}\n")
    print("Installed dependencies:")
    print("{0:{space}}".format("PACKAGE", space=max_length + padding) + "VERSION")
    print("\n".join(formatted_versions))

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        print("\nPyTorch information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"GPU type: {torch.cuda.get_device_name()}")
            print(
                "CUDNN version: "
                f"{_format_cudnn_version(torch.backends.cudnn.version())}"
            )
