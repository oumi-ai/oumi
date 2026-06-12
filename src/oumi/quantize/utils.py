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

"""Common utilities for quantization operations."""

from pathlib import Path

from oumi.utils.logging import logger

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def warn_if_local_gpu_below_inference_capability(scheme) -> None:
    """Warn if the local GPU's compute capability is below the scheme's
    inference threshold.

    Quantization itself runs in higher precision and doesn't require the
    target SM, so we don't block. But the same-machine quant-and-serve
    workflow would otherwise burn hours of calibration before vLLM rejects
    the saved model — this gives the user an early signal.
    """
    import torch

    from oumi.quantize.constants import SCHEME_REGISTRY

    if not torch.cuda.is_available():
        return
    info = SCHEME_REGISTRY.get(scheme)
    if info is None:
        return
    major, minor = torch.cuda.get_device_capability()
    local_cc = major + minor / 10.0
    if local_cc < info.min_compute_capability:
        logger.warning(
            f"Local GPU compute capability is SM {local_cc:.1f} but scheme "
            f"'{scheme.value}' requires SM {info.min_compute_capability:.1f} "
            "for inference. Quantization will still run, but the resulting "
            "model cannot be served on this GPU. Confirm your serving target "
            f"supports SM {info.min_compute_capability:.1f}+ before continuing."
        )


def pop_with_override_warning(
    kwargs: dict, keys: tuple[str, ...], context: str
) -> None:
    """Drop ``keys`` from ``kwargs`` in place, warning for any that were set.

    Used when a quantizer needs to pass these kwargs explicitly to
    ``from_pretrained`` and a user-supplied value would silently lose.
    """
    for key in keys:
        if key in kwargs:
            logger.warning(
                f"Ignoring user-supplied model_kwargs['{key}']={kwargs[key]!r}: "
                f"{context} sets this explicitly during quantization."
            )
            kwargs.pop(key)


def assert_output_path_writable(output_path: str) -> None:
    """Fail fast if the output path is not writable.

    Saves the user from discovering a read-only / full disk after hours of
    calibration. Creates parent directories if needed.
    """
    import os
    import tempfile

    path = Path(output_path)
    if path.exists() and path.is_file():
        raise RuntimeError(
            f"Output path '{output_path}' exists and is a file, not a directory."
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Cannot create output directory '{output_path}': {e}"
        ) from e
    if not os.access(path, os.W_OK):
        raise RuntimeError(f"Output directory '{output_path}' is not writable.")
    try:
        with tempfile.NamedTemporaryFile(
            dir=path, prefix=".oumi_writable_probe_", delete=True
        ):
            pass
    except OSError as e:
        raise RuntimeError(
            f"Cannot write to output directory '{output_path}': {e}"
        ) from e


def get_directory_size(path: str) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    path_obj = Path(path)
    for file_path in path_obj.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    size = float(size_bytes)
    for unit in _SIZE_UNITS:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"
