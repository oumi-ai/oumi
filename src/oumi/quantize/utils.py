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

from oumi.core.configs import QuantizationConfig
from oumi.core.configs.quantization_config import QuantizationScheme
from oumi.utils.logging import logger

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")

# kwargs that quantizers must control directly when calling from_pretrained;
# user-supplied values would silently lose. See load_model_and_tokenizer.
_FORCED_MODEL_KWARGS = ("device_map", "torch_dtype", "quantization_config")


def warn_if_local_gpu_below_inference_capability(
    scheme: QuantizationScheme, min_compute_capability: float
) -> None:
    """Warn if the local GPU's compute capability is below ``min_compute_capability``.

    Quantization itself runs in higher precision and doesn't require the
    target SM, so we don't block. But the same-machine quant-and-serve
    workflow would otherwise burn hours of calibration before vLLM rejects
    the saved model — this gives the user an early signal.
    """
    import torch

    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    local_cc = major + minor / 10.0
    if local_cc < min_compute_capability:
        logger.warning(
            f"Local GPU compute capability is SM {local_cc:.1f} but scheme "
            f"'{scheme.value}' requires SM {min_compute_capability:.1f} "
            "for inference. Quantization will still run, but the resulting "
            "model cannot be served on this GPU. Confirm your serving target "
            f"supports SM {min_compute_capability:.1f}+ before continuing."
        )


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


def load_model_and_tokenizer(config: QuantizationConfig, **forced_kwargs):
    """Load model + tokenizer for a quantizer, owning the load-time kwargs.

    ``forced_kwargs`` are passed directly to ``from_pretrained``. Any
    overlapping keys in ``config.model.model_kwargs`` are dropped with a
    warning so the quantizer's choice always wins.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = dict(config.model.model_kwargs or {})
    for key in _FORCED_MODEL_KWARGS:
        if key in model_kwargs:
            logger.warning(
                f"Ignoring user-supplied model_kwargs[{key!r}]={model_kwargs[key]!r}: "
                "quantization sets this explicitly."
            )
            model_kwargs.pop(key)
    model_kwargs["trust_remote_code"] = config.model.trust_remote_code

    forced = {"device_map": "auto", **forced_kwargs}
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        **forced,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name or config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
        **(config.model.tokenizer_kwargs or {}),
    )
    return model, tokenizer


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
