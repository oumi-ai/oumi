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

import collections
import subprocess
from pathlib import Path
from typing import Any

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


# Mapping from local-file extension to the HuggingFace ``datasets`` builtin
# loader name. Anything not in this map falls back to the HF Hub path.
_LOCAL_DATASET_LOADERS: dict[str, str] = {
    ".jsonl": "json",
    ".json": "json",
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "text",
}


def load_calibration_dataset(config: QuantizationConfig):
    """Load calibration data from either a HuggingFace Hub repo or a local file.

    Resolution order:
      1. If ``config.calibration_dataset`` is an existing local file with a
         recognized extension (``.jsonl``, ``.json``, ``.parquet``, ``.csv``,
         ``.tsv``, ``.txt``), load it via the matching ``datasets`` builtin
         loader. ``calibration_split`` is ignored — local files always live
         under the synthetic ``train`` split.
      2. Otherwise treat the value as a HF Hub repo id (or a directory with
         a custom dataset script) and pass it through to ``load_dataset``
         along with ``calibration_split``.

    Returns a HuggingFace ``Dataset`` truncated to ``calibration_samples``.
    """
    from datasets import load_dataset

    src = config.calibration_dataset
    src_path = Path(src).expanduser()

    if src_path.is_file():
        suffix = src_path.suffix.lower()
        if suffix not in _LOCAL_DATASET_LOADERS:
            raise ValueError(
                f"Calibration file '{src}' has unsupported extension '{suffix}'. "
                f"Supported: {sorted(_LOCAL_DATASET_LOADERS)}."
            )
        loader = _LOCAL_DATASET_LOADERS[suffix]
        logger.info(
            f"Loading local calibration data: {src_path} "
            f"(loader={loader}, samples={config.calibration_samples})"
        )
        return load_dataset(
            loader,
            data_files=str(src_path),
            split=f"train[:{config.calibration_samples}]",
        )

    logger.info(
        f"Loading calibration data: {src} "
        f"(split={config.calibration_split}, samples={config.calibration_samples})"
    )
    return load_dataset(
        src,
        split=f"{config.calibration_split}[:{config.calibration_samples}]",
    )


# Single-text columns recognized when no chat/instruction structure is found.
# Listed in preference order — first match wins.
_TEXT_COLUMNS = (
    "text",
    "content",
    "body",
    "prompt",
    "instruction",
    "input",
    "question",
    "query",
)


def calibration_row_to_text(row: dict[str, Any]) -> str:
    """Render a calibration sample to plaintext for ``llama-imatrix``.

    Recognizes:
      * Oumi/chat format: ``{"messages": [{"role": ..., "content": ...}, ...]}``
        — concatenates content fields.
      * Alpaca format: ``{"instruction": ..., "input": ..., "output": ...}``
        — concatenates instruction + input + output.
      * Q/A pair formats: ``{"request": ..., "response": ...}``,
        ``{"question": ..., "answer": ...}``, ``{"prompt": ..., "response": ...}``.
      * Plain-text columns: ``text``, ``content``, ``body``, etc.

    Returns an empty string if no recognized text could be extracted.
    """
    # Oumi/chat format.
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                elif isinstance(content, list):
                    # Multimodal turns: pick the text segments.
                    for seg in content:
                        if isinstance(seg, dict) and seg.get("type") == "text":
                            text = seg.get("text")
                            if isinstance(text, str) and text:
                                parts.append(text)
        if parts:
            return "\n\n".join(parts)

    # Alpaca-style instruction/output (with optional input).
    if "instruction" in row and "output" in row:
        chunks = [str(row["instruction"]).strip()]
        inp = row.get("input")
        if isinstance(inp, str) and inp.strip():
            chunks.append(inp.strip())
        chunks.append(str(row["output"]).strip())
        return "\n\n".join(c for c in chunks if c)

    # Q/A pair shapes.
    for q_key, a_key in (
        ("request", "response"),
        ("question", "answer"),
        ("prompt", "response"),
    ):
        q, a = row.get(q_key), row.get(a_key)
        if isinstance(q, str) and isinstance(a, str):
            return f"{q}\n\n{a}"

    # Single-text columns.
    for key in _TEXT_COLUMNS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def run_subprocess(
    cmd: list[str],
    *,
    log_prefix: str,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    tail_lines: int = 60,
) -> None:
    """Run a subprocess, streaming output line-by-line to oumi's logger.

    Each stdout/stderr line is logged as ``"[{log_prefix}] {line}"``. On
    non-zero exit, the last ``tail_lines`` of combined output are included
    in the raised RuntimeError so the caller doesn't lose error context.

    Args:
        cmd: Command and arguments. Passed to ``subprocess.Popen`` directly
            (no shell).
        log_prefix: Prepended to every emitted log line.
        timeout: Optional wall-clock timeout in seconds.
        env: Environment variables. ``None`` means inherit the parent's.
        cwd: Working directory.
        tail_lines: How many trailing output lines to include in the error
            message on failure.

    Raises:
        RuntimeError: If the subprocess exits non-zero or times out.
    """
    logger.info(f"[{log_prefix}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        bufsize=1,
    )
    tail: collections.deque[str] = collections.deque(maxlen=tail_lines)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            logger.info(f"[{log_prefix}] {line}")
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"{log_prefix} timed out after {timeout}s. Last output:\n"
            + "\n".join(tail)
        ) from e

    if rc != 0:
        raise RuntimeError(
            f"{log_prefix} failed (exit {rc}). Last {len(tail)} line(s):\n"
            + "\n".join(tail)
        )
