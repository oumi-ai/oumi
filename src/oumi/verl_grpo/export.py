# Copyright 2026 - Oumi
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

"""Exports a verl GRPO checkpoint as an HF model directory or a PEFT LoRA adapter.

verl's own model merger does the conversion, in verl's environment, as a subprocess.
This module never imports verl.

Run with `python -m oumi.verl_grpo.export --actor-dir DIR --output-dir DIR
--base-model MODEL --verl-python PATH`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

CHECKPOINT_TRACKER_FILENAME: Final = "latest_checkpointed_iteration.txt"

_SCRATCH_PREFIX: Final = ".verl_merge_"
_SHADOW_DIRNAME: Final = "actor_shadow"
_HF_DIRNAME: Final = "huggingface"
_HF_CONFIG_FILENAME: Final = "config.json"
_ADAPTER_DIRNAME: Final = "lora_adapter"
_ADAPTER_CONFIG_FILENAME: Final = "adapter_config.json"
_ADAPTER_WEIGHTS_FILENAME: Final = "adapter_model.safetensors"
_NON_ADAPTER_FILENAMES: Final = frozenset({"config.json", "generation_config.json"})
_STDERR_TAIL_CHARS: Final = 2000


class VerlExportError(RuntimeError):
    """Raised when a verl checkpoint can't be exported."""


def latest_checkpoint_actor_dir(checkpoint_root: Path) -> Path:
    """Returns `<root>/global_step_<N>/actor` for the step verl last checkpointed."""
    tracker = checkpoint_root / CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        raise VerlExportError(
            f"no '{CHECKPOINT_TRACKER_FILENAME}' in {checkpoint_root}"
        )
    step = tracker.read_text().strip()
    actor_dir = checkpoint_root / f"global_step_{step}" / "actor"
    if not actor_dir.is_dir():
        raise VerlExportError(f"no 'global_step_{step}/actor' in {checkpoint_root}")
    return actor_dir


def _is_weights_file(path: Path) -> bool:
    return path.name.startswith("model") and path.suffix == ".safetensors"


def _build_shadow_dir(actor_dir: Path, shadow: Path) -> Path:
    """Mirrors `actor_dir` with symlinks, minus `attn_implementation` in the HF config.

    verl's merger needs CUDA when the HF config requests a kernel such as
    flash_attention_2. A shadow keeps the checkpoint itself untouched.
    """
    shadow.mkdir()
    source = actor_dir.resolve()
    for entry in source.iterdir():
        if entry.name != _HF_DIRNAME or not entry.is_dir():
            (shadow / entry.name).symlink_to(entry)
            continue
        hf_shadow = shadow / _HF_DIRNAME
        hf_shadow.mkdir()
        for hf_entry in entry.iterdir():
            if hf_entry.name != _HF_CONFIG_FILENAME:
                (hf_shadow / hf_entry.name).symlink_to(hf_entry)
        config = json.loads((entry / _HF_CONFIG_FILENAME).read_text())
        config.pop("attn_implementation", None)
        (hf_shadow / _HF_CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
    return shadow


def _needs_shadow(actor_dir: Path) -> bool:
    config_path = actor_dir / _HF_DIRNAME / _HF_CONFIG_FILENAME
    if not config_path.is_file():
        return False
    return "attn_implementation" in json.loads(config_path.read_text())


def _run_merger(local_dir: Path, target_dir: Path, verl_python: str) -> None:
    env = dict(os.environ)
    interpreter_dir = Path(verl_python).parent
    if interpreter_dir != Path():
        env["PATH"] = f"{interpreter_dir}{os.pathsep}{env.get('PATH', '')}"
    command = [
        verl_python,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(target_dir),
    ]
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True)
    except OSError as exc:
        raise VerlExportError(f"can't run '{verl_python}': {exc}") from exc
    if result.returncode != 0:
        raise VerlExportError(
            f"verl's model merger exited with code {result.returncode}: "
            f"{result.stderr[-_STDERR_TAIL_CHARS:].strip()}"
        )


def _place_lora(
    adapter_dir: Path,
    extras_dir: Path,
    output_dir: Path,
    *,
    base_model: str,
    exclude_modules: str | None,
) -> None:
    """Copies the adapter from `adapter_dir` and non-weight files from `extras_dir`."""
    if (output_dir / "config.json").exists() or any(
        _is_weights_file(path) for path in output_dir.glob("*.safetensors")
    ):
        raise VerlExportError(
            f"'{output_dir}' already holds a full model; export the adapter elsewhere"
        )
    config = json.loads((adapter_dir / _ADAPTER_CONFIG_FILENAME).read_text())
    alpha = config.get("lora_alpha")
    if not isinstance(alpha, int | float) or alpha <= 0:
        raise VerlExportError(f"the adapter has an invalid lora_alpha: {alpha!r}")
    config["base_model_name_or_path"] = base_model
    if exclude_modules is not None:
        config["exclude_modules"] = exclude_modules

    (output_dir / _ADAPTER_CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
    shutil.copyfile(
        adapter_dir / _ADAPTER_WEIGHTS_FILENAME, output_dir / _ADAPTER_WEIGHTS_FILENAME
    )
    for path in extras_dir.iterdir():
        if (
            path.is_file()
            and not _is_weights_file(path)
            and path.name not in _NON_ADAPTER_FILENAMES
        ):
            shutil.copyfile(path, output_dir / path.name)


def _place_full_model(merged_dir: Path, output_dir: Path) -> None:
    has_weights = any(
        _is_weights_file(path) for path in merged_dir.glob("*.safetensors")
    )
    if not (merged_dir / "config.json").is_file() or not has_weights:
        raise VerlExportError(
            "verl's model merger wrote no config.json and model*.safetensors"
        )
    for path in merged_dir.iterdir():
        if path.is_file():
            shutil.move(path, output_dir / path.name)


def export_verl_checkpoint(
    actor_dir: Path,
    output_dir: Path,
    *,
    base_model: str,
    verl_python: str,
    exclude_modules: str | None = None,
) -> Path:
    """Exports the FSDP checkpoint in `actor_dir` to `output_dir`.

    A full-parameter checkpoint becomes an HF model directory. A LoRA checkpoint
    becomes a PEFT adapter, with the tokenizer files beside it.

    Args:
        actor_dir: The checkpoint's `global_step_<N>/actor` directory. Never written.
        output_dir: Where the export goes.
        base_model: Base model id or path, recorded in an adapter's config.
        verl_python: Python interpreter of the environment that has verl installed.
        exclude_modules: Regex for modules the adapter excludes. The merger drops
            this field, so it is written back when given.

    Returns:
        `output_dir`.

    Raises:
        VerlExportError: If the merger fails or its output is unusable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX, dir=output_dir))
    try:
        local_dir = actor_dir
        if _needs_shadow(actor_dir):
            local_dir = _build_shadow_dir(actor_dir, scratch / _SHADOW_DIRNAME)
        _run_merger(local_dir, scratch, verl_python)
        adapter_dir = scratch / _ADAPTER_DIRNAME
        if (adapter_dir / _ADAPTER_CONFIG_FILENAME).is_file():
            _place_lora(
                adapter_dir,
                scratch,
                output_dir,
                base_model=base_model,
                exclude_modules=exclude_modules,
            )
        else:
            _place_full_model(scratch, output_dir)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Export a verl GRPO checkpoint as an HF model or a LoRA adapter."
    )
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--verl-python",
        required=True,
        help="Python interpreter of the environment that has verl installed.",
    )
    parser.add_argument("--exclude-modules", default=None)
    args = parser.parse_args(argv)

    try:
        export_verl_checkpoint(
            args.actor_dir,
            args.output_dir,
            base_model=args.base_model,
            verl_python=args.verl_python,
            exclude_modules=args.exclude_modules,
        )
    except VerlExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
