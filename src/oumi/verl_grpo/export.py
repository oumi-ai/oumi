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

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from oumi.utils.logging import logger

CHECKPOINT_TRACKER_FILENAME: Final = "latest_checkpointed_iteration.txt"

_HF_DIRNAME: Final = "huggingface"
_HF_CONFIG_FILENAME: Final = "config.json"
_ADAPTER_DIRNAME: Final = "lora_adapter"
_ADAPTER_CONFIG_FILENAME: Final = "adapter_config.json"
_ADAPTER_WEIGHTS_FILENAME: Final = "adapter_model.safetensors"
_BASE_FILL_FILENAME: Final = "model-base-fill.safetensors"
_WEIGHT_INDEX_FILENAME: Final = "model.safetensors.index.json"


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


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise VerlExportError(f"can't read '{path}': {exc}") from exc


def _merger_input_dir(actor_dir: Path, shadow: Path) -> Path:
    """Returns the directory the merger reads: `actor_dir`, or a shadow of it.

    verl's merger needs CUDA when the HF config requests a kernel such as
    flash_attention_2. When it does, the shadow mirrors `actor_dir` with symlinks,
    except for an HF config without `attn_implementation`. The checkpoint itself
    is never written.
    """
    hf_dir = actor_dir.resolve() / _HF_DIRNAME
    if not (hf_dir / _HF_CONFIG_FILENAME).is_file():
        return actor_dir
    config = _read_json(hf_dir / _HF_CONFIG_FILENAME)
    if "attn_implementation" not in config:
        return actor_dir
    shadow.mkdir()
    for entry in hf_dir.parent.iterdir():
        if entry != hf_dir:
            (shadow / entry.name).symlink_to(entry)
    hf_shadow = shadow / _HF_DIRNAME
    hf_shadow.mkdir()
    for hf_entry in hf_dir.iterdir():
        if hf_entry.name != _HF_CONFIG_FILENAME:
            (hf_shadow / hf_entry.name).symlink_to(hf_entry)
    del config["attn_implementation"]
    (hf_shadow / _HF_CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
    return shadow


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
        result = subprocess.run(command, env=env)
    except OSError as exc:
        raise VerlExportError(f"can't run '{verl_python}': {exc}") from exc
    if result.returncode != 0:
        raise VerlExportError(
            f"verl's model merger exited with code {result.returncode}; "
            "see its output above"
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
    if (output_dir / _HF_CONFIG_FILENAME).exists() or any(
        _is_weights_file(path) for path in output_dir.glob("*.safetensors")
    ):
        raise VerlExportError(
            f"'{output_dir}' already holds a full model; export the adapter elsewhere"
        )
    config = _read_json(adapter_dir / _ADAPTER_CONFIG_FILENAME)
    if not (adapter_dir / _ADAPTER_WEIGHTS_FILENAME).is_file():
        raise VerlExportError(f"no '{_ADAPTER_WEIGHTS_FILENAME}' in {adapter_dir}")
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
            and path.name
            not in {
                _HF_CONFIG_FILENAME,
                "generation_config.json",
                "model.safetensors.index.json",
            }
        ):
            shutil.copyfile(path, output_dir / path.name)


def _tensor_names(path: Path) -> list[str]:
    with safe_open(path, framework="pt") as handle:
        return list(handle.keys())


def _exported_weight_map(output_dir: Path) -> dict[str, str]:
    index_path = output_dir / _WEIGHT_INDEX_FILENAME
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]
    return {
        name: path.name
        for path in output_dir.glob("*.safetensors")
        if _is_weights_file(path)
        for name in _tensor_names(path)
    }


def _fill_missing_base_tensors(output_dir: Path, base_model: str) -> None:
    """Adds the base tensors that the export lacks, as a shard of its own.

    Some architectures share tensors between layers, so the merger can omit tensors
    that serving engines still require. Only tensor names are compared.
    """
    base_dir = Path(base_model)
    if not base_dir.is_dir():
        base_dir = Path(
            snapshot_download(
                base_model,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
        )
    exported = _exported_weight_map(output_dir)
    missing_by_file: dict[Path, list[str]] = {}
    for path in sorted(base_dir.glob("*.safetensors")):
        missing = [name for name in _tensor_names(path) if name not in exported]
        if missing:
            missing_by_file[path] = missing
    if not missing_by_file:
        return

    filled = {}
    for path, names in missing_by_file.items():
        with safe_open(path, framework="pt") as handle:
            for name in names:
                filled[name] = handle.get_tensor(name).contiguous()
    save_file(filled, str(output_dir / _BASE_FILL_FILENAME), metadata={"format": "pt"})

    index_path = output_dir / _WEIGHT_INDEX_FILENAME
    index = (
        json.loads(index_path.read_text())
        if index_path.is_file()
        else {"metadata": {}, "weight_map": exported}
    )
    index["weight_map"].update(dict.fromkeys(filled, _BASE_FILL_FILENAME))
    fill_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in filled.values()
    )
    metadata = index.setdefault("metadata", {})
    if "total_size" in metadata:
        metadata["total_size"] += fill_bytes
    else:
        metadata["total_size"] = sum(
            path.stat().st_size for path in output_dir.glob("*.safetensors")
        )
    index_path.write_text(json.dumps(index, indent=2))
    logger.info(
        f"Filled {len(filled)} tensors missing from the export from '{base_model}': "
        f"{sorted(filled)[:5]}"
    )


def _place_full_model(merged_dir: Path, output_dir: Path) -> None:
    has_weights = any(
        _is_weights_file(path) for path in merged_dir.glob("*.safetensors")
    )
    if not (merged_dir / _HF_CONFIG_FILENAME).is_file() or not has_weights:
        raise VerlExportError(
            "verl's model merger wrote no config.json and model*.safetensors"
        )
    for path in merged_dir.iterdir():
        if path.is_file():
            shutil.move(path, output_dir / path.name)


def _is_0_7_1_lora_checkpoint(actor_dir: Path) -> bool:
    """Whether the adapter is already in `lora_adapter/`, as verl 0.7.1 saved it.

    The merger would write `lora_alpha: 0` for these, so they are placed directly.
    """
    return (actor_dir / _ADAPTER_DIRNAME / _ADAPTER_CONFIG_FILENAME).is_file() and (
        not (actor_dir / "lora_train_meta.json").exists()
    )


def export_verl_checkpoint(
    actor_dir: Path,
    output_dir: Path,
    *,
    base_model: str,
    verl_python: str,
    exclude_modules: str | None = None,
) -> Path:
    """Exports the FSDP checkpoint in `actor_dir` to `output_dir`.

    A full-parameter checkpoint becomes an HF model directory, with any tensors the
    merger omitted copied in from the base model. A LoRA checkpoint becomes a PEFT
    adapter, with the tokenizer files beside it.

    Args:
        actor_dir: The checkpoint's `global_step_<N>/actor` directory. Never written.
        output_dir: Where the export goes.
        base_model: Base model id or local directory. Recorded in an adapter's config;
            supplies the missing tensors of a full export.
        verl_python: Python interpreter of the environment that has verl installed.
        exclude_modules: Regex for modules the adapter excludes. The merger drops
            this field, so it is written back when given.

    Returns:
        `output_dir`.

    Raises:
        VerlExportError: If the merger fails or its output is unusable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_0_7_1_lora_checkpoint(actor_dir):
        if not (actor_dir / _HF_DIRNAME).is_dir():
            raise VerlExportError(f"no '{_HF_DIRNAME}' tokenizer files in {actor_dir}")
        _place_lora(
            actor_dir / _ADAPTER_DIRNAME,
            actor_dir / _HF_DIRNAME,
            output_dir,
            base_model=base_model,
            exclude_modules=exclude_modules,
        )
        return output_dir
    scratch = Path(tempfile.mkdtemp(prefix=".verl_merge_", dir=output_dir))
    try:
        local_dir = _merger_input_dir(actor_dir, scratch / "actor_shadow")
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
            _fill_missing_base_tensors(output_dir, base_model)
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
