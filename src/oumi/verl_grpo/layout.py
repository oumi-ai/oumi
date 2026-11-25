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

"""Moves a full model between a base's multimodal layout and the text-only one.

A dual-mode model such as Qwen3.5 trained with ``model.text_only`` saves only its
language model. verl 0.9.1 and vLLM build such a model from its multimodal config,
so a text-only parent is filled out from its base into a new checkpoint before
training, and an export of a ``text_only`` run is cut back to the text-only layout.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Final

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from oumi.utils.logging import logger

INDEX_FILENAME: Final = "model.safetensors.index.json"
_FILL_FILENAME: Final = "model-base-layout.safetensors"
_SINGLE_FILENAME: Final = "model.safetensors"
_PARENT_FILENAME: Final = "model-parent.safetensors"
# Where a multimodal HF model keeps its language model.
_LANGUAGE_MODEL_PREFIXES: Final = ("model.language_model.", "lm_head.")


def weight_map(model_dir: Path) -> dict[str, str]:
    """Maps each tensor name to its HF weight shard."""
    index = model_dir / INDEX_FILENAME
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    result = {}
    for path in model_dir.glob("model*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            result.update(dict.fromkeys(handle.keys(), path.name))
    return result


def prepare_for_verl(parent_dir: Path, base_dir: Path, output_dir: Path) -> Path:
    """Returns a checkpoint of the parent that verl can train from.

    A parent with every base tensor and the base's architectures is returned as is.
    Otherwise `output_dir` gets the parent's files, the base tensors it lacks in a
    shard of their own, and, when its architectures differ, such as a text-only save
    of a multimodal base, the base's config and the tokenizer and processor files it
    lacks. Weight files are hard-linked where possible; `parent_dir` is not changed.

    Returns:
        `parent_dir` or `output_dir`.
    """
    base_map = weight_map(base_dir)
    parent_map = weight_map(parent_dir)
    missing = [name for name in base_map if name not in parent_map]
    config = json.loads((parent_dir / "config.json").read_text())
    base_config = json.loads((base_dir / "config.json").read_text())
    relayout = config.get("architectures") != base_config.get("architectures")
    if not missing and not relayout:
        return parent_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    # Loaders read a lone model.safetensors and skip the index beside it.
    renames = {_SINGLE_FILENAME: _PARENT_FILENAME}
    for path in parent_dir.iterdir():
        if not path.is_file() or path.name == INDEX_FILENAME:
            continue
        target = output_dir / renames.get(path.name, path.name)
        # Only weights are linked: the base's config may overwrite the rest.
        if path.suffix == ".safetensors":
            _link_or_copy(path, target)
        else:
            shutil.copy(path, target)
    output_map = {n: renames.get(s, s) for n, s in parent_map.items()}
    if missing:
        filled = {}
        for shard in sorted({base_map[name] for name in missing}):
            with safe_open(base_dir / shard, framework="pt") as handle:
                filled.update(
                    {n: handle.get_tensor(n) for n in missing if base_map[n] == shard}
                )
        save_file(filled, str(output_dir / _FILL_FILENAME), metadata={"format": "pt"})
        output_map.update(dict.fromkeys(filled, _FILL_FILENAME))
    _write_index(output_dir, output_map)
    if relayout:
        for path in base_dir.glob("*"):
            if path.suffix in (".json", ".jinja") and not path.name.startswith("model"):
                if path.name == "config.json" or not (output_dir / path.name).exists():
                    shutil.copy(path, output_dir / path.name)
    logger.info(
        f"Filled {len(missing)} tensors{' and the layout' if relayout else ''} "
        f"from '{base_dir}' into '{output_dir}'."
    )
    return output_dir


def to_text_only(model_dir: Path) -> bool:
    """Cuts a multimodal model down to its language model, as a text-only run saves it.

    A model without a separate text config is left alone.

    Returns:
        Whether the model was cut down.
    """
    config = AutoConfig.from_pretrained(model_dir)
    text_config = config.get_text_config()
    if text_config is config:
        return False
    model_map = weight_map(model_dir)
    kept = {
        n: s for n, s in model_map.items() if n.startswith(_LANGUAGE_MODEL_PREFIXES)
    }
    for shard in sorted(set(model_map.values())):
        names = [n for n, s in kept.items() if s == shard]
        if not names:
            (model_dir / shard).unlink()
        elif len(names) < sum(s == shard for s in model_map.values()):
            with safe_open(model_dir / shard, framework="pt") as handle:
                tensors = {n: handle.get_tensor(n) for n in names}
            save_file(tensors, str(model_dir / shard), metadata={"format": "pt"})
    if (model_dir / INDEX_FILENAME).is_file():
        _write_index(model_dir, kept)
    text_config.architectures = [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[text_config.model_type]
    ]
    text_config.save_pretrained(model_dir)
    logger.info(
        f"Kept the {len(kept)} language model tensors of {len(model_map)} in "
        f"'{model_dir}'."
    )
    return True


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_index(model_dir: Path, model_map: dict[str, str]) -> None:
    total_size = sum((model_dir / s).stat().st_size for s in set(model_map.values()))
    (model_dir / INDEX_FILENAME).write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": model_map},
            indent=2,
        )
    )
