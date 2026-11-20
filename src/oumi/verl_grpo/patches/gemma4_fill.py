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

"""Puts back the Gemma 4 weights that verl's merged export leaves out.

Gemma 4's KV-shared layers have no ``k_proj``, ``v_proj`` or ``k_norm`` in the HF
model, so verl's merger omits them. vLLM still expects them and refuses the export.
"""

import json
import re
from pathlib import Path
from typing import Final

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from oumi.utils.logging import logger

_FILL_FILENAME: Final = "model-base-fill.safetensors"
_INDEX_FILENAME: Final = "model.safetensors.index.json"
_KV_TENSOR = re.compile(r"\.self_attn\.(k_proj|v_proj|k_norm)\.weight$")


def fill_gemma4_kv_tensors(merged_dir: Path, base_model: str) -> None:
    """Copies the base's KV tensors that a merged Gemma 4 model lacks into it.

    They go into a shard of their own, listed in the weight index. Other models are
    left alone.
    """
    config = json.loads((merged_dir / "config.json").read_text())
    if config.get("model_type") not in ("gemma4", "gemma4_text"):
        return
    base_dir = Path(base_model)
    if not base_dir.is_dir():
        base_dir = Path(
            snapshot_download(
                base_model, allow_patterns=["model*.safetensors", _INDEX_FILENAME]
            )
        )
    base_map = _weight_map(base_dir)
    weight_map = _weight_map(merged_dir)
    missing = [n for n in base_map if n not in weight_map and _KV_TENSOR.search(n)]
    if not missing:
        return

    filled = {}
    for shard in sorted({base_map[name] for name in missing}):
        with safe_open(base_dir / shard, framework="pt") as handle:
            filled.update(
                {n: handle.get_tensor(n) for n in missing if base_map[n] == shard}
            )
    save_file(filled, str(merged_dir / _FILL_FILENAME), metadata={"format": "pt"})
    weight_map.update(dict.fromkeys(filled, _FILL_FILENAME))
    total_size = sum((merged_dir / s).stat().st_size for s in set(weight_map.values()))
    (merged_dir / _INDEX_FILENAME).write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
    )
    logger.info(f"Filled {len(filled)} Gemma 4 KV tensors from '{base_model}'.")


def _weight_map(model_dir: Path) -> dict[str, str]:
    """Maps each tensor name to its HF weight shard."""
    index = model_dir / _INDEX_FILENAME
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    weight_map = {}
    for path in model_dir.glob("model*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            weight_map.update(dict.fromkeys(handle.keys(), path.name))
    return weight_map
