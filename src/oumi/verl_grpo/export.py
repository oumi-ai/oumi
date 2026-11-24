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

"""Exports a verl GRPO checkpoint with verl's own model merger.

A full fine-tune becomes an HF model directory, and a LoRA run a PEFT adapter.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from oumi.verl_grpo.patches.gemma4_fill import fill_gemma4_kv_tensors


def latest_checkpoint_actor_dir(checkpoint_root: Path) -> Path:
    """Returns `<root>/global_step_<N>/actor` for the step verl last checkpointed."""
    step = (checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip()
    return checkpoint_root / f"global_step_{step}" / "actor"


def export_verl_checkpoint(
    actor_dir: Path,
    output_dir: Path,
    *,
    base_model: str,
    verl_python: str,
    exclude_modules: str | None = None,
) -> Path:
    """Exports the checkpoint in `actor_dir` to `output_dir`.

    Args:
        actor_dir: The checkpoint's `global_step_<N>/actor` directory.
        output_dir: Where the export goes.
        base_model: Base model id or path. Recorded in a LoRA adapter's config, and
            the source of the weights a Gemma 4 export lacks.
        verl_python: Python interpreter of the env that has verl installed.
        exclude_modules: Regex for modules a LoRA adapter excludes.

    Returns:
        `output_dir`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        merged = Path(tmp)
        subprocess.run(
            [
                verl_python,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(merged),
            ],
            check=True,
        )
        # For LoRA, the merger writes the adapter and a merged full model. Keep the
        # adapter, plus the tokenizer files.
        adapter = merged / "lora_adapter"
        if adapter.is_dir():
            config_path = adapter / "adapter_config.json"
            config = json.loads(config_path.read_text())
            config["base_model_name_or_path"] = base_model
            if exclude_modules is not None:
                # The merger drops it, and PEFT then wraps Gemma 4's vision and
                # audio layers.
                config["exclude_modules"] = exclude_modules
            config_path.write_text(json.dumps(config, indent=2))
            for path in merged.iterdir():
                if path.is_file() and not path.name.startswith(
                    ("model", "config.json", "generation_config.json")
                ):
                    shutil.copy(path, adapter / path.name)
            merged = adapter
        else:
            fill_gemma4_kv_tensors(merged, base_model)
        for path in merged.iterdir():
            shutil.move(path, output_dir / path.name)
    return output_dir
