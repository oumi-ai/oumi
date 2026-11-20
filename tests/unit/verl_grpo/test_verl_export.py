import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from oumi.verl_grpo.export import export_verl_checkpoint, latest_checkpoint_actor_dir

# Stands in for `python -m verl.model_merger merge`, writing what the real one writes.
_FAKE_MERGER = """#!PYTHON
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
Path(os.environ["FAKE_ARGV"]).write_text(json.dumps(argv))
target = Path(argv[argv.index("--target_dir") + 1])
for name in ("config.json", "generation_config.json", "model.safetensors"):
    (target / name).write_text("{}")
(target / "tokenizer.json").write_text("{}")
if os.environ.get("FAKE_LORA"):
    adapter = target / "lora_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": None, "exclude_modules": None})
    )
    (adapter / "adapter_model.safetensors").write_text("")
"""


@pytest.fixture
def fake_merger(tmp_path, monkeypatch) -> str:
    path = tmp_path / "fake_python"
    path.write_text(_FAKE_MERGER.replace("PYTHON", sys.executable, 1))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    return str(path)


def _names(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


def test_full_fine_tune_exports_the_merged_model(tmp_path, fake_merger):
    out = export_verl_checkpoint(
        tmp_path / "actor",
        tmp_path / "out",
        base_model="org/base",
        verl_python=fake_merger,
    )

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[:4] == ["-m", "verl.model_merger", "merge", "--backend"]
    assert argv[argv.index("--local_dir") + 1] == str(tmp_path / "actor")
    assert _names(out) == {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
    }


def test_lora_exports_the_adapter_and_tokenizer(tmp_path, fake_merger, monkeypatch):
    monkeypatch.setenv("FAKE_LORA", "1")

    out = export_verl_checkpoint(
        tmp_path / "actor",
        tmp_path / "out",
        base_model="org/base",
        verl_python=fake_merger,
        exclude_modules="(.*vision_tower.*)",
    )

    assert _names(out) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
    }
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "org/base"
    assert config["exclude_modules"] == "(.*vision_tower.*)"


def test_a_merger_failure_raises_and_leaves_no_files(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(subprocess.CalledProcessError):
        export_verl_checkpoint(
            tmp_path / "actor",
            out,
            base_model="org/base",
            verl_python=str(shutil.which("false")),
        )
    assert list(out.iterdir()) == []


def test_latest_checkpoint_actor_dir(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6\n")
    assert latest_checkpoint_actor_dir(tmp_path) == tmp_path / "global_step_6/actor"
