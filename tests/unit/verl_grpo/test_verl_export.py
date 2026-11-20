import json
import stat
import sys
from pathlib import Path

import pytest

from oumi.verl_grpo.errors import VerlExportError
from oumi.verl_grpo.export import (
    export_verl_checkpoint,
    latest_checkpoint_actor_dir,
    main,
)

_FAKE_MERGER = """#!PYTHON
import json, os, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

argv = sys.argv
record = Path(os.environ["FAKE_RECORD"])
(record / "argv.json").write_text(json.dumps(argv))
local_dir = Path(argv[argv.index("--local_dir") + 1])
shadow_config = local_dir / "huggingface" / "config.json"
if shadow_config.exists():
    (record / "shadow_config.json").write_text(shadow_config.read_text())
target = Path(argv[argv.index("--target_dir") + 1])
mode = os.environ["FAKE_MERGE"]
if mode == "fail":
    print("merger exploded", file=sys.stderr)
    sys.exit(3)
(target / "tokenizer.json").write_text("{}")
if mode == "full":
    (target / "config.json").write_text("{}")
    save_file({"a": torch.zeros(2)}, str(target / "model.safetensors"))
else:
    (target / "tokenizer_config.json").write_text("{}")
    for name in ("config.json", "generation_config.json", "model.safetensors"):
        (target / name).write_text("{}")
    (target / "model.safetensors.index.json").write_text("{}")
    adapter = target / "lora_adapter"
    adapter.mkdir()
    alpha = 0 if mode == "lora_alpha0" else 16
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": None,
                "exclude_modules": None,
                "lora_alpha": alpha,
            }
        )
    )
    save_file({"w": torch.zeros(2)}, str(adapter / "adapter_model.safetensors"))
"""


@pytest.fixture
def fake_merger(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "bin" / "fake_python"
    path.parent.mkdir()
    path.write_text(_FAKE_MERGER.replace("PYTHON", sys.executable, 1))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "record").mkdir()
    monkeypatch.setenv("FAKE_RECORD", str(tmp_path / "record"))
    monkeypatch.setenv("FAKE_MERGE", "full")
    return path


@pytest.fixture
def actor(tmp_path) -> Path:
    actor_dir = tmp_path / "ckpt" / "global_step_6" / "actor"
    (actor_dir / "huggingface").mkdir(parents=True)
    (actor_dir / "huggingface" / "config.json").write_text(
        json.dumps({"attn_implementation": "flash_attention_2", "hidden_size": 8})
    )
    (actor_dir / "fsdp_config.json").write_text("{}")
    return actor_dir


def _recorded_argv(tmp_path: Path) -> list[str]:
    return json.loads((tmp_path / "record" / "argv.json").read_text())


def test_full_model_export_runs_verl_s_merger_and_places_the_model(
    tmp_path, fake_merger
):
    actor_dir = tmp_path / "plain_actor"
    actor_dir.mkdir()

    out = export_verl_checkpoint(
        actor_dir, tmp_path / "out", base_model="org/base", verl_python=str(fake_merger)
    )

    argv = _recorded_argv(tmp_path)
    assert argv[1:6] == ["-m", "verl.model_merger", "merge", "--backend", "fsdp"]
    assert argv[argv.index("--local_dir") + 1] == str(actor_dir)
    assert {p.name for p in out.iterdir()} == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }


def test_lora_export_places_the_adapter_and_tokenizer_only(
    tmp_path, actor, fake_merger, monkeypatch
):
    monkeypatch.setenv("FAKE_MERGE", "lora")

    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model="org/base", verl_python=str(fake_merger)
    )

    assert {p.name for p in out.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "org/base"
    assert config["exclude_modules"] is None


def test_an_adapter_with_zero_alpha_is_refused(
    tmp_path, actor, fake_merger, monkeypatch
):
    monkeypatch.setenv("FAKE_MERGE", "lora_alpha0")

    with pytest.raises(VerlExportError, match="lora_alpha"):
        export_verl_checkpoint(
            actor, tmp_path / "out", base_model="org/base", verl_python=str(fake_merger)
        )


def test_lora_export_refuses_an_output_dir_holding_a_full_model(
    tmp_path, actor, fake_merger, monkeypatch
):
    monkeypatch.setenv("FAKE_MERGE", "lora")
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text("{}")

    with pytest.raises(VerlExportError, match="full model"):
        export_verl_checkpoint(
            actor, out, base_model="org/base", verl_python=str(fake_merger)
        )


def test_merger_failure_raises_with_its_exit_code_and_leaves_no_scratch(
    tmp_path, actor, fake_merger, monkeypatch, capfd
):
    monkeypatch.setenv("FAKE_MERGE", "fail")
    out = tmp_path / "out"

    with pytest.raises(VerlExportError, match="exited with code 3") as exc_info:
        export_verl_checkpoint(
            actor, out, base_model="org/base", verl_python=str(fake_merger)
        )

    assert "see its output above" in str(exc_info.value)
    assert "merger exploded" in capfd.readouterr().err
    assert list(out.iterdir()) == []


def test_latest_checkpoint_follows_the_tracker(tmp_path):
    (tmp_path / "global_step_3" / "actor").mkdir(parents=True)
    (tmp_path / "global_step_6" / "actor").mkdir(parents=True)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6\n")

    assert latest_checkpoint_actor_dir(tmp_path) == tmp_path / "global_step_6" / "actor"


def test_latest_checkpoint_without_tracker_or_step_dir_raises(tmp_path):
    with pytest.raises(VerlExportError, match="latest_checkpointed_iteration.txt"):
        latest_checkpoint_actor_dir(tmp_path)

    (tmp_path / "latest_checkpointed_iteration.txt").write_text("9")
    with pytest.raises(VerlExportError, match="global_step_9"):
        latest_checkpoint_actor_dir(tmp_path)


def test_cli_returns_zero_on_success_and_one_on_error(
    tmp_path, actor, fake_merger, monkeypatch, capsys
):
    args = [
        "--actor-dir",
        str(actor),
        "--output-dir",
        str(tmp_path / "out"),
        "--base-model",
        "org/base",
        "--verl-python",
        str(fake_merger),
    ]
    assert main(args) == 0

    monkeypatch.setenv("FAKE_MERGE", "fail")
    assert main(args) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_merge_runs_from_a_shadow_dir_without_attn_implementation(
    tmp_path, actor, fake_merger
):
    before = sorted(p.relative_to(actor) for p in actor.rglob("*"))

    export_verl_checkpoint(
        actor, tmp_path / "out", base_model="org/base", verl_python=str(fake_merger)
    )

    assert sorted(p.relative_to(actor) for p in actor.rglob("*")) == before
    argv = _recorded_argv(tmp_path)
    local_dir = Path(argv[argv.index("--local_dir") + 1])
    recorded = json.loads((tmp_path / "record" / "shadow_config.json").read_text())
    assert local_dir != actor
    assert recorded == {"hidden_size": 8}
    original = json.loads((actor / "huggingface" / "config.json").read_text())
    assert original["attn_implementation"] == "flash_attention_2"
    assert not local_dir.exists()


def test_lora_export_writes_back_exclude_modules(
    tmp_path, actor, fake_merger, monkeypatch
):
    monkeypatch.setenv("FAKE_MERGE", "lora")

    out = export_verl_checkpoint(
        actor,
        tmp_path / "out",
        base_model="org/base",
        verl_python=str(fake_merger),
        exclude_modules="(.*vision_tower.*)",
    )

    config = json.loads((out / "adapter_config.json").read_text())
    assert config["exclude_modules"] == "(.*vision_tower.*)"


def test_unreadable_merger_or_checkpoint_files_raise_export_errors(
    tmp_path, actor, fake_merger, monkeypatch
):
    (actor / "huggingface" / "config.json").write_text("{not json")
    with pytest.raises(VerlExportError, match="can't read"):
        export_verl_checkpoint(
            actor, tmp_path / "out", base_model="org/base", verl_python=str(fake_merger)
        )
