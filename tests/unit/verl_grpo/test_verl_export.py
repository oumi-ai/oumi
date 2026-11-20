import json
import stat
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

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
target = Path(argv[argv.index("--target_dir") + 1])
mode = os.environ["FAKE_MERGE"]
if mode == "fail":
    print("merger exploded", file=sys.stderr)
    sys.exit(3)
(target / "tokenizer.json").write_text("{}")
if mode == "full":
    (target / "config.json").write_text("{}")
    save_file({"a": torch.zeros(2)}, str(target / "model.safetensors"))
elif mode == "full_sharded":
    (target / "config.json").write_text("{}")
    save_file({"a": torch.zeros(2)}, str(target / "model-00001-of-00001.safetensors"))
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {"a": "model-00001-of-00001.safetensors"},
            }
        )
    )
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
def base_dir(tmp_path) -> Path:
    path = tmp_path / "base"
    path.mkdir()
    save_file(
        {"a": torch.zeros(2, dtype=torch.bfloat16)}, str(path / "model.safetensors")
    )
    return path


@pytest.fixture
def actor(tmp_path) -> Path:
    actor_dir = tmp_path / "ckpt" / "global_step_6" / "actor"
    (actor_dir / "huggingface").mkdir(parents=True)
    (actor_dir / "huggingface" / "config.json").write_text(
        json.dumps({"hidden_size": 8})
    )
    (actor_dir / "fsdp_config.json").write_text("{}")
    return actor_dir


def _recorded_argv(tmp_path: Path) -> list[str]:
    return json.loads((tmp_path / "record" / "argv.json").read_text())


def test_full_model_export_runs_verl_s_merger_and_places_the_model(
    tmp_path, fake_merger, base_dir
):
    actor_dir = tmp_path / "plain_actor"
    actor_dir.mkdir()

    out = export_verl_checkpoint(
        actor_dir,
        tmp_path / "out",
        base_model=str(base_dir),
        verl_python=str(fake_merger),
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
    tmp_path, actor, fake_merger, base_dir, monkeypatch, capsys
):
    args = [
        "--actor-dir",
        str(actor),
        "--output-dir",
        str(tmp_path / "out"),
        "--base-model",
        str(base_dir),
        "--verl-python",
        str(fake_merger),
    ]
    assert main(args) == 0

    monkeypatch.setenv("FAKE_MERGE", "fail")
    assert main(args) == 1
    assert capsys.readouterr().err.startswith("error: ")


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


def test_full_export_fills_tensors_missing_from_the_base(
    tmp_path, actor, fake_merger, base_dir
):
    save_file(
        {
            "a": torch.zeros(2, dtype=torch.bfloat16),
            "b": torch.arange(3, dtype=torch.bfloat16),
        },
        str(base_dir / "model.safetensors"),
    )

    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model=str(base_dir), verl_python=str(fake_merger)
    )

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        "a": "model.safetensors",
        "b": "model-base-fill.safetensors",
    }
    assert index["metadata"]["total_size"] == 8 + 6
    filled = load_file(out / "model-base-fill.safetensors")
    assert filled.keys() == {"b"}
    assert filled["b"].dtype == torch.bfloat16
    assert torch.equal(filled["b"], load_file(base_dir / "model.safetensors")["b"])


def test_full_export_fill_extends_a_sharded_export_index(
    tmp_path, actor, fake_merger, base_dir, monkeypatch
):
    monkeypatch.setenv("FAKE_MERGE", "full_sharded")
    save_file(
        {"a": torch.zeros(2), "c": torch.ones(2)}, str(base_dir / "model.safetensors")
    )

    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model=str(base_dir), verl_python=str(fake_merger)
    )

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        "a": "model-00001-of-00001.safetensors",
        "c": "model-base-fill.safetensors",
    }
    assert index["metadata"]["total_size"] == 8 + 8


def test_full_export_fill_ignores_non_hf_safetensors_in_the_base(
    tmp_path, actor, fake_merger, base_dir
):
    save_file(
        {"a": torch.zeros(2), "b": torch.ones(2)}, str(base_dir / "model.safetensors")
    )
    save_file({"native.x": torch.ones(2)}, str(base_dir / "consolidated.safetensors"))

    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model=str(base_dir), verl_python=str(fake_merger)
    )

    assert load_file(out / "model-base-fill.safetensors").keys() == {"b"}


def test_full_export_fill_reads_only_the_shards_the_base_index_names(
    tmp_path, actor, fake_merger, base_dir
):
    save_file(
        {"a": torch.zeros(2), "b": torch.ones(2)}, str(base_dir / "model-1.safetensors")
    )
    save_file({"stray": torch.ones(2)}, str(base_dir / "model-stray.safetensors"))
    (base_dir / "model.safetensors").unlink()
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "model-1.safetensors", "b": "model-1.safetensors"}}
        )
    )

    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model=str(base_dir), verl_python=str(fake_merger)
    )

    assert load_file(out / "model-base-fill.safetensors").keys() == {"b"}


def test_a_hub_base_is_downloaded_with_hf_weight_patterns_only(
    tmp_path, actor, fake_merger, base_dir, monkeypatch
):
    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(base_dir)

    monkeypatch.setattr("oumi.verl_grpo.export.snapshot_download", fake_download)

    export_verl_checkpoint(
        actor, tmp_path / "out", base_model="org/base", verl_python=str(fake_merger)
    )

    assert calls == [
        (
            "org/base",
            {"allow_patterns": ["model*.safetensors", "model.safetensors.index.json"]},
        )
    ]


def test_a_failed_base_download_raises_and_leaves_output_dir_empty(
    tmp_path, actor, fake_merger, monkeypatch
):
    def failing_download(repo_id, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("oumi.verl_grpo.export.snapshot_download", failing_download)
    out = tmp_path / "out"

    with pytest.raises(
        VerlExportError, match="can't download base weights for 'org/b'"
    ):
        export_verl_checkpoint(
            actor, out, base_model="org/b", verl_python=str(fake_merger)
        )

    assert list(out.iterdir()) == []


def test_full_export_whose_names_mostly_miss_the_base_is_refused(
    tmp_path, actor, fake_merger, base_dir
):
    save_file(
        {name: torch.zeros(2) for name in ("x", "y", "z")},
        str(base_dir / "model.safetensors"),
    )
    out = tmp_path / "out"

    with pytest.raises(VerlExportError, match="3 of the base model's 3 tensors"):
        export_verl_checkpoint(
            actor, out, base_model=str(base_dir), verl_python=str(fake_merger)
        )

    assert list(out.iterdir()) == []


def test_full_export_with_nothing_missing_adds_no_shard(
    tmp_path, actor, fake_merger, base_dir
):
    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model=str(base_dir), verl_python=str(fake_merger)
    )

    assert not (out / "model-base-fill.safetensors").exists()
    assert not (out / "model.safetensors.index.json").exists()


def test_lora_export_never_fills_from_the_base(
    tmp_path, actor, fake_merger, base_dir, monkeypatch
):
    monkeypatch.setenv("FAKE_MERGE", "lora")
    save_file({"zzz": torch.zeros(2)}, str(base_dir / "model.safetensors"))

    out = export_verl_checkpoint(
        actor, tmp_path / "out", base_model=str(base_dir), verl_python=str(fake_merger)
    )

    assert not (out / "model-base-fill.safetensors").exists()


def _write_0_7_1_actor(actor_dir: Path, lora_alpha: int) -> None:
    (actor_dir / "lora_adapter").mkdir(parents=True)
    (actor_dir / "lora_adapter" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "old", "lora_alpha": lora_alpha})
    )
    save_file(
        {"w": torch.zeros(2)}, str(actor_dir / "lora_adapter/adapter_model.safetensors")
    )
    (actor_dir / "huggingface").mkdir()
    for name in ("config.json", "tokenizer.json"):
        (actor_dir / "huggingface" / name).write_text("{}")


def test_0_7_1_lora_layout_is_placed_without_the_merger(tmp_path, fake_merger):
    actor_dir = tmp_path / "legacy_actor"
    _write_0_7_1_actor(actor_dir, lora_alpha=128)

    out = export_verl_checkpoint(
        actor_dir,
        tmp_path / "out",
        base_model="org/base",
        verl_python=str(fake_merger),
        exclude_modules="(.*vision_tower.*)",
    )

    assert not (tmp_path / "record" / "argv.json").exists()
    assert {p.name for p in out.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
    }
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["lora_alpha"] == 128
    assert config["base_model_name_or_path"] == "org/base"
    assert config["exclude_modules"] == "(.*vision_tower.*)"


def test_0_7_1_lora_layout_with_zero_alpha_is_refused(tmp_path, fake_merger):
    actor_dir = tmp_path / "legacy_actor"
    _write_0_7_1_actor(actor_dir, lora_alpha=0)

    with pytest.raises(VerlExportError, match="lora_alpha"):
        export_verl_checkpoint(
            actor_dir,
            tmp_path / "out",
            base_model="org/base",
            verl_python=str(fake_merger),
        )
