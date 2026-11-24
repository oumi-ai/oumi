import json

import torch
from safetensors.torch import load_file, save_file

from oumi.verl_grpo.patches.gemma4_fill import fill_gemma4_kv_tensors

KV = "model.language_model.layers.20.self_attn.k_proj.weight"


def _model(path, model_type, tensors):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    save_file(tensors, str(path / "model.safetensors"))
    return path


def test_gemma4_gets_the_base_s_missing_kv_tensors(tmp_path):
    kv = torch.arange(4, dtype=torch.bfloat16)
    base = _model(
        tmp_path / "base", "gemma4", {"a": torch.zeros(2), KV: kv, "mtp.x": kv.clone()}
    )
    merged = _model(tmp_path / "merged", "gemma4", {"a": torch.ones(2)})

    fill_gemma4_kv_tensors(merged, str(base))

    filled = load_file(merged / "model-base-fill.safetensors")
    assert filled.keys() == {KV}
    assert torch.equal(filled[KV], kv)
    index = json.loads((merged / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        "a": "model.safetensors",
        KV: "model-base-fill.safetensors",
    }


def test_other_models_are_left_alone(tmp_path):
    base = _model(tmp_path / "base", "qwen3_5", {KV: torch.zeros(2)})
    merged = _model(tmp_path / "merged", "qwen3_5", {"a": torch.ones(2)})

    fill_gemma4_kv_tensors(merged, str(base))

    assert {p.name for p in merged.iterdir()} == {"config.json", "model.safetensors"}
