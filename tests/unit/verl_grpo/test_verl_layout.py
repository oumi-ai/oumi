import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config

from oumi.verl_grpo.layout import fill_from_base, to_text_only, weight_map

_LANGUAGE = "model.language_model.layers.0.mlp.weight"
_VISION = "model.visual.blocks.0.attn.weight"


def _write_model(model_dir: Path, config: dict, names: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    save_file(
        {n: torch.full((2,), float(i)) for i, n in enumerate(names)},
        str(model_dir / "model.safetensors"),
    )


def test_fill_gives_a_text_only_model_the_base_layout(tmp_path):
    base_config = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    _write_model(tmp_path / "base", base_config, [_LANGUAGE, _VISION])
    (tmp_path / "base" / "preprocessor_config.json").write_text("{}")
    parent = tmp_path / "parent"
    _write_model(parent, {"architectures": ["Qwen3_5ForCausalLM"]}, [_LANGUAGE])

    assert fill_from_base(parent, tmp_path / "base")

    assert json.loads((parent / "config.json").read_text()) == base_config
    assert (parent / "preprocessor_config.json").is_file()
    shards = weight_map(parent)
    assert shards[_LANGUAGE] == "model.safetensors"
    assert shards[_VISION] == "model-base-layout.safetensors"


def test_fill_adds_missing_tensors_but_keeps_a_matching_layout(tmp_path):
    # Gemma 4's HF save drops the KV-shared layers' k_proj, which vLLM still loads.
    kv = "model.language_model.layers.20.self_attn.k_proj.weight"
    base_config = {"architectures": ["Gemma4ForConditionalGeneration"]}
    _write_model(tmp_path / "base", base_config, [_LANGUAGE, kv])
    parent = tmp_path / "parent"
    parent_config = {**base_config, "torch_dtype": "bfloat16"}
    _write_model(parent, parent_config, [_LANGUAGE])

    assert fill_from_base(parent, tmp_path / "base")

    assert json.loads((parent / "config.json").read_text()) == parent_config
    assert weight_map(parent)[kv] == "model-base-layout.safetensors"


def test_fill_leaves_a_model_in_the_base_layout_alone(tmp_path):
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    _write_model(tmp_path / "base", config, [_LANGUAGE, _VISION])
    parent = tmp_path / "parent"
    _write_model(parent, config, [_LANGUAGE, _VISION])

    assert not fill_from_base(parent, tmp_path / "base")
    assert {p.name for p in parent.iterdir()} == {"config.json", "model.safetensors"}


def test_text_only_keeps_the_language_model_under_its_text_config(tmp_path):
    model = tmp_path / "export"
    model.mkdir()
    Qwen3_5Config().save_pretrained(model)
    save_file(
        {_LANGUAGE: torch.ones(2), _VISION: torch.ones(2)},
        str(model / "model.safetensors"),
    )

    assert to_text_only(model)

    config = json.loads((model / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert config["model_type"] == "qwen3_5_text"
    assert set(weight_map(model)) == {_LANGUAGE}


def test_text_only_leaves_a_text_only_model_alone(tmp_path):
    model = tmp_path / "export"
    model.mkdir()
    Qwen3_5Config().get_text_config().save_pretrained(model)
    save_file({_LANGUAGE: torch.ones(2)}, str(model / "model.safetensors"))

    assert not to_text_only(model)
    assert set(weight_map(model)) == {_LANGUAGE}
