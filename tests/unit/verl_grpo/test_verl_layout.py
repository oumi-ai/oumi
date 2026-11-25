import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5Config,
)

from oumi.verl_grpo.layout import prepare_for_verl, to_text_only, weight_map

_LANGUAGE = "model.language_model.layers.0.mlp.weight"
_VISION = "model.visual.blocks.0.attn.weight"


def _write_model(model_dir: Path, config: dict, names: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    save_file(
        {n: torch.full((2,), float(i)) for i, n in enumerate(names)},
        str(model_dir / "model.safetensors"),
    )


def _tiny_llama(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )
    return LlamaForCausalLM(config)


def test_prepare_fills_missing_tensors_that_transformers_then_loads(tmp_path):
    # Gemma 4's HF save drops tensors vLLM still loads, such as the KV-shared k_proj.
    missing = "model.layers.0.mlp.up_proj.weight"
    base = _tiny_llama(seed=0)
    base.save_pretrained(tmp_path / "base")
    parent = _tiny_llama(seed=1)
    parent.save_pretrained(tmp_path / "parent")
    weights = load_file(tmp_path / "parent" / "model.safetensors")
    del weights[missing]
    save_file(weights, str(tmp_path / "parent" / "model.safetensors"))
    parent_files = {p.name for p in (tmp_path / "parent").iterdir()}

    out = prepare_for_verl(tmp_path / "parent", tmp_path / "base", tmp_path / "out")

    assert out == tmp_path / "out"
    assert {p.name for p in (tmp_path / "parent").iterdir()} == parent_files
    # A lone model.safetensors beside the index would make the loader skip the fill.
    assert not (out / "model.safetensors").exists()
    loaded = AutoModelForCausalLM.from_pretrained(out).state_dict()
    assert torch.equal(loaded[missing], base.state_dict()[missing])
    kept = "model.layers.0.mlp.down_proj.weight"
    assert torch.equal(loaded[kept], parent.state_dict()[kept])


def test_prepare_gives_a_text_only_parent_the_base_layout(tmp_path):
    base_config = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    _write_model(tmp_path / "base", base_config, [_LANGUAGE, _VISION])
    (tmp_path / "base" / "preprocessor_config.json").write_text("{}")
    parent_config = {"architectures": ["Qwen3_5ForCausalLM"]}
    _write_model(tmp_path / "parent", parent_config, [_LANGUAGE])

    out = prepare_for_verl(tmp_path / "parent", tmp_path / "base", tmp_path / "out")

    assert json.loads((out / "config.json").read_text()) == base_config
    assert (out / "preprocessor_config.json").is_file()
    assert set(weight_map(out)) == {_LANGUAGE, _VISION}
    parent = json.loads((tmp_path / "parent" / "config.json").read_text())
    assert parent == parent_config


def test_prepare_returns_a_parent_in_the_base_layout_as_is(tmp_path):
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    _write_model(tmp_path / "base", config, [_LANGUAGE, _VISION])
    _write_model(tmp_path / "parent", config, [_LANGUAGE, _VISION])

    out = prepare_for_verl(tmp_path / "parent", tmp_path / "base", tmp_path / "out")

    assert out == tmp_path / "parent"
    assert not (tmp_path / "out").exists()


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
