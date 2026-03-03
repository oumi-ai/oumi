# pyright: reportReturnType=false, reportTypedDictNotRequiredAccess=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Tests for oumi.mcp.config_service — search, metadata, inference, parsing."""

from pathlib import Path
from unittest.mock import patch

import pytest

from oumi.mcp.config_service import (
    build_metadata,
    clear_config_caches,
    determine_peft_type,
    extract_datasets,
    extract_header_comment,
    extract_key_settings,
    find_config_match,
    get_categories,
    infer_task_type,
    load_yaml_strict,
    parse_yaml,
    search_configs,
)
from oumi.mcp.models import ConfigMetadata


def _meta(
    path: str = "recipes/llama/sft/train.yaml",
    description: str = "Fine-tune Llama",
    model_name: str = "meta-llama/Llama-3.1-8B",
    task_type: str = "sft",
    datasets: list[str] | None = None,
    reward_functions: list[str] | None = None,
    peft_type: str = "",
) -> ConfigMetadata:
    return {
        "path": path,
        "description": description,
        "model_name": model_name,
        "task_type": task_type,
        "datasets": datasets or [],
        "reward_functions": reward_functions or [],
        "peft_type": peft_type,
    }


@pytest.mark.parametrize(
    "trainer, expected",
    [
        ("GRPO_TRAINER", "grpo"),
        ("DPO_TRAINER", "dpo"),
        ("KTO", "kto"),
        ("SFT_TRAINER", "sft"),
    ],
)
def test_infer_task_type_from_trainer(trainer: str, expected: str):
    assert infer_task_type(trainer, "some/path.yaml") == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("recipes/llama/grpo/train.yaml", "grpo"),
        ("recipes/llama/dpo/train.yaml", "dpo"),
        ("recipes/llama/sft/train.yaml", "sft"),
        ("recipes/llama/eval/eval.yaml", "evaluation"),
        ("recipes/llama/inference/config.yaml", "inference"),
        ("recipes/llama/pretrain/config.yaml", "pretraining"),
        ("recipes/llama/synth/config.yaml", "synthesis"),
        ("recipes/llama/quantize/config.yaml", "quantization"),
    ],
)
def test_infer_task_type_from_path(path: str, expected: str):
    assert infer_task_type("", path) == expected


def test_infer_task_type_unknown():
    assert infer_task_type("", "recipes/llama/other/config.yaml") == ""


def test_infer_task_type_empty():
    assert infer_task_type("", "") == ""


def test_infer_task_type_trainer_takes_precedence():
    assert infer_task_type("GRPO", "recipes/llama/dpo/train.yaml") == "grpo"


def test_extract_datasets_single():
    cfg = {"data": {"train": {"datasets": [{"dataset_name": "tatsu-lab/alpaca"}]}}}
    assert extract_datasets(cfg) == ["tatsu-lab/alpaca"]


def test_extract_datasets_multiple_splits():
    cfg = {
        "data": {
            "train": {"datasets": [{"dataset_name": "ds_train"}]},
            "validation": {"datasets": [{"dataset_name": "ds_val"}]},
            "test": {"datasets": [{"dataset_name": "ds_test"}]},
        }
    }
    assert extract_datasets(cfg) == ["ds_train", "ds_val", "ds_test"]


def test_extract_datasets_multiple_per_split():
    cfg = {
        "data": {
            "train": {"datasets": [{"dataset_name": "ds1"}, {"dataset_name": "ds2"}]}
        }
    }
    assert extract_datasets(cfg) == ["ds1", "ds2"]


def test_extract_datasets_empty():
    assert extract_datasets({}) == []


def test_extract_datasets_missing_key():
    cfg = {"data": {"train": {"datasets": [{"other": "value"}]}}}
    assert extract_datasets(cfg) == []


def test_extract_datasets_non_dict_entry():
    cfg = {"data": {"train": {"datasets": ["just_a_string"]}}}
    assert extract_datasets(cfg) == []


def test_extract_datasets_non_dict_split():
    cfg = {"data": {"train": "not_a_dict"}}
    assert extract_datasets(cfg) == []


@pytest.mark.parametrize(
    "cfg,path,expected",
    [
        ({"peft": {"lora_r": 16}}, "p.yaml", "lora"),
        ({"peft": {"lora_r": 16, "q_lora": True}}, "p.yaml", "qlora"),
        ({"peft": {"lora_r": 16}}, "recipes/llama/qlora/train.yaml", "qlora"),
    ],
)
def test_peft_detected(cfg, path, expected):
    assert determine_peft_type(cfg, path) == expected


@pytest.mark.parametrize(
    "cfg,path",
    [
        ({}, "p.yaml"),
        ({"peft": {}}, "p.yaml"),
        ({"peft": {"lora_r": 0}}, "p.yaml"),
    ],
)
def test_peft_not_detected(cfg, path):
    assert determine_peft_type(cfg, path) is None


def test_extract_key_settings_training():
    cfg = {
        "training": {
            "learning_rate": 1e-4,
            "num_train_epochs": 3,
            "max_steps": 1000,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        }
    }
    s = extract_key_settings(cfg)
    assert s["learning_rate"] == 1e-4
    assert s["max_steps"] == 1000


def test_extract_key_settings_model_dtype_rename():
    cfg = {"model": {"model_max_length": 2048, "torch_dtype_str": "bfloat16"}}
    s = extract_key_settings(cfg)
    assert s["model_max_length"] == 2048
    assert s["torch_dtype"] == "bfloat16"


def test_extract_key_settings_empty():
    assert extract_key_settings({}) == {}


def test_extract_key_settings_none_skipped():
    cfg = {"training": {"learning_rate": None, "max_steps": 100}}
    s = extract_key_settings(cfg)
    assert "learning_rate" not in s
    assert s["max_steps"] == 100


def test_parse_yaml_valid(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("model:\n  model_name: gpt2\n")
    clear_config_caches()
    assert parse_yaml(str(p)) == {"model": {"model_name": "gpt2"}}


def test_parse_yaml_empty(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    clear_config_caches()
    assert parse_yaml(str(p)) == {}


def test_parse_yaml_invalid(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(":\n  invalid: [unclosed")
    clear_config_caches()
    assert parse_yaml(str(p)) == {}


def test_parse_yaml_returns_deepcopy(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("key: value\n")
    clear_config_caches()
    r1 = parse_yaml(str(p))
    r1["key"] = "mutated"
    assert parse_yaml(str(p))["key"] == "value"


def test_extract_header_comment_simple(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("# Fine-tune Llama 3.1 8B\nmodel:\n")
    assert extract_header_comment(p) == "Fine-tune Llama 3.1 8B"


def test_extract_header_comment_two_line_limit(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("# One\n# Two\n# Three\nmodel:\n")
    assert extract_header_comment(p) == "One Two"


@pytest.mark.parametrize("prefix", ["Usage:", "See Also:", "Requirements:"])
def test_extract_header_comment_skips_prefixes(tmp_path: Path, prefix: str):
    p = tmp_path / "c.yaml"
    p.write_text(f"# Description\n# {prefix} stuff\nmodel:\n")
    assert extract_header_comment(p) == "Description"


def test_extract_header_comment_blank_line_continues(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("# One\n\n# Two\nmodel:\n")
    assert extract_header_comment(p) == "One Two"


def test_extract_header_comment_none(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  model_name: gpt2\n")
    assert extract_header_comment(p) == ""


def test_extract_header_comment_nonexistent():
    assert extract_header_comment(Path("/nonexistent/config.yaml")) == ""


def test_build_metadata_full(tmp_path: Path):
    d = tmp_path / "recipes" / "llama" / "sft"
    d.mkdir(parents=True)
    p = d / "train.yaml"
    p.write_text(
        "# Fine-tune Llama\n"
        "model:\n  model_name: meta-llama/Llama-3.1-8B\n"
        "training:\n  trainer_type: SFT_TRAINER\n"
        "data:\n  train:\n    datasets:\n      - dataset_name: tatsu-lab/alpaca\n"
    )
    clear_config_caches()
    m = build_metadata(p, tmp_path)
    assert m["path"] == "recipes/llama/sft/train.yaml"
    assert m["model_name"] == "meta-llama/Llama-3.1-8B"
    assert m["task_type"] == "sft"
    assert m["datasets"] == ["tatsu-lab/alpaca"]


def test_build_metadata_defaults(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    clear_config_caches()
    m = build_metadata(p, tmp_path)
    assert m["model_name"] == ""
    assert m["task_type"] == ""
    assert m["datasets"] == []
    assert m["peft_type"] == ""


def test_build_metadata_lora(tmp_path: Path):
    p = tmp_path / "lora.yaml"
    p.write_text("model:\n  model_name: gpt2\npeft:\n  lora_r: 16\n")
    clear_config_caches()
    assert build_metadata(p, tmp_path)["peft_type"] == "lora"


def test_build_metadata_reward_functions(tmp_path: Path):
    p = tmp_path / "grpo.yaml"
    p.write_text(
        "model:\n  model_name: gpt2\n"
        "training:\n  trainer_type: GRPO\n"
        "  reward_functions:\n    - accuracy_reward\n    - format_reward\n"
    )
    clear_config_caches()
    m = build_metadata(p, tmp_path)
    assert m["reward_functions"] == ["accuracy_reward", "format_reward"]
    assert m["task_type"] == "grpo"


@pytest.fixture
def sample_configs():
    return [
        _meta(path="recipes/llama/sft/train.yaml"),
        _meta(path="recipes/llama/dpo/train.yaml", task_type="dpo"),
        _meta(path="recipes/mistral/sft/train.yaml", model_name="mistral"),
        _meta(path="recipes/llama/eval/eval.yaml", task_type="evaluation"),
    ]


def test_find_config_exact(sample_configs):
    r = find_config_match("recipes/llama/sft/train.yaml", sample_configs)
    assert r is not None and r["path"] == "recipes/llama/sft/train.yaml"


def test_find_config_partial(sample_configs):
    r = find_config_match("llama/sft", sample_configs)
    assert r is not None and r["path"] == "recipes/llama/sft/train.yaml"


def test_find_config_case_insensitive(sample_configs):
    r = find_config_match("LLAMA/SFT", sample_configs)
    assert r is not None and "llama/sft" in r["path"]


def test_find_config_no_match(sample_configs):
    assert find_config_match("nonexistent", sample_configs) is None


def test_find_config_prefers_train_yaml():
    configs = [
        _meta(path="recipes/llama/sft/eval.yaml"),
        _meta(path="recipes/llama/sft/train.yaml"),
    ]
    r = find_config_match("llama/sft", configs)
    assert r is not None and r["path"] == "recipes/llama/sft/train.yaml"


def test_find_config_empty_list():
    assert find_config_match("anything", []) is None


def test_search_configs_query(sample_configs):
    results = search_configs(sample_configs, query=["llama", "sft"])
    assert all("llama" in r["path"] and "sft" in r["path"] for r in results)


def test_search_configs_no_match(sample_configs):
    assert search_configs(sample_configs, query=["nonexistent"]) == []


def test_search_configs_content_match(tmp_path: Path):
    d1 = tmp_path / "recipes" / "llama" / "sft"
    d1.mkdir(parents=True)
    (d1 / "train.yaml").write_text("model:\n  model_name: special_xyz\n")
    d2 = tmp_path / "recipes" / "mistral" / "sft"
    d2.mkdir(parents=True)
    (d2 / "train.yaml").write_text("model:\n  model_name: gpt2\n")
    configs = [
        _meta(path="recipes/llama/sft/train.yaml"),
        _meta(path="recipes/mistral/sft/train.yaml"),
    ]
    with patch("oumi.mcp.config_service.get_configs_dir", return_value=tmp_path):
        results = search_configs(configs, content_match=["special_xyz"])
    assert len(results) == 1
    assert "llama" in results[0]["path"]


def test_search_configs_content_match_and_logic(tmp_path: Path):
    d1 = tmp_path / "recipes" / "llama"
    d1.mkdir(parents=True)
    (d1 / "train.yaml").write_text("alpha: 1\nbeta: 2\n")
    d2 = tmp_path / "recipes" / "mistral"
    d2.mkdir(parents=True)
    (d2 / "train.yaml").write_text("alpha: 1\ngamma: 3\n")
    configs = [
        _meta(path="recipes/llama/train.yaml"),
        _meta(path="recipes/mistral/train.yaml"),
    ]
    with patch("oumi.mcp.config_service.get_configs_dir", return_value=tmp_path):
        results = search_configs(configs, content_match=["alpha", "beta"])
    assert len(results) == 1
    assert "llama" in results[0]["path"]


def test_get_categories_structure(tmp_path: Path):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "llama").mkdir()
    (tmp_path / "recipes" / "mistral").mkdir()
    (tmp_path / "apis").mkdir()
    (tmp_path / "apis" / "openai").mkdir()
    (tmp_path / "other").mkdir()

    r = get_categories(tmp_path, 10, oumi_version="0.7", configs_source="bundled:0.7")
    assert "recipes" in r["categories"]
    assert "llama" in r["model_families"]
    assert "openai" in r["api_providers"]
    assert r["total_configs"] == 10


def test_get_categories_empty(tmp_path: Path):
    r = get_categories(tmp_path, 0)
    assert r["categories"] == []
    assert r["model_families"] == []


def test_load_yaml_strict_valid(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  model_name: gpt2\n")
    cfg, err = load_yaml_strict(p)
    assert err is None
    assert cfg == {"model": {"model_name": "gpt2"}}


def test_load_yaml_strict_empty(tmp_path: Path):
    p = tmp_path / "e.yaml"
    p.write_text("")
    cfg, err = load_yaml_strict(p)
    assert cfg is None
    assert "empty" in err.lower()


def test_load_yaml_strict_invalid(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(":\n  invalid: [unclosed")
    cfg, err = load_yaml_strict(p)
    assert cfg is None
    assert "Invalid YAML" in err


def test_load_yaml_strict_list_root(tmp_path: Path):
    p = tmp_path / "list.yaml"
    p.write_text("- item1\n- item2\n")
    cfg, err = load_yaml_strict(p)
    assert cfg is None
    assert "mapping" in err.lower()
