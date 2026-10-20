"""Unit tests for ``oumi.core.trainers.hf_trainer`` helpers."""

import json
import pathlib

from oumi.core.trainers.hf_trainer import _patch_name_or_path_in_config


def test_patches_null_name_or_path(tmp_path: pathlib.Path) -> None:
    """Null ``_name_or_path`` is replaced with ``model_name``."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"_name_or_path": None, "architectures": ["Qwen3ForCausalLM"]})
    )

    _patch_name_or_path_in_config(str(tmp_path), "Qwen/Qwen3-0.6B")

    result = json.loads(config_path.read_text())
    assert result["_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert result["architectures"] == ["Qwen3ForCausalLM"]


def test_patches_missing_name_or_path(tmp_path: pathlib.Path) -> None:
    """``_name_or_path`` is added when the key is absent entirely."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))

    _patch_name_or_path_in_config(str(tmp_path), "Qwen/Qwen3-0.6B")

    result = json.loads(config_path.read_text())
    assert result["_name_or_path"] == "Qwen/Qwen3-0.6B"


def test_preserves_existing_name_or_path(tmp_path: pathlib.Path) -> None:
    """A non-empty ``_name_or_path`` is left alone."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"_name_or_path": "user/preserved"}))

    _patch_name_or_path_in_config(str(tmp_path), "Qwen/Qwen3-0.6B")

    result = json.loads(config_path.read_text())
    assert result["_name_or_path"] == "user/preserved"


def test_noop_when_config_missing(tmp_path: pathlib.Path) -> None:
    """Adapter-only save dirs (no ``config.json``) are silently skipped."""
    _patch_name_or_path_in_config(str(tmp_path), "Qwen/Qwen3-0.6B")
    assert not (tmp_path / "config.json").exists()


def test_noop_on_malformed_config(tmp_path: pathlib.Path) -> None:
    """A corrupt ``config.json`` is left untouched."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json")

    _patch_name_or_path_in_config(str(tmp_path), "Qwen/Qwen3-0.6B")

    assert config_path.read_text() == "{not valid json"
