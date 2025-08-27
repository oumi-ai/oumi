import os
import tempfile
from pathlib import Path
from unittest import mock

import datasets
import pytest

from oumi.utils.hf_utils import (
    find_hf_token,
    get_hf_chat_template,
    is_cached_to_disk_hf_dataset,
)


def test_is_saved_to_disk_hf_dataset():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        ds = datasets.Dataset.from_dict(
            {"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]}
        )
        ds_dir = Path(output_temp_dir) / "toy_dataset"
        assert not is_cached_to_disk_hf_dataset(ds_dir)

        ds_dir.mkdir(parents=True, exist_ok=True)
        assert not is_cached_to_disk_hf_dataset(ds_dir)

        ds.save_to_disk(ds_dir, num_shards=2)
        assert is_cached_to_disk_hf_dataset(ds_dir)

        for filename in ("dataset_info.json", "state.json"):
            sub_path: Path = Path(ds_dir) / filename
            assert sub_path.exists() and sub_path.is_file()
            sub_path.unlink()
            assert not is_cached_to_disk_hf_dataset(ds_dir)


@mock.patch.dict(os.environ, {}, clear=True)
def test_find_hf_token_none():
    assert find_hf_token() is None


@mock.patch.dict(os.environ, {"HF_TOKEN": "my-test-token"}, clear=True)
def test_find_hf_token_from_hf_token_env_var():
    assert find_hf_token() == "my-test-token"


@mock.patch.dict(
    os.environ, {"HF_TOKEN_PATH": "/tmp/hf/cache/non-existent-token"}, clear=True
)
def test_find_hf_token_from_hf_token_path_that_doesnt_exist():
    with pytest.raises(
        FileNotFoundError,
        match=("Missing HF token file: '/tmp/hf/cache/non-existent-token'"),
    ):
        find_hf_token()


@mock.patch.dict(os.environ, {}, clear=True)
def test_find_hf_token_from_hf_token_path():
    with tempfile.NamedTemporaryFile(mode="w") as f:
        os.environ["HF_TOKEN_PATH"] = f.name
        f.writelines(["my-test-token-123"])
        f.flush()
        assert find_hf_token() == "my-test-token-123"


@mock.patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True)
def test_find_hf_token_from_hf_token_path_with_empty_hf_token():
    with tempfile.NamedTemporaryFile(mode="w") as f:
        os.environ["HF_TOKEN_PATH"] = f.name
        f.writelines(["my-test-token-123"])
        f.flush()
        assert find_hf_token() == "my-test-token-123"


@mock.patch.dict(os.environ, {}, clear=True)
def test_find_hf_token_from_hf_home_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["HF_HOME"] = tmp_dir
        token_file = Path(tmp_dir) / "token"
        token_file.write_text("my-test-token-999")
        assert find_hf_token() == "my-test-token-999"


def test_get_hf_chat_template_empty():
    assert get_hf_chat_template("", trust_remote_code=False) is None
    assert get_hf_chat_template("", trust_remote_code=True) is None


@pytest.mark.parametrize(
    "tokenizer_name,trust_remote_code",
    [
        ("HuggingFaceTB/SmolLM2-135M-Instruct", False),
        ("Qwen/Qwen2-VL-2B-Instruct", False),
    ],
)
def test_get_hf_chat_template(tokenizer_name: str, trust_remote_code: bool):
    chat_template = get_hf_chat_template(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    assert chat_template is not None
    assert len(chat_template) > 0
