from pathlib import Path

import pytest

from oumi.utils.peft_utils import get_lora_rank


def test_get_lora_rank_valid(mocker):
    # Mock the load_json function to return a valid config
    mocker.patch("oumi.utils.io_utils.load_json", return_value={"r": 4})
    adapter_dir = Path("/fake/dir")
    assert get_lora_rank(adapter_dir) == 4


def test_get_lora_rank_missing_key(mocker):
    # Mock the load_json function to return a config without the "r" key
    mocker.patch("oumi.utils.io_utils.load_json", return_value={})
    adapter_dir = Path("/fake/dir")
    with pytest.raises(ValueError, match="LoRA rank not found in adapter config"):
        get_lora_rank(adapter_dir)


def test_get_lora_rank_invalid_type(mocker):
    # Mock the load_json function to return a config with a non-int "r" key
    mocker.patch("oumi.utils.io_utils.load_json", return_value={"r": "not_an_int"})
    adapter_dir = Path("/fake/dir")
    with pytest.raises(ValueError, match="LoRA rank in adapter config not an int"):
        get_lora_rank(adapter_dir)
