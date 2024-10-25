import pytest

from oumi.utils.io_utils import save_json
from oumi.utils.peft_utils import get_lora_rank


def test_get_lora_rank_fail_no_r(tmp_path):
    data = {"base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"}
    save_json(data, tmp_path / "adapter_config.json")

    with pytest.raises(ValueError, match="LoRA rank not found in adapter config:"):
        get_lora_rank(tmp_path)


def test_get_lora_rank_fail_r_not_int(tmp_path):
    data = {"base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct", "r": "foo"}
    save_json(data, tmp_path / "adapter_config.json")

    with pytest.raises(ValueError, match="LoRA rank in adapter config not an int:"):
        get_lora_rank(tmp_path)


def test_get_lora_rank_successful(root_testdata_dir):
    adapter_config_path = root_testdata_dir / "adapter_config.json"
    adapter_dir = adapter_config_path.parent
    assert get_lora_rank(adapter_dir) == 64
