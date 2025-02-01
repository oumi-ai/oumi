# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Union

from oumi.utils.io_utils import load_json

_ADAPTER_CONFIG_RANK_KEY = "r"


def get_lora_rank(adapter_dir: Union[str, Path]) -> int:
    """Gets the LoRA rank for a saved adapter model.

    Example config:
    https://github.com/huggingface/peft/blob/b5db9c935021a54fb5d1a479457ce63ad94e2fe5/docs/source/developer_guides/checkpoint.md?plain=1#L182

    Args:
        adapter_dir: The directory containing the adapter model.

    Returns:
        int: The LoRA rank.

    Raises:
        ValueError: If the LoRA rank is not found in the adapter config or isn't an int.
    """
    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    adapter_config = load_json(adapter_config_path)
    if _ADAPTER_CONFIG_RANK_KEY not in adapter_config:
        raise ValueError(
            f"LoRA rank not found in adapter config: {adapter_config_path}"
        )
    if not isinstance(adapter_config[_ADAPTER_CONFIG_RANK_KEY], int):
        raise ValueError(
            f"LoRA rank in adapter config not an int: {adapter_config_path}"
        )
    return adapter_config[_ADAPTER_CONFIG_RANK_KEY]
