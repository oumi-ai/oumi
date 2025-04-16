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

from enum import Enum

from oumi.utils.logging import logger


class AliasType(str, Enum):
    """The type of configs we support with aliases."""

    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"
    JOB = "job"


_ALIASES: dict[str, dict[AliasType, str]] = {
    "llama4-scout": {
        AliasType.TRAIN: "oumi://configs/recipes/llama4/sft/scout_base_full/train.yaml",
        AliasType.JOB: "oumi://configs/recipes/llama4/sft/scout_base_full/train.yaml",
    },
    "llama4-scout-instruct-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama4/sft/scout_instruct_lora/train.yaml",
    },
    "llama4-scout-instruct-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml",
    },
    "llama4-scout-instruct": {
        AliasType.TRAIN: "oumi://configs/recipes/llama4/sft/scout_instruct_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/llama4/inference/scout_instruct_infer.yaml",
        AliasType.JOB: "oumi://configs/recipes/llama4/sft/scout_instruct_full/gcp_job.yaml",
        AliasType.EVAL: "oumi://configs/recipes/llama4/evaluation/scout_instruct_eval.yaml",
    },
    "llama4-maverick": {
        AliasType.INFER: "oumi://configs/recipes/llama4/inference/maverick_instruct_together_infer.yaml",
    },
}


def try_get_config_name_for_alias(
    alias: str,
    alias_type: AliasType,
) -> str:
    """Gets the config path for a given alias.

    This function resolves the config path for a given alias and alias type.
    If the alias is not found, the original alias is returned.

    Args:
        alias (str): The alias to resolve.
        alias_type (AliasType): The type of config to resolve.

    Returns:
        str: The resolved config path (or the original alias if not found).
    """
    if alias in _ALIASES and alias_type in _ALIASES[alias]:
        config_path = _ALIASES[alias][alias_type]
        logger.info(f"Resolved alias '{alias}' to '{config_path}'")
        return config_path
    return alias
