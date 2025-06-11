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
    # Llama 4 family.
    "llama4-scout": {
        AliasType.TRAIN: "oumi://configs/recipes/llama4/sft/scout_base_full/train.yaml",
        AliasType.JOB: "oumi://configs/recipes/llama4/sft/scout_base_full/gcp_job.yaml",
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
    # Qwen3 family.
    "qwen3-30b-a3b": {
        AliasType.INFER: "oumi://configs/recipes/qwen3/inference/30b_a3b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml",
    },
    "qwen3-30b-a3b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml",
        AliasType.JOB: "oumi://configs/recipes/qwen3/sft/30b_a3b_lora/gcp_job.yaml",
    },
    "qwen3-32b": {
        AliasType.INFER: "oumi://configs/recipes/qwen3/inference/32b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/qwen3/evaluation/32b_eval.yaml",
    },
    "qwen3-32b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/qwen3/sft/32b_lora/train.yaml",
        AliasType.JOB: "oumi://configs/recipes/qwen3/sft/32b_lora/gcp_job.yaml",
    },
    # Phi family.
    "phi4-reasoning-plus": {
        AliasType.TRAIN: "oumi://configs/recipes/phi4/sft/reasoning_plus/full_train.yaml",
        AliasType.JOB: "oumi://configs/recipes/phi4/sft/reasoning_plus/full_gcp_job.yaml",
        AliasType.INFER: "oumi://configs/recipes/phi4/inference/reasoning_plus_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/phi4/evaluation/reasoning_plus_eval.yaml",
    },
    "phi4-reasoning-plus-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml",
        AliasType.JOB: "oumi://configs/recipes/phi4/sft/reasoning_plus/lora_gcp_job.yaml",
    },
    "phi4-reasoning-plus-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/phi4/sft/reasoning_plus/qlora_train.yaml",
        AliasType.JOB: "oumi://configs/recipes/phi4/sft/reasoning_plus/qlora_gcp_job.yaml",
    },
    # Falcon H1 family
    "falcon_h1_0_5b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/0_5b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_0_5b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_0_5b/full_gcp_job.yaml",
    },
    "falcon_h1_1_5b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/1_5b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_1_5b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_gcp_job.yaml",
    },
    "falcon_h1_1_5b_deep": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/1_5b_deep_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_1_5b_deep/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_gcp_job.yaml",
    },
    "falcon_h1_3b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/3b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_3b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_3b/full_gcp_job.yaml",
    },
    "falcon_h1_7b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/7b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_7b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_7b/full_gcp_job.yaml",
    },
    "falcon_h1_34b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/34b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_34b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_34b/full_gcp_job.yaml",
    },
    # Hosted models.
    "claude-3-5-sonnet": {
        AliasType.INFER: "oumi://configs/apis/anthropic/infer_claude_3_5_sonnet.yaml",
        AliasType.EVAL: "oumi://configs/apis/anthropic/eval_claude_3_5_sonnet.yaml",
    },
    "claude-3-7-sonnet": {
        AliasType.INFER: "oumi://configs/apis/anthropic/infer_claude_3_7_sonnet.yaml",
        AliasType.EVAL: "oumi://configs/apis/anthropic/eval_claude_3_7_sonnet.yaml",
    },
    "gemini-1-5-pro": {
        AliasType.INFER: "oumi://configs/apis/gemini/infer_gemini_1_5_pro.yaml",
        AliasType.EVAL: "oumi://configs/apis/gemini/eval_gemini_1_5_pro.yaml",
    },
    "gpt-4o": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_4o.yaml",
        AliasType.EVAL: "oumi://configs/apis/openai/eval_gpt_4o.yaml",
    },
    "gpt-o1-preview": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_o1_preview.yaml",
        AliasType.EVAL: "oumi://configs/apis/openai/eval_gpt_o1_preview.yaml",
    },
    "llama-3-3-70b": {
        AliasType.INFER: "oumi://configs/apis/vertex/infer_llama_3_3_70b.yaml",
        AliasType.EVAL: "oumi://configs/apis/vertex/eval_llama_3_3_70b.yaml",
    },
    "llama-3-1-405b": {
        AliasType.INFER: "oumi://configs/apis/vertex/infer_llama_3_1_405b.yaml",
        AliasType.EVAL: "oumi://configs/apis/vertex/eval_llama_3_1_405b.yaml",
    },
    "molmo-7b-o": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/molmo/sft/molmo_o_full/train.yaml",
    },
    "molmo-7b-d": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml",
    },
    "molmo-7b-o-grpo": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/molmo/grpo/train.yaml",
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
