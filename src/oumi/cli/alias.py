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
    QUANTIZE = "quantize"
    JUDGE = "judge"
    TUNE = "tune"
    ANALYZE = "analyze"


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
    # Qwen3-Next family.
    "qwen3-next-80b-a3b": {
        AliasType.INFER: "oumi://configs/recipes/qwen3_next/inference/80b_a3b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/qwen3_next/evaluation/80b_a3b_eval.yaml",
    },
    "qwen3-next-80b-a3b-instruct": {
        AliasType.INFER: "oumi://configs/recipes/qwen3_next/inference/80b_a3b_instruct_infer.yaml",
    },
    "qwen3-next-80b-a3b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml",
    },
    # Gemma 3 family.
    "gemma3-4b": {
        AliasType.TRAIN: "oumi://configs/recipes/gemma3/sft/4b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/gemma3/inference/4b_instruct_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/gemma3/evaluation/4b/eval.yaml",
    },
    "gemma3-12b": {
        AliasType.INFER: "oumi://configs/recipes/gemma3/inference/12b_instruct_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/gemma3/evaluation/12b/eval.yaml",
    },
    "gemma3-12b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/gemma3/sft/12b_lora/train.yaml",
    },
    "gemma3-27b": {
        AliasType.INFER: "oumi://configs/recipes/gemma3/inference/27b_instruct_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/gemma3/evaluation/27b/eval.yaml",
    },
    "gemma3-27b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/gemma3/sft/27b_lora/train.yaml",
    },
    # OLMo 3 family.
    "olmo3-7b": {
        AliasType.TRAIN: "oumi://configs/recipes/olmo3/sft/7b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/olmo3/inference/7b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/olmo3/evaluation/7b/eval.yaml",
    },
    "olmo3-32b": {
        AliasType.INFER: "oumi://configs/recipes/olmo3/inference/32b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/olmo3/evaluation/32b/eval.yaml",
    },
    "olmo3-32b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/olmo3/sft/32b_lora/train.yaml",
    },
    # Qwen3-VL family.
    "qwen3-vl-2b": {
        AliasType.INFER: "oumi://configs/recipes/vision/qwen3_vl_2b/inference/infer.yaml",
    },
    "qwen3-vl-4b": {
        AliasType.INFER: "oumi://configs/recipes/vision/qwen3_vl_4b/inference/infer.yaml",
    },
    "qwen3-vl-8b": {
        AliasType.INFER: "oumi://configs/recipes/vision/qwen3_vl_8b/inference/infer.yaml",
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
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_0_5b/full_lambda_job.yaml",
    },
    "falcon_h1_1_5b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/1_5b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_1_5b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_lambda_job.yaml",
    },
    "falcon_h1_1_5b_deep": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/1_5b_deep_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_1_5b_deep/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_lambda_job.yaml",
    },
    "falcon_h1_3b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/3b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_3b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_3b/full_lambda_job.yaml",
    },
    "falcon_h1_7b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/7b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_7b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_7b/full_lambda_job.yaml",
    },
    "falcon_h1_34b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/falcon_h1/inference/34b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_h1/evaluation/falcon_h1_34b/eval.yaml",
        AliasType.JOB: "oumi://configs/recipes/falcon_h1/sft/falcon_h1_34b/full_lambda_job.yaml",
    },
    # Falcon E family.
    "falcon-e-1b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_e/sft/falcon_e_1b/full_train.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_e/evaluation/falcon_e_1b/eval.yaml",
    },
    "falcon-e-1b-instruct": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_e/evaluation/falcon_e_1b_instruct/eval.yaml",
    },
    "falcon-e-3b": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_e/sft/falcon_e_3b/full_train.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_e/evaluation/falcon_e_3b/eval.yaml",
    },
    "falcon-e-3b-instruct": {
        AliasType.TRAIN: "oumi://configs/recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml",
        AliasType.EVAL: "oumi://configs/recipes/falcon_e/evaluation/falcon_e_3b_instruct/eval.yaml",
    },
    # DeepSeek R1 family.
    "deepseek-r1-distill-llama-8b": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/deepseek_r1/inference/distill_llama_8b/infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml",
    },
    "deepseek-r1-distill-llama-8b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml",
    },
    "deepseek-r1-distill-llama-8b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml",
    },
    "deepseek-r1-distill-llama-70b": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/deepseek_r1/inference/distill_llama_70b/infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml",
    },
    "deepseek-r1-distill-llama-70b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml",
    },
    "deepseek-r1-distill-llama-70b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml",
    },
    "deepseek-r1-distill-qwen-1.5b": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/deepseek_r1/inference/distill_qwen_1_5b/infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml",
    },
    "deepseek-r1-distill-qwen-1.5b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml",
    },
    "deepseek-r1-distill-qwen-32b": {
        AliasType.INFER: "oumi://configs/recipes/deepseek_r1/inference/distill_qwen_32b/infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml",
    },
    "deepseek-r1-distill-qwen-32b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml",
    },
    "deepseek-r1-671b": {
        AliasType.INFER: "oumi://configs/recipes/deepseek_r1/inference/671b_together/infer.yaml",
    },
    # Llama 3.1 family.
    "llama3.1-8b": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/8b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/llama3_1/inference/8b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/llama3_1/evaluation/8b_eval.yaml",
    },
    "llama3.1-8b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/8b_lora/train.yaml",
    },
    "llama3.1-8b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/8b_qlora/train.yaml",
    },
    "llama3.1-70b": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/70b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/llama3_1/inference/70b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/llama3_1/evaluation/70b_eval.yaml",
    },
    "llama3.1-70b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/70b_lora/train.yaml",
    },
    "llama3.1-70b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/70b_qlora/train.yaml",
    },
    "llama3.1-405b": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/405b_full/train.yaml",
    },
    "llama3.1-405b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/405b_lora/train.yaml",
    },
    "llama3.1-405b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_1/sft/405b_qlora/train.yaml",
    },
    # Llama 3.2 family.
    "llama3.2-1b": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_2/sft/1b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/llama3_2/inference/1b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/llama3_2/evaluation/1b_eval.yaml",
    },
    "llama3.2-3b": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_2/sft/3b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/llama3_2/inference/3b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/llama3_2/evaluation/3b_eval.yaml",
    },
    "llama3.2-3b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_2/sft/3b_lora/train.yaml",
    },
    "llama3.2-3b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_2/sft/3b_qlora/train.yaml",
    },
    # Llama 3.3 family.
    "llama3.3-70b": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_3/sft/70b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/llama3_3/inference/70b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/llama3_3/evaluation/70b_eval.yaml",
    },
    "llama3.3-70b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_3/sft/70b_lora/train.yaml",
    },
    "llama3.3-70b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/llama3_3/sft/70b_qlora/train.yaml",
    },
    # QwQ family.
    "qwq-32b": {
        AliasType.TRAIN: "oumi://configs/recipes/qwq/sft/full_train.yaml",
        AliasType.INFER: "oumi://configs/recipes/qwq/inference/infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/qwq/evaluation/eval.yaml",
    },
    "qwq-32b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/qwq/sft/lora_train.yaml",
    },
    "qwq-32b-qlora": {
        AliasType.TRAIN: "oumi://configs/recipes/qwq/sft/qlora_train.yaml",
    },
    # SmolLM family.
    "smollm-135m": {
        AliasType.TRAIN: "oumi://configs/recipes/smollm/sft/135m/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/smollm/inference/135m_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/smollm/evaluation/135m/eval.yaml",
        AliasType.TUNE: "oumi://configs/recipes/smollm/tuning/135m/tune.yaml",
    },
    # GPT-2 family.
    "gpt2": {
        AliasType.TRAIN: "oumi://configs/recipes/gpt2/pretraining/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/gpt2/inference/infer.yaml",
    },
    # GPT-OSS family.
    "gpt-oss-20b": {
        AliasType.INFER: "oumi://configs/recipes/gpt_oss/inference/20b_infer.yaml",
    },
    "gpt-oss-20b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml",
    },
    "gpt-oss-120b": {
        AliasType.INFER: "oumi://configs/recipes/gpt_oss/inference/120b_infer.yaml",
    },
    # Vision models - Llama 3.2 Vision.
    "llama3.2-vision-11b": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml",
    },
    "llama3.2-vision-11b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml",
    },
    "llama3.2-vision-90b": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/llama3_2_vision/sft/90b_full/train.yaml",
    },
    # Vision models - LLaVA.
    "llava-7b": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/llava_7b/sft/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/vision/llava_7b/inference/infer.yaml",
    },
    # Vision models - Phi3 Vision.
    "phi3-vision": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/phi3/sft/full/train.yaml",
    },
    "phi3-vision-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/phi3/sft/lora/train.yaml",
    },
    # Vision models - Phi4 Vision.
    "phi4-vision": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/phi4/sft/full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/vision/phi4/inference/infer.yaml",
    },
    "phi4-vision-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/phi4/sft/lora/train.yaml",
    },
    # Vision models - Qwen2-VL.
    "qwen2-vl-2b": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml",
        AliasType.EVAL: "oumi://configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml",
    },
    "qwen2-vl-2b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml",
    },
    # Vision models - Qwen2.5-VL.
    "qwen2.5-vl-3b": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml",
    },
    "qwen2.5-vl-3b-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml",
    },
    "qwen2.5-vl-7b": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml",
    },
    # Vision models - SmolVLM.
    "smolvlm": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/smolvlm/sft/full/train.yaml",
        AliasType.INFER: "oumi://configs/recipes/vision/smolvlm/inference/infer.yaml",
    },
    "smolvlm-lora": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/smolvlm/sft/lora/train.yaml",
    },
    # Vision models - InternVL3.
    "internvl3": {
        AliasType.TRAIN: "oumi://configs/recipes/vision/internvl3/sft/full/train.yaml",
    },
    # Hosted models - Anthropic.
    "claude-3-5-sonnet": {
        AliasType.INFER: "oumi://configs/apis/anthropic/infer_claude_3_5_sonnet.yaml",
        AliasType.EVAL: "oumi://configs/apis/anthropic/eval_claude_3_5_sonnet.yaml",
    },
    "claude-3-7-sonnet": {
        AliasType.INFER: "oumi://configs/apis/anthropic/infer_claude_3_7_sonnet.yaml",
        AliasType.EVAL: "oumi://configs/apis/anthropic/eval_claude_3_7_sonnet.yaml",
    },
    "claude-opus-4-1": {
        AliasType.INFER: "oumi://configs/apis/anthropic/infer_claude_opus_4_1.yaml",
    },
    # Hosted models - Google Gemini.
    "gemini-1-5-pro": {
        AliasType.INFER: "oumi://configs/apis/gemini/infer_gemini_1_5_pro.yaml",
        AliasType.EVAL: "oumi://configs/apis/gemini/eval_gemini_1_5_pro.yaml",
    },
    "gemini-2-5-pro": {
        AliasType.INFER: "oumi://configs/apis/gemini/infer_gemini_2_5_pro.yaml",
    },
    # Hosted models - OpenAI GPT-4 series.
    "gpt-4o": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_4o.yaml",
        AliasType.EVAL: "oumi://configs/apis/openai/eval_gpt_4o.yaml",
    },
    "gpt-4o-mini": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_4o_mini.yaml",
    },
    "chatgpt-4o-latest": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_chatgpt_4o_latest.yaml",
    },
    "gpt-4-1": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_4_1.yaml",
    },
    "gpt-4-1-mini": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_4_1_mini.yaml",
    },
    # Hosted models - OpenAI GPT-5 series.
    "gpt-5": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_5.yaml",
    },
    "gpt-5-mini": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_5_mini.yaml",
    },
    "gpt-5-nano": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_5_nano.yaml",
    },
    "gpt-5-chat-latest": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_5_chat_latest.yaml",
    },
    # Hosted models - OpenAI o-series reasoning models.
    "gpt-o1-preview": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_gpt_o1_preview.yaml",
        AliasType.EVAL: "oumi://configs/apis/openai/eval_gpt_o1_preview.yaml",
    },
    "o1": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_o1.yaml",
    },
    "o1-mini": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_o1_mini.yaml",
    },
    "o3-mini": {
        AliasType.INFER: "oumi://configs/apis/openai/infer_o3_mini.yaml",
    },
    # Hosted models - Vertex AI.
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
    # Judge aliases (for generic judges only)
    "format-compliance": {
        AliasType.JUDGE: "oumi://configs/projects/judges/generic/format_compliance.yaml",
    },
    "instruction-following": {
        AliasType.JUDGE: "oumi://configs/projects/judges/generic/instruction_following.yaml",
    },
    "topic-adherence": {
        AliasType.JUDGE: "oumi://configs/projects/judges/generic/topic_adherence.yaml",
    },
    "truthfulness": {
        AliasType.JUDGE: "oumi://configs/projects/judges/generic/truthfulness.yaml",
    },
    "safety": {
        AliasType.JUDGE: "oumi://configs/projects/judges/generic/safety.yaml",
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
