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

"""LLM-judge reward for conversational GRPO rollouts (wraps SimpleJudge)."""

import functools
import os

from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.registry import RegistryType, register
from oumi.judges.simple_judge import SimpleJudge


@functools.lru_cache(maxsize=8)
def _get_judge(judge_config_path: str) -> SimpleJudge:
    """Build (once per config path) the judge; reused across reward calls."""
    return SimpleJudge(JudgeConfig.from_path(judge_config_path))


@register("conversation_llm_judge", RegistryType.REWARD_FUNCTION)
def conversation_llm_judge_reward(
    data_source, solution_str, ground_truth, extra_info, judge_config_path=None
) -> float:
    """Score the finished conversation against the goal via an LLM judge.

    Judge config supplies the prompt template (placeholders {conversation}, {goal}),
    judgment type, and its own inference engine. Returns the judgment field's score.
    """
    path = (
        judge_config_path
        or (extra_info or {}).get("judge_config_path")
        or os.environ.get("OUMI_JUDGE_CONFIG_PATH")
    )
    if not path:
        raise ValueError(
            "conversation_llm_judge needs 'judge_config_path' (arg, extra_info, "
            "or the OUMI_JUDGE_CONFIG_PATH env var)."
        )
    judge = _get_judge(path)
    result = judge.judge(
        [{"conversation": solution_str, "goal": ground_truth or ""}]
    )[0]
    scores = result.field_scores or {}
    value = scores.get("judgment")
    return float(value) if value is not None else 0.0
