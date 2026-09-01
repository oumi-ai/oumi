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

r"""LLM-judge reward for the letter-counting GRPO task.

Drop-in replacement for `count_letters_verl` that delegates scoring to a
`SimpleJudge` (see `configs/examples/letter_counting/grpo/letter_count_judge_v2.yaml`)
instead of the rule-based `\boxed{N}` parser. The judge is instructed to
reproduce the same reward scale: 0.0 for an exact count, decaying towards
-2.0 with distance, and -3.0 for unparseable responses.
"""

import copy
import functools
import os
import threading
from typing import Any

from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.registry import RegistryType, register

# Imported at module level on purpose: verl's RewardLoopWorker imports this
# module once in the actor's main thread, then calls the reward from many
# executor threads concurrently. Importing SimpleJudge lazily inside those
# threads deadlocks on the import machinery inside ray actors.
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger

_JUDGE_CONFIG_PATH_ENV_VAR = "OUMI_LETTER_COUNT_JUDGE_CONFIG"
_DEFAULT_JUDGE_CONFIG_PATH = (
    "configs/examples/letter_counting/grpo/letter_count_judge_v2.yaml"
)

# The reward scale the judge is instructed to use.
_MIN_SCORE = -3.0
_MAX_SCORE = 0.0

_judge_config: JudgeConfig | None = None
_judge_config_lock = threading.Lock()


def _get_judge_config() -> JudgeConfig:
    """Parse the judge YAML once per process."""
    global _judge_config
    if _judge_config is None:
        with _judge_config_lock:
            if _judge_config is None:
                config_path = os.environ.get(
                    _JUDGE_CONFIG_PATH_ENV_VAR, _DEFAULT_JUDGE_CONFIG_PATH
                )
                _judge_config = JudgeConfig.from_path(config_path)
    return _judge_config


@functools.lru_cache(maxsize=65536)
def _judge_score(response: str, target: str) -> float:
    """Score one completion with the judge, memoized on (response, target).

    A fresh SimpleJudge is built per call (construction is milliseconds): the
    remote engine keeps asyncio primitives (locks, waiter futures) on the
    instance, but each judge() call runs in a new event loop, so a shared
    instance leaks concurrency permits across dead loops until every call
    blocks forever in acquire().
    """
    judge = SimpleJudge(copy.deepcopy(_get_judge_config()))
    output = judge.judge([{"reference": target, "response": response}])[0]

    # FLOAT judgments carry no field_scores; read the typed value.
    score: Any = (output.field_scores or {}).get("judgment")
    if score is None:
        value = (output.field_values or {}).get("judgment")
        if isinstance(value, (int, float, bool)):
            score = float(value)

    if score is None:
        logger.warning(
            "Letter-count judge returned no parseable judgment "
            f"(raw output: {output.raw_output!r}); assigning {_MIN_SCORE}."
        )
        return _MIN_SCORE

    # Never let a hallucinated score leave the reward scale.
    # return min(max(float(score), _MIN_SCORE), _MAX_SCORE)
    return float(score)


@register("judge_count_letters_verl", RegistryType.REWARD_FUNCTION)
def judge_count_letters_verl(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """verl-style reward function that scores completions with an LLM judge.

    Mirrors `count_letters_verl`'s interface and reward scale, but the score
    comes from a `SimpleJudge` configured by the YAML at
    `$OUMI_LETTER_COUNT_JUDGE_CONFIG` (default:
    `configs/examples/letter_counting/grpo/letter_count_judge_v2.yaml`,
    resolved against the working directory).

    Args:
        data_source: The dataset name for the sample (unused).
        solution_str: The model's decoded response.
        ground_truth: The target letter count, as a string.
        extra_info: Extra information about the sample (unused).

    Returns:
        The judge's score in [-3.0, 0.0]. Judge failures (API errors,
        unparseable judge output) fall back to -3.0.
    """
    try:
        return _judge_score(solution_str, str(ground_truth))
    except Exception:
        logger.exception(
            "Letter-count judge call failed; assigning the minimum reward "
            f"({_MIN_SCORE}) to this sample."
        )
        return _MIN_SCORE
