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

r"""LLM-judge rewards for counting the letters in a string.

Drop-in counterparts to the deterministic rewards in `count_letters_rewards.py`,
which stay available and unchanged.

These rewards differ from the deterministic ones in exactly one step. The
deterministic reward extracts the model's answer with the `_extract_prediction`
regex and then scores it with `compute_letter_count_reward`; here an LLM judge
performs the extraction and the result is handed to that same
`compute_letter_count_reward`. The equivalence check, the graded distance penalty,
and the reward scale are therefore shared code, not a reimplementation: for any
rollout where the judge and the regex read the same answer, both rewards return
exactly the same number. An A/B between them isolates extraction quality.

Where they can differ is where the regex is brittle. It demands exactly one
`\boxed{<digits>}` in the whole response, so it reads nothing (-3.0) from a rollout
that boxes an intermediate value before its final answer, or that writes
`\boxed{ 3 }` with whitespace inside the braces. The judge is asked to read those the
way a person would, while still reporting "no answer" for the cases the regex also
rejects: a count given only in prose, an unclosed box, or `\boxed{three}`.

Cost: every rollout needs a judge inference call, which is far slower and more
expensive than a regex. Prefer the batched `count_letters_judge` entry point, which
judges a whole trl batch in one call; verl's interface is necessarily per-rollout.
"""

import functools
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.datasets.grpo.rewards.count_letters_rewards import compute_letter_count_reward
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger

DEFAULT_JUDGE_CONFIG = "configs/projects/judges/letter_count/count_letters_reward.yaml"

# Key holding the judge's verdict in its structured output.
_JUDGMENT_KEY = "judgment"

# What the judge reports when the rollout has no well-formed boxed answer to read.
_NO_ANSWER = "NONE"

# Shown to the judge when the caller has no question text for a rollout.
_MISSING_QUESTION = "(not provided)"


@functools.lru_cache(maxsize=8)
def _get_judge(judge_config: str) -> SimpleJudge:
    """Returns a judge for `judge_config`, building it at most once per config.

    Building a judge constructs an inference engine, which is far too expensive to
    repeat for every rollout.
    """
    logger.info(f"Building letter-count reward judge from: {judge_config}")
    return SimpleJudge(judge_config)


def _parse_judged_count(judgment: Any) -> int | None:
    """Returns the count the judge read from the rollout, or None if it read none.

    This parses the judge's own structured verdict, not the model's response: the
    judge reports either the digits it read or `NONE`.
    """
    if not isinstance(judgment, str):
        return None
    judgment = judgment.strip()
    if not judgment or judgment.upper() == _NO_ANSWER:
        return None
    try:
        return int(judgment)
    except ValueError:
        logger.warning(
            f"Letter-count reward judge reported {judgment!r}, which is neither an "
            f"integer nor {_NO_ANSWER}; treating the rollout as having no answer."
        )
        return None


def _reward_for_count(count: int | None, target_count: int) -> float:
    """Scores an extracted count using the deterministic reward's own function.

    The count is re-serialized into the canonical ``\\boxed{N}`` form that
    `compute_letter_count_reward` parses (and an empty response when the judge read
    no answer, which that function scores as unparseable). Routing through it rather
    than reimplementing its formula is what guarantees that the judge-based and
    deterministic rewards stay on an identical scale.
    """
    completion = f"\\boxed{{{count}}}" if count is not None else ""
    return compute_letter_count_reward(completion, target_count)


def compute_letter_count_judge_rewards(
    responses: list[str],
    target_counts: list[int],
    questions: list[str] | None = None,
    judge_config: str = DEFAULT_JUDGE_CONFIG,
) -> list[float]:
    """Computes judge-based rewards for a batch of completions.

    Args:
        responses: The completion strings from the LLM.
        target_counts: The target letter count for each completion.
        questions: The question each completion answers, if available. Used only as
            context for the judge.
        judge_config: Path to the judge config used to extract each answer.

    Returns:
        One reward per completion, in the order given, on the same scale as
        `compute_letter_count_reward`. A rollout the judge could not process scores
        as unparseable, exactly as the deterministic reward scores a response it
        cannot extract an answer from.
    """
    if len(responses) != len(target_counts):
        raise ValueError(
            f"Got {len(responses)} responses but {len(target_counts)} target counts."
        )
    if questions is not None and len(questions) != len(responses):
        raise ValueError(
            f"Got {len(questions)} questions but {len(responses)} responses."
        )
    if not responses:
        return []

    judge_inputs = [
        {
            "question": questions[i] if questions else _MISSING_QUESTION,
            "response": responses[i],
        }
        for i in range(len(responses))
    ]

    # A rollout the judge never scores is treated as having no extractable answer,
    # which is the same outcome the deterministic reward gives an unparseable one.
    rewards = [_reward_for_count(None, target) for target in target_counts]
    try:
        # `judge_partial` keeps the rollouts that were judged successfully instead of
        # raising when any single one fails, so a flaky API call costs one rollout's
        # signal rather than the whole training step.
        result = _get_judge(judge_config).judge_partial(judge_inputs)
    except Exception:
        logger.exception(
            f"Letter-count reward judge failed for all {len(responses)} rollouts in "
            "the batch; scoring them as unparseable."
        )
        return rewards

    for index, judge_output in result.successful:
        count = _parse_judged_count(judge_output.field_values.get(_JUDGMENT_KEY))
        rewards[index] = _reward_for_count(count, target_counts[index])

    if result.has_failures:
        logger.warning(
            f"Letter-count reward judge failed on {len(result.failures)} of "
            f"{len(responses)} rollouts; scoring those as unparseable. "
            f"First errors: {dict(list(result.error_messages.items())[:3])}"
        )
    return rewards


@register("count_letters_judge_verl", RegistryType.REWARD_FUNCTION)
def count_letters_judge_verl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> float:
    r"""verl-style judge reward for counting letters in a string.

    The judge-based counterpart of `count_letters_verl`, matching its signature and
    reward scale. verl calls reward functions one rollout at a time, so this issues
    one judge call per rollout; see `count_letters_judge` for the batched interface.

    Args:
        data_source: The dataset name for the sample (unused).
        solution_str: The model's decoded response.
        ground_truth: The target letter count, as a string.
        extra_info: Extra information about the sample. A `question` entry, if
            present, is passed to the judge as context.

    Returns:
        The reward value: 0.0 for an exact count, decaying towards -2.0 as the
        count the judge read gets further from the target, and -3.0 when the judge
        finds no boxed answer to read (or could not score the rollout).
    """
    question = (extra_info or {}).get("question")
    rewards = compute_letter_count_judge_rewards(
        responses=[solution_str],
        target_counts=[int(ground_truth)],
        questions=[question] if isinstance(question, str) else None,
    )
    return rewards[0]


@register("count_letters_judge", RegistryType.REWARD_FUNCTION)
def _count_letters_judge(
    completions: list[list[dict[str, Any]]],
    letter_count: list[int],
    **kwargs: Any,
) -> list[float]:
    """Custom judge reward function for counting letters in a string.

    The judge-based counterpart of `count_letters`, following trl's batched
    GRPOTrainer reward interface, which lets the whole batch be judged in one call.
    For details on custom reward functions in trl, see:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function.

    Args:
        completions: The list of completions from the LLM.
        letter_count: The list of target counts of letters.
        kwargs: Other dataset columns forwarded by trl. A `prompt` column, if
            present, supplies the question passed to the judge as context.

    Returns:
        The reward values for each completion, on the same scale as the
        deterministic `count_letters` reward.
    """
    responses = [completion[0]["content"] for completion in completions]

    # trl forwards the dataset's other columns; `prompt` holds the chat messages the
    # completion answers, whose last user turn is the question.
    questions: list[str] | None = None
    prompts = kwargs.get("prompt")
    if isinstance(prompts, list) and len(prompts) == len(responses):
        questions = [_extract_question(prompt) for prompt in prompts]

    return compute_letter_count_judge_rewards(
        responses=responses,
        target_counts=letter_count,
        questions=questions,
    )


def _extract_question(prompt: Any) -> str:
    """Returns the last user message in a chat-formatted prompt, if there is one."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return _MISSING_QUESTION
