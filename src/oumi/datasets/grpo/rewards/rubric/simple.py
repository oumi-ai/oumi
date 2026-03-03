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

"""Simple rubric-based reward function.

Evaluates model completions against string rubrics using an LLM judge.
Returns the fraction of rubrics satisfied (0.0 to 1.0).
"""

from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.datasets.grpo.rewards.rubric.core import (
    RubricJudge,
    RubricStats,
    format_string_rubrics,
    validate_inputs,
)
from oumi.utils.logging import logger


# Module-level state
_judge: RubricJudge | None = None
_stats = RubricStats()


def get_stats() -> RubricStats:
    """Get the current statistics for rubric_reward."""
    return _stats


def reset_stats() -> None:
    """Reset the statistics."""
    global _stats
    _stats = RubricStats()


def _get_judge(model: str) -> RubricJudge:
    """Get or create the judge instance."""
    global _judge
    if _judge is None or _judge.model != model:
        _judge = RubricJudge(
            model=model,
            system_instruction=(
                "You are an expert evaluator. Assess responses against rubrics "
                "fairly and consistently. For each rubric, determine if the "
                "response satisfies it (1) or not (0). Return valid JSON only."
            ),
            prompt_template=(
                "Evaluate the response against the rubrics.\n\n"
                "## Task\n{prompt}\n\n"
                "## Response to Evaluate\n{response}\n\n"
                "## Rubrics\n{rubrics}\n\n"
                "For each rubric, determine if it is satisfied (1) or not (0).\n"
                "Return a JSON object with:\n"
                '- "scores": dict mapping rubric number to 0 or 1\n'
                '- "total_score": fraction of rubrics satisfied (0.0 to 1.0)\n\n'
                'Example: {"scores": {"1": 1, "2": 0, "3": 1}, '
                '"total_score": 0.67}\n\n'
                "Output only valid JSON."
            ),
        )
    return _judge


@register("rubric_reward", RegistryType.REWARD_FUNCTION)
def rubric_reward(
    completions: list[list[dict[str, Any]]],
    prompts: list[str] | None = None,
    rubrics: list[list[str]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Evaluate completions against string rubrics using an LLM judge.

    This is the simplest rubric-based reward function. It evaluates each
    completion against a list of string rubrics and returns the fraction
    of rubrics that are satisfied.

    Args:
        completions: List of model completions to evaluate.
        prompts: List of original prompts/tasks.
        rubrics: List of rubric lists (strings) for each completion.
        **kwargs: Additional arguments:
            - judge_model: Model to use for judging (default: "gpt-4o-mini")
            - system_prompt: Optional system prompts to prepend

    Returns:
        List of float rewards in [0.0, 1.0], representing the fraction
        of rubrics satisfied for each completion.

    Example:
        >>> rewards = rubric_reward(
        ...     completions=[[{"content": "The answer is 42."}]],
        ...     prompts=["What is the answer?"],
        ...     rubrics=[["Provides a numerical answer", "Is concise"]],
        ... )
        >>> print(rewards)  # e.g., [1.0] if both rubrics satisfied
    """
    global _stats

    # Handle alternative parameter names
    prompt_list = prompts or kwargs.get("prompt", [])
    rubric_list = rubrics or kwargs.get("rubrics", [])
    system_prompts: list[Any] = kwargs.get("system_prompt", [])

    # Validate inputs
    completion_strs, prompt_list, rubric_list, count = validate_inputs(
        completions, prompt_list, rubric_list, "rubric_reward"
    )

    if count == 0:
        return [0.0] * len(completions)

    # Get judge
    judge_model = str(kwargs.get("judge_model", "gpt-4o-mini"))
    judge = _get_judge(judge_model)

    # Evaluate each completion
    rewards = []
    for i, (comp, prompt, rubric_items) in enumerate(
        zip(completion_strs, prompt_list, rubric_list)
    ):
        # Ensure rubrics is a list
        if not isinstance(rubric_items, list):
            rubric_items = [rubric_items]

        # Build full prompt with system prompt if provided
        full_prompt = prompt
        if system_prompts and i < len(system_prompts) and system_prompts[i]:
            full_prompt = f"[System: {system_prompts[i]}]\n\n{prompt}"

        # Format rubrics
        rubrics_text = format_string_rubrics(rubric_items)

        # Evaluate
        result = judge.evaluate(full_prompt, comp, rubrics_text)

        if result.success:
            _stats.record_success(result.score, result.time_ms)
        else:
            _stats.record_failure(result.time_ms)

        rewards.append(result.score)

    # Log batch summary
    if _stats.should_log():
        logger.info(_stats.get_summary())

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info(f"[rubric_reward] Batch: size={count}, avg_reward={avg_reward:.3f}")

    return rewards
