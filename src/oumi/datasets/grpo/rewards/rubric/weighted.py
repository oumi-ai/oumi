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

"""Weighted rubric-based reward function.

Evaluates model completions against weighted rubrics using an LLM judge.
Supports priority weighting and pitfall criteria (negative weights).
"""

import json
import re
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.datasets.grpo.rewards.rubric.core import (
    RubricJudge,
    RubricStats,
    clamp,
    extract_json_object,
    validate_inputs,
)
from oumi.utils.logging import logger


# Module-level state
_judge: RubricJudge | None = None
_stats = RubricStats()


def get_stats() -> RubricStats:
    """Get the current statistics for weighted_rubric_reward."""
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
                "You are an expert evaluator. Assess responses against weighted "
                "rubrics fairly and consistently. For each rubric, give 1 if "
                "satisfied, 0 if not. Return valid JSON only."
            ),
            prompt_template=(
                "Evaluate the response against the weighted rubrics.\n\n"
                "## Task\n{prompt}\n\n"
                "## Response to Evaluate\n{response}\n\n"
                "## Rubrics (with weights)\n{rubrics}\n\n"
                "Evaluate the response against EACH rubric independently.\n"
                "For each rubric, determine if it is satisfied (1) or not (0).\n\n"
                "Return a JSON object with:\n"
                '- "scores": dict mapping rubric name to 0 or 1\n'
                '- "weighted_score": the weighted average '
                "(sum(weight * score) / sum(weights))\n\n"
                "Output only valid JSON, no other text."
            ),
        )
    return _judge


def validate_weighted_rubrics(rubrics: list[dict]) -> None:
    """Validate that rubrics are properly formatted weighted dicts.

    Args:
        rubrics: List of rubric dictionaries.

    Raises:
        ValueError: If rubrics are not valid weighted format.
    """
    for i, rubric in enumerate(rubrics, 1):
        if not isinstance(rubric, dict):
            raise ValueError(
                f"Weighted rubrics must be dicts. Got {type(rubric).__name__} "
                f"at position {i}."
            )
        if "description" not in rubric and "weight" not in rubric:
            raise ValueError(
                f"Weighted rubric must include 'description' or 'weight'. "
                f"Missing in rubric {i}: {rubric}"
            )


def format_weighted_rubrics(rubrics: list[dict]) -> str:
    """Format weighted rubrics for the judge prompt.

    Args:
        rubrics: List of weighted rubric dictionaries.

    Returns:
        Formatted rubrics text.
    """
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        name = rubric.get("name") or f"rubric_{i}"
        description = rubric.get("description", "")
        weight = rubric.get("weight", 1.0)
        lines.append(f"{i}. [{name}] (weight={weight}): {description}")
    return "\n".join(lines)


def compute_weighted_score(
    rubrics: list[dict],
    per_rubric_scores: dict[str, float],
) -> float:
    """Compute weighted score from per-rubric scores.

    Supports negative weights for pitfall criteria. When a rubric has a
    negative weight (pitfall):
    - Score 1 (avoided pitfall) -> positive contribution
    - Score 0 (hit pitfall) -> negative contribution

    Args:
        rubrics: List of weighted rubric dicts.
        per_rubric_scores: Dict mapping rubric names to scores (0 or 1).

    Returns:
        Weighted score normalized to [0.0, 1.0].
    """
    total_weight = 0.0
    weighted_sum = 0.0
    has_negative = False

    for i, rubric in enumerate(rubrics, 1):
        name = rubric.get("name") or f"rubric_{i}"
        weight = float(rubric.get("weight", 1.0))
        is_pitfall = weight < 0

        if is_pitfall:
            has_negative = True

        raw_score = clamp(float(per_rubric_scores.get(name, 0)))

        if is_pitfall:
            abs_weight = abs(weight)
            # Pitfall avoided (1) -> +abs_weight, hit (0) -> -abs_weight
            contribution = abs_weight * (2 * raw_score - 1)
            weighted_sum += contribution
            total_weight += abs_weight
        else:
            weighted_sum += weight * raw_score
            total_weight += weight

    if total_weight == 0:
        return 0.0

    if not has_negative:
        return clamp(weighted_sum / total_weight)

    # Normalize to [0, 1] for pitfall-style scoring
    normalized = (weighted_sum + total_weight) / (2 * total_weight)
    return clamp(normalized)


def parse_weighted_response(response: str) -> tuple[dict[str, float], float | None]:
    """Parse per-rubric scores and weighted score from judge response.

    Args:
        response: Raw judge response text.

    Returns:
        Tuple of (scores dict, weighted_score or None).
    """
    scores: dict[str, float] = {}
    weighted_score: float | None = None

    # Try to extract JSON object
    json_text = extract_json_object(response)
    if json_text:
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                # Extract scores
                raw_scores = data.get("scores", data)
                if isinstance(raw_scores, dict):
                    for key, value in raw_scores.items():
                        if key not in ("scores", "weighted_score"):
                            try:
                                scores[str(key)] = float(value)
                            except (TypeError, ValueError):
                                continue

                # Extract weighted_score
                if "weighted_score" in data:
                    try:
                        weighted_score = float(data["weighted_score"])
                    except (TypeError, ValueError):
                        pass

                return scores, weighted_score
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction
    weighted_match = re.search(
        r'"?weighted_score"?\s*:\s*([+-]?\d+(?:\.\d+)?)',
        response,
    )
    if weighted_match:
        try:
            weighted_score = float(weighted_match.group(1))
        except ValueError:
            pass

    # Extract individual scores
    score_pattern = r'"([^"]+)"\s*:\s*([+-]?\d+(?:\.\d+)?)'
    for match in re.finditer(score_pattern, response):
        name, score_str = match.groups()
        if name not in ("scores", "weighted_score"):
            try:
                scores[name] = float(score_str)
            except ValueError:
                continue

    return scores, weighted_score


@register("weighted_rubric_reward", RegistryType.REWARD_FUNCTION)
def weighted_rubric_reward(
    completions: list[list[dict[str, Any]]],
    prompts: list[str] | None = None,
    rubrics: list[list[dict]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Evaluate completions against weighted rubrics using an LLM judge.

    This reward function supports weighted rubrics where each rubric has
    a weight indicating its importance. It also supports "pitfall" criteria
    with negative weights that penalize violations.

    Args:
        completions: List of model completions to evaluate.
        prompts: List of original prompts/tasks.
        rubrics: List of rubric lists, where each rubric is a dict with:
            - name (optional): Identifier for the rubric
            - description: What the rubric evaluates
            - weight: Importance weight (negative for pitfalls)
        **kwargs: Additional arguments:
            - judge_model: Model to use for judging (default: "gpt-4o-mini")
            - system_prompt: Optional system prompts to prepend

    Returns:
        List of float rewards in [0.0, 1.0], representing the weighted
        score for each completion.

    Example:
        >>> rewards = weighted_rubric_reward(
        ...     completions=[[{"content": "The answer is 42."}]],
        ...     prompts=["What is the answer?"],
        ...     rubrics=[[
        ...         {"name": "accuracy", "description": "Correct answer", "weight": 2.0},
        ...         {"name": "concise", "description": "Brief response", "weight": 1.0},
        ...         {"name": "verbose", "description": "Avoids fluff", "weight": -1.0},
        ...     ]],
        ... )
    """
    global _stats

    # Handle alternative parameter names
    prompt_list = prompts or kwargs.get("prompt", [])
    rubric_list = rubrics or kwargs.get("rubrics", [])
    system_prompts: list[Any] = kwargs.get("system_prompt", [])

    # Validate inputs
    completion_strs, prompt_list, rubric_list, count = validate_inputs(
        completions, prompt_list, rubric_list, "weighted_rubric_reward"
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

        # Validate rubrics
        try:
            validate_weighted_rubrics(rubric_items)
        except ValueError as e:
            logger.warning(f"[weighted_rubric_reward] Invalid rubrics: {e}")
            rewards.append(0.0)
            _stats.record_failure()
            continue

        # Build full prompt with system prompt if provided
        full_prompt = prompt
        if system_prompts and i < len(system_prompts) and system_prompts[i]:
            full_prompt = f"[System: {system_prompts[i]}]\n\n{prompt}"

        # Format rubrics
        rubrics_text = format_weighted_rubrics(rubric_items)

        # Evaluate
        result = judge.evaluate(full_prompt, comp, rubrics_text)

        if result.success:
            # Parse the response
            per_rubric_scores, weighted_score = parse_weighted_response(
                result.raw_response
            )

            # Use weighted_score if provided, otherwise compute
            if weighted_score is not None:
                score = clamp(weighted_score)
            elif per_rubric_scores:
                score = compute_weighted_score(rubric_items, per_rubric_scores)
            else:
                score = result.score

            _stats.record_success(score, result.time_ms)
            rewards.append(score)
        else:
            _stats.record_failure(result.time_ms)
            rewards.append(0.0)

    # Log batch summary
    if _stats.should_log():
        logger.info(_stats.get_summary())

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info(
        f"[weighted_rubric_reward] Batch: size={count}, avg_reward={avg_reward:.3f}"
    )

    return rewards
