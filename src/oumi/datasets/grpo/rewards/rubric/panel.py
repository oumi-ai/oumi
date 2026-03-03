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

"""Panel-based rubric reward function.

Evaluates model completions using a panel of multiple LLM judges for
more robust and reliable scoring. Supports both string and weighted rubrics.
"""

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

from oumi.core.registry import RegistryType, register
from oumi.datasets.grpo.rewards.rubric.core import (
    RubricJudge,
    RubricStats,
    clamp,
    format_string_rubrics,
    validate_inputs,
)
from oumi.datasets.grpo.rewards.rubric.weighted import (
    compute_weighted_score,
    format_weighted_rubrics,
    parse_weighted_response,
    validate_weighted_rubrics,
)
from oumi.utils.logging import logger


class AggregationStrategy(str, Enum):
    """Strategies for aggregating scores from multiple judges."""

    MEAN = "mean"
    MEDIAN = "median"


@dataclass
class PanelMember:
    """Configuration for a single judge in a panel."""

    model: str = "gpt-4o-mini"
    weight: float = 1.0
    temperature: float = 0.0

    @classmethod
    def from_config(cls, config: str | dict) -> "PanelMember":
        """Create a PanelMember from a string or dict config."""
        if isinstance(config, str):
            return cls(model=config)
        elif isinstance(config, dict):
            return cls(
                model=config.get("model", "gpt-4o-mini"),
                weight=config.get("weight", 1.0),
                temperature=config.get("temperature", 0.0),
            )
        else:
            raise ValueError(f"Invalid panel member config: {config}")


@dataclass
class PanelStats(RubricStats):
    """Extended statistics for panel evaluation."""

    variance_history: list[float] = field(default_factory=list)

    def record_panel_result(
        self,
        reward: float,
        time_ms: float,
        variance: float | None = None,
    ) -> None:
        """Record a panel evaluation result."""
        self.record_success(reward, time_ms)
        if variance is not None:
            self.variance_history.append(variance)

    @property
    def avg_variance(self) -> float:
        """Average variance across panel evaluations."""
        if not self.variance_history:
            return 0.0
        return sum(self.variance_history) / len(self.variance_history)

    def get_summary(self) -> str:
        """Return a summary including variance info."""
        base = super().get_summary()
        if self.variance_history:
            return f"{base}, avg_variance={self.avg_variance:.4f}"
        return base


# Module-level state
_judges: dict[str, RubricJudge] = {}
_stats = PanelStats()


def get_stats() -> PanelStats:
    """Get the current statistics for panel_rubric_reward."""
    return _stats


def reset_stats() -> None:
    """Reset the statistics."""
    global _stats
    _stats = PanelStats()


def _get_judge(model: str, temperature: float, weighted: bool) -> RubricJudge:
    """Get or create a judge instance."""
    cache_key = f"{model}_{temperature}_{weighted}"
    if cache_key not in _judges:
        if weighted:
            system_instruction = (
                "You are an expert evaluator. Assess responses against weighted "
                "rubrics fairly and consistently. For each rubric, give 1 if "
                "satisfied, 0 if not. Return valid JSON only."
            )
            prompt_template = (
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
            )
        else:
            system_instruction = (
                "You are an expert evaluator. Assess responses against rubrics "
                "fairly and consistently. For each rubric, determine if the "
                "response satisfies it (1) or not (0). Return valid JSON only."
            )
            prompt_template = (
                "Evaluate the response against the rubrics.\n\n"
                "## Task\n{prompt}\n\n"
                "## Response to Evaluate\n{response}\n\n"
                "## Rubrics\n{rubrics}\n\n"
                "For each rubric, determine if it is satisfied (1) or not (0).\n"
                "Return a JSON object with:\n"
                '- "scores": dict mapping rubric number to 0 or 1\n'
                '- "total_score": fraction of rubrics satisfied (0.0 to 1.0)\n\n'
                "Example: {\"scores\": {\"1\": 1, \"2\": 0, \"3\": 1}, "
                '"total_score": 0.67}\n\n'
                "Output only valid JSON."
            )

        _judges[cache_key] = RubricJudge(
            model=model,
            temperature=temperature,
            system_instruction=system_instruction,
            prompt_template=prompt_template,
        )
    return _judges[cache_key]


def aggregate_scores(
    scores: list[float],
    weights: list[float],
    strategy: AggregationStrategy,
) -> tuple[float, float]:
    """Aggregate scores from multiple judges.

    Args:
        scores: List of scores from each judge.
        weights: List of weights for each judge.
        strategy: Aggregation strategy to use.

    Returns:
        Tuple of (aggregated_score, variance).
    """
    if not scores:
        return 0.0, 0.0

    if strategy == AggregationStrategy.MEAN:
        # Weighted mean
        if sum(weights) == 0:
            result = sum(scores) / len(scores)
        else:
            result = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    elif strategy == AggregationStrategy.MEDIAN:
        result = statistics.median(scores)
    else:
        result = sum(scores) / len(scores)

    variance = statistics.variance(scores) if len(scores) > 1 else 0.0
    return clamp(result), variance


def is_weighted_rubrics(rubrics: list) -> bool:
    """Check if rubrics are in weighted dict format."""
    if not rubrics:
        return False
    return all(isinstance(r, dict) for r in rubrics)


def load_panel_config(path: str) -> list[PanelMember]:
    """Load panel configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        List of PanelMember instances.

    Expected YAML format:
        judges:
          - model: gpt-4o
            weight: 2.0
          - model: gpt-4o-mini
            weight: 1.0
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    judges_config = config.get("judges", [])
    return [PanelMember.from_config(j) for j in judges_config]


@register("panel_rubric_reward", RegistryType.REWARD_FUNCTION)
def panel_rubric_reward(
    completions: list[list[dict[str, Any]]],
    prompts: list[str] | None = None,
    rubrics: list[list] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Evaluate completions using a panel of multiple LLM judges.

    This reward function uses multiple judges to evaluate each completion
    and aggregates their scores for more robust evaluation. Supports both
    simple string rubrics and weighted rubrics.

    Args:
        completions: List of model completions to evaluate.
        prompts: List of original prompts/tasks.
        rubrics: List of rubric lists. Can be:
            - Strings: ["criterion 1", "criterion 2"]
            - Weighted dicts: [{"description": "...", "weight": 2.0}, ...]
        **kwargs: Additional arguments:
            - judges: List of judge configs. Each can be:
                - str: Model name (e.g., "gpt-4o")
                - dict: {"model": "gpt-4o", "weight": 2.0, "temperature": 0.0}
            - panel_config_path: Path to YAML file with panel configuration
            - aggregation: "mean" (default) or "median"
            - system_prompt: Optional system prompts to prepend

    Returns:
        List of float rewards in [0.0, 1.0], representing the aggregated
        score from all judges for each completion.

    Example:
        >>> rewards = panel_rubric_reward(
        ...     completions=[[{"content": "The answer is 42."}]],
        ...     prompts=["What is the answer?"],
        ...     rubrics=[["Provides a numerical answer", "Is concise"]],
        ...     judges=["gpt-4o", "gpt-4o-mini", "claude-3-haiku"],
        ...     aggregation="mean",
        ... )
    """
    global _stats

    # Handle alternative parameter names
    prompt_list = prompts or kwargs.get("prompt", [])
    rubric_list = rubrics or kwargs.get("rubrics", [])
    system_prompts: list[Any] = kwargs.get("system_prompt", [])

    # Validate inputs
    completion_strs, prompt_list, rubric_list, count = validate_inputs(
        completions, prompt_list, rubric_list, "panel_rubric_reward"
    )

    if count == 0:
        return [0.0] * len(completions)

    # Parse panel configuration
    panel_members: list[PanelMember] = []

    # Load from file if path provided
    panel_config_path = kwargs.get("panel_config_path")
    if panel_config_path:
        panel_members = load_panel_config(panel_config_path)

    # Override with inline judges config
    judges_config = kwargs.get("judges", [])
    if judges_config:
        panel_members = [PanelMember.from_config(j) for j in judges_config]

    # Default to single judge if no panel configured
    if not panel_members:
        judge_model = str(kwargs.get("judge_model", "gpt-4o-mini"))
        panel_members = [PanelMember(model=judge_model)]

    # Parse aggregation strategy
    agg_str = str(kwargs.get("aggregation", "mean")).lower()
    try:
        aggregation = AggregationStrategy(agg_str)
    except ValueError:
        logger.warning(
            f"[panel_rubric_reward] Unknown aggregation '{agg_str}', using 'mean'"
        )
        aggregation = AggregationStrategy.MEAN

    # Evaluate each completion
    rewards = []
    for i, (comp, prompt, rubric_items) in enumerate(
        zip(completion_strs, prompt_list, rubric_list)
    ):
        # Ensure rubrics is a list
        if not isinstance(rubric_items, list):
            rubric_items = [rubric_items]

        # Determine if weighted
        weighted = is_weighted_rubrics(rubric_items)

        # Validate weighted rubrics
        if weighted:
            try:
                validate_weighted_rubrics(rubric_items)
            except ValueError as e:
                logger.warning(f"[panel_rubric_reward] Invalid rubrics: {e}")
                rewards.append(0.0)
                _stats.record_failure()
                continue

        # Build full prompt with system prompt if provided
        full_prompt = prompt
        if system_prompts and i < len(system_prompts) and system_prompts[i]:
            full_prompt = f"[System: {system_prompts[i]}]\n\n{prompt}"

        # Format rubrics
        if weighted:
            rubrics_text = format_weighted_rubrics(rubric_items)
        else:
            rubrics_text = format_string_rubrics(rubric_items)

        # Evaluate with each judge in the panel
        panel_scores: list[float] = []
        panel_weights: list[float] = []
        total_time_ms = 0.0

        for member in panel_members:
            judge = _get_judge(member.model, member.temperature, weighted)
            result = judge.evaluate(full_prompt, comp, rubrics_text)
            total_time_ms += result.time_ms

            if result.success:
                if weighted:
                    per_rubric_scores, weighted_score = parse_weighted_response(
                        result.raw_response
                    )
                    if weighted_score is not None:
                        score = clamp(weighted_score)
                    elif per_rubric_scores:
                        score = compute_weighted_score(rubric_items, per_rubric_scores)
                    else:
                        score = result.score
                else:
                    score = result.score

                panel_scores.append(score)
                panel_weights.append(member.weight)
            else:
                logger.debug(
                    f"[panel_rubric_reward] Judge {member.model} failed for item {i}"
                )

        # Aggregate panel scores
        if panel_scores:
            reward, variance = aggregate_scores(
                panel_scores, panel_weights, aggregation
            )
            _stats.record_panel_result(reward, total_time_ms, variance)
            rewards.append(reward)
        else:
            _stats.record_failure(total_time_ms)
            rewards.append(0.0)

    # Log batch summary
    if _stats.should_log():
        logger.info(_stats.get_summary())

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info(
        f"[panel_rubric_reward] Batch: size={count}, avg_reward={avg_reward:.3f}, "
        f"judges={len(panel_members)}"
    )

    return rewards
