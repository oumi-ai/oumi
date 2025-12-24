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

"""Rubric-based reward function for RLVR (RL from Verifiable Rewards).

This module provides a reward function that uses LLM judges to evaluate
model completions against a set of rubrics. This is the core component
for implementing RLVR with rubric-based evaluation.
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.registry import RegistryType, register
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger


class AggregationStrategy(str, Enum):
    """Strategies for aggregating scores from multiple judges."""

    MEAN = "mean"  # Simple average
    WEIGHTED_MEAN = "weighted_mean"  # Weighted by judge weights
    MEDIAN = "median"  # Median score (robust to outliers)
    MIN = "min"  # Most conservative (lowest score)
    MAX = "max"  # Most lenient (highest score)
    MAJORITY_VOTE = "majority_vote"  # Round to 0/1, take majority


@dataclass
class JudgePanelMember:
    """Configuration for a single judge in a panel.

    Attributes:
        model: The model name (e.g., "gpt-4o-mini", "gpt-4o").
        weight: Weight for this judge in weighted aggregation (default: 1.0).
        name: Optional name for logging (defaults to model name).
        temperature: Temperature for this judge (default: 0.0).
        role: Optional role description (e.g., "technical accuracy", "tone").
    """

    model: str = "gpt-4o-mini"
    weight: float = 1.0
    name: str | None = None
    temperature: float = 0.0
    role: str | None = None

    def __post_init__(self) -> None:
        """Set default name to model name if not provided."""
        if self.name is None:
            self.name = self.model


@dataclass
class JudgePanelConfig:
    """Configuration for a panel of judges.

    Attributes:
        judges: List of judge configurations.
        aggregation: How to aggregate scores from multiple judges.
        require_unanimous: If True, all judges must agree (for majority vote).
        min_agreement: Minimum fraction of judges that must agree (0.0-1.0).
    """

    judges: list[JudgePanelMember] = field(default_factory=list)
    aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN
    require_unanimous: bool = False
    min_agreement: float = 0.0  # 0 = no minimum required

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "JudgePanelConfig":
        """Create a JudgePanelConfig from a dictionary.

        Args:
            config: Dictionary with panel configuration.

        Returns:
            JudgePanelConfig instance.
        """
        judges = []
        for j in config.get("judges", []):
            if isinstance(j, str):
                # Simple string format: just model name
                judges.append(JudgePanelMember(model=j))
            elif isinstance(j, dict):
                judges.append(JudgePanelMember(**j))
            elif isinstance(j, JudgePanelMember):
                judges.append(j)

        aggregation = config.get("aggregation", "weighted_mean")
        if isinstance(aggregation, str):
            aggregation = AggregationStrategy(aggregation)

        return cls(
            judges=judges,
            aggregation=aggregation,
            require_unanimous=config.get("require_unanimous", False),
            min_agreement=config.get("min_agreement", 0.0),
        )

    @classmethod
    def single_judge(cls, model: str = "gpt-4o-mini") -> "JudgePanelConfig":
        """Create a single-judge panel (default behavior).

        Args:
            model: The model to use.

        Returns:
            JudgePanelConfig with one judge.
        """
        return cls(judges=[JudgePanelMember(model=model)])


def _aggregate_scores(
    scores: list[float],
    weights: list[float],
    strategy: AggregationStrategy,
) -> tuple[float, dict[str, Any]]:
    """Aggregate scores from multiple judges.

    Args:
        scores: List of scores from each judge.
        weights: List of weights for each judge.
        strategy: Aggregation strategy to use.

    Returns:
        Tuple of (aggregated_score, details_dict).
    """
    if not scores:
        return 0.0, {"error": "no scores"}

    details: dict[str, Any] = {
        "strategy": strategy.value,
        "individual_scores": scores,
        "weights": weights,
    }

    if strategy == AggregationStrategy.MEAN:
        result = sum(scores) / len(scores)

    elif strategy == AggregationStrategy.WEIGHTED_MEAN:
        if sum(weights) == 0:
            result = sum(scores) / len(scores)
        else:
            result = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    elif strategy == AggregationStrategy.MEDIAN:
        result = statistics.median(scores)

    elif strategy == AggregationStrategy.MIN:
        result = min(scores)

    elif strategy == AggregationStrategy.MAX:
        result = max(scores)

    elif strategy == AggregationStrategy.MAJORITY_VOTE:
        # Round scores to 0 or 1, then take majority
        votes = [1 if s >= 0.5 else 0 for s in scores]
        result = 1.0 if sum(votes) > len(votes) / 2 else 0.0
        details["votes"] = votes

    else:
        # Default to mean
        result = sum(scores) / len(scores)

    details["aggregated_score"] = result
    details["score_variance"] = statistics.variance(scores) if len(scores) > 1 else 0.0

    return result, details


# Global judge instances (lazily initialized)
_JUDGE_INSTANCE: SimpleJudge | None = None
_JUDGE_PANEL: dict[str, SimpleJudge] = {}  # Cache for panel judges by model name


# Pricing per 1M tokens (as of Dec 2024)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is a rough approximation; actual counts may vary by model.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Rough estimate: 4 chars per token for English
    return max(1, len(text) // 4)


@dataclass
class JudgeStats:
    """Statistics for a single judge in a panel."""

    name: str = ""
    model: str = ""
    calls: int = 0
    successful: int = 0
    failed: int = 0
    total_score: float = 0.0
    total_time_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def avg_score(self) -> float:
        """Average score across successful evaluations."""
        return self.total_score / self.successful if self.successful > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average evaluation time in milliseconds."""
        return self.total_time_ms / self.successful if self.successful > 0 else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD based on token usage."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["gpt-4o-mini"])
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class RubricRewardStats:
    """Statistics tracker for rubric-based reward computation."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_reward: float = 0.0
    total_judge_time_ms: float = 0.0
    rewards_history: list[float] = field(default_factory=list)
    log_interval: int = 10  # Log every N calls

    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    judge_model: str = "gpt-4o-mini"

    # Panel tracking
    panel_stats: dict[str, JudgeStats] = field(default_factory=dict)
    panel_variance_history: list[float] = field(default_factory=list)

    def record_success(
        self,
        reward: float,
        judge_time_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a successful reward computation."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_reward += reward
        self.total_judge_time_ms += judge_time_ms
        self.rewards_history.append(reward)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_failure(self, input_tokens: int = 0) -> None:
        """Record a failed reward computation."""
        self.total_calls += 1
        self.failed_calls += 1
        self.rewards_history.append(0.0)
        # Still count input tokens for failed calls (we still paid for the prompt)
        self.total_input_tokens += input_tokens

    def record_panel_result(
        self,
        judge_name: str,
        judge_model: str,
        score: float,
        time_ms: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
    ) -> None:
        """Record result from a single judge in a panel."""
        if judge_name not in self.panel_stats:
            self.panel_stats[judge_name] = JudgeStats(
                name=judge_name, model=judge_model
            )

        stats = self.panel_stats[judge_name]
        stats.calls += 1
        stats.input_tokens += input_tokens
        stats.output_tokens += output_tokens

        if success:
            stats.successful += 1
            stats.total_score += score
            stats.total_time_ms += time_ms
        else:
            stats.failed += 1

    def record_panel_variance(self, variance: float) -> None:
        """Record variance across panel judges for this evaluation."""
        self.panel_variance_history.append(variance)

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD based on model pricing."""
        # If we have panel stats, sum their costs
        if self.panel_stats:
            return sum(s.estimated_cost_usd for s in self.panel_stats.values())
        # Otherwise use single judge pricing
        pricing = MODEL_PRICING.get(self.judge_model, MODEL_PRICING["gpt-4o-mini"])
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @property
    def avg_reward(self) -> float:
        """Average reward across all calls."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_reward / self.successful_calls

    @property
    def avg_judge_time_ms(self) -> float:
        """Average judge inference time in milliseconds."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_judge_time_ms / self.successful_calls

    @property
    def avg_panel_variance(self) -> float:
        """Average variance across panel judges."""
        if not self.panel_variance_history:
            return 0.0
        return sum(self.panel_variance_history) / len(self.panel_variance_history)

    @property
    def success_rate(self) -> float:
        """Success rate of judge calls."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    def should_log(self) -> bool:
        """Check if we should log stats (every log_interval calls)."""
        return self.total_calls > 0 and self.total_calls % self.log_interval == 0

    def get_summary(self) -> str:
        """Get a summary string of current stats."""
        base = (
            f"RubricReward Stats: "
            f"calls={self.total_calls}, "
            f"success_rate={self.success_rate:.1%}, "
            f"avg_reward={self.avg_reward:.3f}, "
            f"avg_judge_time={self.avg_judge_time_ms:.0f}ms, "
            f"failed={self.failed_calls}"
        )
        if self.panel_stats:
            base += f", panel_variance={self.avg_panel_variance:.4f}"
        return base

    def get_cost_summary(self) -> str:
        """Get a summary string of token usage and costs."""
        if self.panel_stats:
            # Panel mode: show per-judge costs
            lines = [f"Panel Cost Summary (total: ${self.estimated_cost_usd:.4f}):"]
            for name, stats in self.panel_stats.items():
                lines.append(
                    f"  {name}: ${stats.estimated_cost_usd:.4f} "
                    f"(in={stats.input_tokens:,}, out={stats.output_tokens:,}, "
                    f"avg_score={stats.avg_score:.3f})"
                )
            return "\n".join(lines)
        else:
            return (
                f"Token Usage: "
                f"input={self.total_input_tokens:,}, "
                f"output={self.total_output_tokens:,}, "
                f"total={self.total_tokens:,} | "
                f"Est. Cost: ${self.estimated_cost_usd:.4f} ({self.judge_model})"
            )

    def get_full_summary(self) -> str:
        """Get a complete summary including stats and costs."""
        return f"{self.get_summary()}\n{self.get_cost_summary()}"

    def get_recent_rewards(self, n: int = 10) -> list[float]:
        """Get the most recent N rewards."""
        return self.rewards_history[-n:] if self.rewards_history else []

    def get_panel_summary(self) -> str:
        """Get a detailed summary of panel judge performance."""
        if not self.panel_stats:
            return "No panel data"

        lines = ["Panel Judge Summary:"]
        for name, stats in self.panel_stats.items():
            lines.append(
                f"  {name} ({stats.model}): "
                f"calls={stats.calls}, "
                f"avg_score={stats.avg_score:.3f}, "
                f"avg_time={stats.avg_time_ms:.0f}ms, "
                f"cost=${stats.estimated_cost_usd:.4f}"
            )
        return "\n".join(lines)


# Global stats tracker
_REWARD_STATS = RubricRewardStats()


def get_rubric_reward_stats() -> RubricRewardStats:
    """Get the global rubric reward statistics.

    Returns:
        The RubricRewardStats instance with current statistics.
    """
    return _REWARD_STATS


def reset_rubric_reward_stats() -> None:
    """Reset the global rubric reward statistics."""
    global _REWARD_STATS
    _REWARD_STATS = RubricRewardStats()


def _build_judge_config(
    model: str,
    temperature: float,
    weighted_rubrics: bool,
    role: str | None = None,
) -> JudgeConfig:
    """Build a JudgeConfig for rubric evaluation.

    Args:
        model: The model name to use for judging.
        temperature: The temperature for generation.
        weighted_rubrics: Whether to use weighted rubric evaluation.
        role: Optional role description for the judge.

    Returns:
        A JudgeConfig instance.
    """
    from oumi.core.configs.inference_config import InferenceConfig
    from oumi.core.configs.inference_engine_type import InferenceEngineType
    from oumi.core.configs.params.generation_params import GenerationParams
    from oumi.core.configs.params.model_params import ModelParams

    role_suffix = f"\n\nYour specific focus is: {role}" if role else ""

    if weighted_rubrics:
        prompt_template = (
            "You are evaluating a response based on specific weighted rubrics.\n\n"
            "## Task\n{prompt}\n\n"
            "## Response to Evaluate\n{response}\n\n"
            "## Rubrics (with weights)\n{rubrics}\n\n"
            "Evaluate the response against EACH rubric independently.\n"
            "For each rubric, determine if it is satisfied (1) or not (0).\n\n"
            "Return a JSON object with two keys:\n"
            '- "scores": a dict mapping each rubric name to 0 or 1\n'
            '- "weighted_score": weighted avg (sum(weight*score)/sum(weights))\n\n'
            "Output only valid JSON, no other text."
        )
        system_instruction = (
            "You are an expert evaluator. Assess responses against weighted "
            "rubrics fairly and consistently. For each rubric, give 1 if "
            "satisfied, 0 if not. Calculate weighted score as "
            f"sum(weight*score)/sum(weights). Return valid JSON only.{role_suffix}"
        )
        max_tokens = 500
    else:
        prompt_template = (
            "You are evaluating a response based on specific rubrics.\n\n"
            "## Task\n{prompt}\n\n"
            "## Response to Evaluate\n{response}\n\n"
            "## Rubrics\n{rubrics}\n\n"
            "Evaluate the response against EACH rubric. "
            "Count how many rubrics the response satisfies.\n"
            "Your judgment should be a float between 0.0 and 1.0, "
            "representing the fraction of rubrics satisfied."
        )
        system_instruction = (
            "You are an expert evaluator. Assess responses against rubrics "
            "fairly and consistently. Return only the fraction of rubrics "
            f"that are satisfied (0.0 to 1.0).{role_suffix}"
        )
        max_tokens = 100

    return JudgeConfig(
        judge_params=JudgeParams(
            prompt_template=prompt_template,
            response_format=JudgeResponseFormat.XML,
            judgment_type=JudgeOutputType.FLOAT,
            include_explanation=False,
            system_instruction=system_instruction,
        ),
        inference_config=InferenceConfig(
            engine=InferenceEngineType.OPENAI,
            model=ModelParams(model_name=model),
            generation=GenerationParams(
                max_new_tokens=max_tokens,
                temperature=temperature,
            ),
        ),
    )


def _get_or_create_judge(
    judge_model: str = "gpt-4o-mini",
    judge_config_path: str | None = None,
    weighted_rubrics: bool = False,
) -> SimpleJudge:
    """Get or create a SimpleJudge instance.

    Args:
        judge_model: The model to use for judging (default: gpt-4o-mini).
        judge_config_path: Optional path to a judge config file.
        weighted_rubrics: Whether to use weighted rubric evaluation.

    Returns:
        A SimpleJudge instance.
    """
    global _JUDGE_INSTANCE, _REWARD_STATS

    _REWARD_STATS.judge_model = judge_model

    if _JUDGE_INSTANCE is not None:
        return _JUDGE_INSTANCE

    if judge_config_path and Path(judge_config_path).exists():
        _JUDGE_INSTANCE = SimpleJudge(judge_config_path)
    else:
        config = _build_judge_config(judge_model, 0.0, weighted_rubrics)
        _JUDGE_INSTANCE = SimpleJudge(config)

    return _JUDGE_INSTANCE


def _create_judge_for_panel(
    member: JudgePanelMember,
    weighted_rubrics: bool = False,
) -> SimpleJudge:
    """Create a judge instance for a panel member.

    Args:
        member: The panel member configuration.
        weighted_rubrics: Whether to use weighted rubric evaluation.

    Returns:
        A SimpleJudge instance.
    """
    global _JUDGE_PANEL

    cache_key = f"{member.model}_{member.temperature}_{weighted_rubrics}"
    if cache_key in _JUDGE_PANEL:
        return _JUDGE_PANEL[cache_key]

    config = _build_judge_config(
        member.model, member.temperature, weighted_rubrics, member.role
    )
    judge = SimpleJudge(config)
    _JUDGE_PANEL[cache_key] = judge
    return judge


def _get_or_create_panel(
    panel_config: JudgePanelConfig,
    weighted_rubrics: bool = False,
) -> list[tuple[JudgePanelMember, SimpleJudge]]:
    """Get or create judges for a panel.

    Args:
        panel_config: Panel configuration.
        weighted_rubrics: Whether to use weighted rubric evaluation.

    Returns:
        List of (member, judge) tuples.
    """
    panel = []
    for member in panel_config.judges:
        judge = _create_judge_for_panel(member, weighted_rubrics)
        panel.append((member, judge))
    return panel


# Global panel config (can be set via environment or load_panel_config)
_PANEL_CONFIG: JudgePanelConfig | None = None


def load_panel_config(config_path: str | None = None) -> JudgePanelConfig | None:
    """Load judge panel configuration from file or environment.

    Configuration sources (in priority order):
    1. Explicit config_path argument
    2. OUMI_JUDGE_PANEL_CONFIG environment variable (path to JSON file)
    3. OUMI_JUDGE_PANEL environment variable (inline JSON)

    Args:
        config_path: Optional path to JSON config file.

    Returns:
        JudgePanelConfig if found, None otherwise.

    Example JSON config:
        {
            "judges": [
                {"model": "gpt-4o-mini", "weight": 1.0, "name": "fast"},
                {"model": "gpt-4o", "weight": 2.0, "name": "strong"}
            ],
            "aggregation": "weighted_mean"
        }
    """
    global _PANEL_CONFIG

    # Return cached config if available
    if _PANEL_CONFIG is not None:
        return _PANEL_CONFIG

    config_data = None

    # Priority 1: Explicit path
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                config_data = json.load(f)
            logger.info(f"[RLVR] Loaded judge panel config from: {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load panel config from {config_path}: {e}")

    # Priority 2: Environment variable path
    if config_data is None:
        env_path = os.environ.get("OUMI_JUDGE_PANEL_CONFIG")
        if env_path and Path(env_path).exists():
            try:
                with open(env_path) as f:
                    config_data = json.load(f)
                logger.info(f"[RLVR] Loaded judge panel config from env: {env_path}")
            except Exception as e:
                logger.warning(f"Failed to load panel config from {env_path}: {e}")

    # Priority 3: Inline JSON from environment
    if config_data is None:
        env_json = os.environ.get("OUMI_JUDGE_PANEL")
        if env_json:
            try:
                config_data = json.loads(env_json)
                logger.info(
                    "[RLVR] Loaded judge panel config from OUMI_JUDGE_PANEL env"
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse OUMI_JUDGE_PANEL: {e}")

    if config_data:
        _PANEL_CONFIG = JudgePanelConfig.from_dict(config_data)
        return _PANEL_CONFIG

    return None


def set_panel_config(config: JudgePanelConfig) -> None:
    """Set the global judge panel configuration programmatically.

    Args:
        config: The panel configuration to use.
    """
    global _PANEL_CONFIG
    _PANEL_CONFIG = config
    logger.info(f"[RLVR] Set judge panel config: {len(config.judges)} judges")


def clear_panel_config() -> None:
    """Clear the global judge panel configuration."""
    global _PANEL_CONFIG
    _PANEL_CONFIG = None


def _extract_completion_strings(completions: list) -> list[str]:
    """Extract string content from completions in various formats.

    TRL may pass completions in different formats depending on version:
    - list[str]: Plain strings
    - list[list[dict]]: List of message dicts
    - list[dict]: Single message dicts

    Args:
        completions: Completions in any supported format.

    Returns:
        List of completion strings.
    """
    completion_strs = []
    for c in completions:
        if isinstance(c, str):
            # Completions are plain strings
            completion_strs.append(c)
        elif isinstance(c, list) and len(c) > 0:
            # Completions are list of message dicts
            if isinstance(c[0], dict):
                completion_strs.append(c[0].get("content", str(c[0])))
            else:
                completion_strs.append(str(c[0]))
        elif isinstance(c, dict):
            # Completion is a single message dict
            completion_strs.append(c.get("content", str(c)))
        else:
            completion_strs.append(str(c))
    return completion_strs


def _is_weighted_rubrics(rubrics: list) -> bool:
    """Check if rubrics are in the weighted format (list of dicts).

    Args:
        rubrics: List of rubrics.

    Returns:
        True if rubrics are weighted format (dicts with name/description/weight).
    """
    if not rubrics:
        return False
    first = rubrics[0]
    return isinstance(first, dict) and "description" in first


def _format_weighted_rubrics(rubrics: list[dict[str, Any]]) -> str:
    """Format weighted rubrics for the judge prompt.

    Args:
        rubrics: List of rubric dicts with name, description, weight.

    Returns:
        Formatted string representation.
    """
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        name = rubric.get("name", f"rubric_{i}")
        description = rubric.get("description", "")
        weight = rubric.get("weight", 1.0)
        eval_type = rubric.get("evaluation_type", "binary")
        lines.append(
            f"{i}. [{name}] (weight={weight}, type={eval_type}): {description}"
        )
    return "\n".join(lines)


def _format_simple_rubrics(rubrics: list[str]) -> str:
    """Format simple string rubrics for the judge prompt.

    Args:
        rubrics: List of rubric strings.

    Returns:
        Formatted string representation.
    """
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubrics))


def _compute_weighted_score(
    rubrics: list[dict[str, Any]],
    per_rubric_scores: dict[str, float],
) -> tuple[float, dict[str, tuple[float, float, bool]]]:
    """Compute weighted score from per-rubric scores.

    Supports negative weights for Pitfall criteria from the RaR paper.
    For pitfall criteria (negative weights):
    - If satisfied (score=1): contributes positively (pitfall was avoided)
    - If unsatisfied (score=0): contributes negatively (pitfall was hit)

    The formula follows the RaR paper:
        r(x, ŷ) = Σ(wⱼ · cⱼ(x, ŷ)) / Σ(|wⱼ|)

    Where wⱼ can be negative for pitfall criteria.

    Args:
        rubrics: List of rubric dicts with weights. Weights can be:
            - Positive (1-5): Essential/Important/Optional criteria
            - Negative (-1 to -2): Pitfall criteria (things to avoid)
        per_rubric_scores: Dict mapping rubric names to scores (0 or 1).

    Returns:
        Tuple of (weighted_score, details) where details maps rubric names
        to (score, weight, is_pitfall) tuples.
    """
    total_weight = 0.0  # Sum of absolute weights for normalization
    weighted_sum = 0.0
    details = {}

    for rubric in rubrics:
        name = rubric.get("name", "")
        weight = float(rubric.get("weight", 1.0))
        is_pitfall = weight < 0

        # Get score, default to 0 if not found
        raw_score = float(per_rubric_scores.get(name, 0))
        raw_score = max(0.0, min(1.0, raw_score))  # Clamp to [0, 1]

        if is_pitfall:
            # For pitfall criteria:
            # - score=1 means the pitfall was avoided (good) -> positive contribution
            # - score=0 means the pitfall was hit (bad) -> negative contribution
            # The weight is negative, so we use its absolute value for normalization
            # and apply the sign based on whether the pitfall was avoided
            abs_weight = abs(weight)
            # Pitfall avoided (1) -> +abs_weight, hit (0) -> -abs_weight
            contribution = abs_weight * (2 * raw_score - 1)  # Maps [0,1] to [-1,1]
            weighted_sum += contribution
            total_weight += abs_weight
        else:
            # For positive criteria, standard weighted sum
            weighted_sum += weight * raw_score
            total_weight += weight

        details[name] = (raw_score, weight, is_pitfall)

    if total_weight == 0:
        return 0.0, details

    # Normalize to [0, 1] range
    # The raw score is in [-total_weight, total_weight], we map to [0, 1]
    normalized_score = (weighted_sum + total_weight) / (2 * total_weight)
    normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp for safety

    return normalized_score, details


def _parse_weighted_response(response_text: str) -> dict[str, float]:
    """Parse per-rubric scores from judge response.

    Tries to extract JSON with scores from the response.

    Args:
        response_text: Raw judge response.

    Returns:
        Dict mapping rubric names to scores.
    """
    import re

    # Try to find JSON in the response
    # Look for patterns like {"scores": {...}, "weighted_score": ...}
    json_pattern = r'\{[^{}]*"scores"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group())
            return data.get("scores", {})
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole response as JSON
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict):
            return data.get("scores", data)
    except json.JSONDecodeError:
        pass

    # Fallback: look for individual rubric scores like "empathy: 1"
    scores = {}
    score_pattern = r'"?(\w+)"?\s*:\s*([01](?:\.\d+)?)'
    for match in re.finditer(score_pattern, response_text):
        name, score = match.groups()
        if name not in ("scores", "weighted_score"):
            scores[name] = float(score)

    return scores


def compute_rubric_reward_panel(
    prompt: str,
    completion: str,
    rubrics: list,
    panel: list[tuple[JudgePanelMember, SimpleJudge]],
    panel_config: JudgePanelConfig,
    log_example: bool = False,
) -> float:
    """Compute reward using a panel of judges.

    Each judge in the panel evaluates the completion independently,
    and scores are aggregated according to the panel configuration.

    Args:
        prompt: The original prompt/task.
        completion: The model's completion to evaluate.
        rubrics: List of rubric strings OR rubric dicts.
        panel: List of (member, judge) tuples.
        panel_config: Configuration for aggregation.
        log_example: Whether to log this example for debugging.

    Returns:
        Aggregated reward in [0.0, 1.0].
    """
    global _REWARD_STATS

    is_weighted = _is_weighted_rubrics(rubrics)
    if is_weighted:
        rubrics_text = _format_weighted_rubrics(rubrics)
    else:
        rubrics_text = _format_simple_rubrics(rubrics)

    input_text = prompt + completion + rubrics_text
    template_overhead = 150 if is_weighted else 100
    input_tokens_per_judge = _estimate_tokens(input_text) + template_overhead

    judge_inputs = [
        {
            "prompt": prompt,
            "response": completion,
            "rubrics": rubrics_text,
        }
    ]

    # Collect scores from all judges
    scores = []
    weights = []
    per_judge_results = []
    total_time_ms = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for member, judge in panel:
        start_time = time.time()
        try:
            outputs = judge.judge(judge_inputs)
            judge_time_ms = (time.time() - start_time) * 1000
            output_tokens = 100 if is_weighted else 30

            score = 0.0
            if outputs and len(outputs) > 0:
                output = outputs[0]
                if is_weighted:
                    raw_response = output.field_values.get("judgment", "")
                    if isinstance(raw_response, str):
                        per_rubric_scores = _parse_weighted_response(raw_response)
                        if per_rubric_scores:
                            score, _ = _compute_weighted_score(
                                rubrics, per_rubric_scores
                            )
                        else:
                            s = output.field_scores.get("judgment")
                            if s is not None:
                                score = max(0.0, min(1.0, float(s)))
                    elif isinstance(raw_response, int | float):
                        score = max(0.0, min(1.0, float(raw_response)))
                else:
                    s = output.field_scores.get("judgment")
                    if s is not None:
                        score = max(0.0, min(1.0, float(s)))
                    else:
                        v = output.field_values.get("judgment")
                        if v is not None:
                            score = max(0.0, min(1.0, float(v)))

            scores.append(score)
            weights.append(member.weight)
            total_time_ms += judge_time_ms
            total_input_tokens += input_tokens_per_judge
            total_output_tokens += output_tokens
            per_judge_results.append((member.name, score, judge_time_ms))

            # Record per-judge stats
            _REWARD_STATS.record_panel_result(
                judge_name=member.name or member.model,
                judge_model=member.model,
                score=score,
                time_ms=judge_time_ms,
                input_tokens=input_tokens_per_judge,
                output_tokens=output_tokens,
                success=True,
            )

        except Exception as e:
            logger.warning(f"Judge {member.name} failed: {e}")
            _REWARD_STATS.record_panel_result(
                judge_name=member.name or member.model,
                judge_model=member.model,
                score=0.0,
                time_ms=0.0,
                input_tokens=input_tokens_per_judge,
                output_tokens=0,
                success=False,
            )

    # Aggregate scores
    if not scores:
        _REWARD_STATS.record_failure(input_tokens=total_input_tokens)
        return 0.0

    reward, agg_details = _aggregate_scores(scores, weights, panel_config.aggregation)

    # Record variance for monitoring judge agreement
    if len(scores) > 1:
        _REWARD_STATS.record_panel_variance(agg_details.get("score_variance", 0.0))

    # Record overall success
    _REWARD_STATS.record_success(
        reward,
        total_time_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )

    # Log if requested or periodically
    if log_example or _REWARD_STATS.should_log():
        _log_panel_example(
            prompt=prompt,
            completion=completion,
            rubrics=rubrics,
            per_judge_results=per_judge_results,
            aggregation=panel_config.aggregation,
            final_reward=reward,
            variance=agg_details.get("score_variance", 0.0),
        )
        logger.info(_REWARD_STATS.get_full_summary())

    return reward


def _log_panel_example(
    prompt: str,
    completion: str,
    rubrics: list,
    per_judge_results: list[tuple[str, float, float]],
    aggregation: AggregationStrategy,
    final_reward: float,
    variance: float,
) -> None:
    """Log a detailed example of panel evaluation."""
    logger.info("=" * 60)
    logger.info("RLVR Panel Evaluation Example:")
    logger.info(f"  Prompt: {prompt}")
    logger.info(f"  Completion: {completion}")
    logger.info(f"  Rubrics: {rubrics}")
    logger.info(f"  Aggregation: {aggregation.value}")
    logger.info("  Per-judge scores:")
    for name, score, time_ms in per_judge_results:
        logger.info(f"    {name}: {score:.3f} ({time_ms:.0f}ms)")
    logger.info(f"  Final reward: {final_reward:.3f} (variance={variance:.4f})")
    logger.info("=" * 60)


def compute_rubric_reward(
    prompt: str,
    completion: str,
    rubrics: list,
    judge: SimpleJudge,
    log_example: bool = False,
) -> float:
    """Compute reward for a single completion based on rubrics.

    Supports both simple string rubrics and weighted rubric objects.

    Args:
        prompt: The original prompt/task.
        completion: The model's completion to evaluate.
        rubrics: List of rubric strings OR rubric dicts with name/description/weight.
        judge: The SimpleJudge instance to use.
        log_example: Whether to log this example for debugging.

    Returns:
        A float reward in [0.0, 1.0]. For weighted rubrics, this is the
        weighted average of per-rubric scores.
    """
    global _REWARD_STATS

    # Detect rubric format and format accordingly
    is_weighted = _is_weighted_rubrics(rubrics)
    if is_weighted:
        rubrics_text = _format_weighted_rubrics(rubrics)
    else:
        rubrics_text = _format_simple_rubrics(rubrics)

    # Estimate input tokens (prompt template + prompt + completion + rubrics)
    # The judge prompt template adds ~100-200 tokens of overhead
    input_text = prompt + completion + rubrics_text
    template_overhead = 150 if is_weighted else 100
    input_tokens = _estimate_tokens(input_text) + template_overhead

    start_time = time.time()

    try:
        # Call the judge
        judge_inputs = [
            {
                "prompt": prompt,
                "response": completion,
                "rubrics": rubrics_text,
            }
        ]
        outputs = judge.judge(judge_inputs)

        judge_time_ms = (time.time() - start_time) * 1000
        reward = 0.0
        per_rubric_details = None

        # Estimate output tokens based on rubric type
        output_tokens = 100 if is_weighted else 30

        if outputs and len(outputs) > 0:
            output = outputs[0]

            if is_weighted:
                # For weighted rubrics, try to parse per-rubric scores
                raw_response = output.field_values.get("judgment", "")
                if isinstance(raw_response, str):
                    per_rubric_scores = _parse_weighted_response(raw_response)
                    if per_rubric_scores:
                        reward, per_rubric_details = _compute_weighted_score(
                            rubrics, per_rubric_scores
                        )
                    else:
                        # Fallback to numeric score if parsing failed
                        score = output.field_scores.get("judgment")
                        if score is not None:
                            reward = max(0.0, min(1.0, float(score)))
                elif isinstance(raw_response, int | float):
                    reward = max(0.0, min(1.0, float(raw_response)))
            else:
                # Simple rubrics: get the overall score
                score = output.field_scores.get("judgment")
                if score is not None:
                    reward = max(0.0, min(1.0, float(score)))
                else:
                    # Try to get from field_values
                    value = output.field_values.get("judgment")
                    if value is not None:
                        reward = max(0.0, min(1.0, float(value)))
                    else:
                        logger.warning(
                            "Judge returned no valid score, defaulting to 0.0"
                        )
                        _REWARD_STATS.record_failure(input_tokens=input_tokens)
                        return 0.0

            # Record success with token counts
            _REWARD_STATS.record_success(
                reward,
                judge_time_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Log example if requested or periodically
            if log_example or _REWARD_STATS.should_log():
                _log_reward_example(
                    prompt=prompt,
                    completion=completion,
                    rubrics=rubrics,
                    reward=reward,
                    judge_time_ms=judge_time_ms,
                    per_rubric_details=per_rubric_details,
                )
                logger.info(_REWARD_STATS.get_full_summary())

            return reward

        logger.warning("Judge returned empty output, defaulting to 0.0")
        _REWARD_STATS.record_failure(input_tokens=input_tokens)
        return 0.0

    except Exception as e:
        logger.warning(f"Error during rubric evaluation: {e}")
        _REWARD_STATS.record_failure(input_tokens=input_tokens)
        return 0.0


def _log_reward_example(
    prompt: str,
    completion: str,
    rubrics: list,
    reward: float,
    judge_time_ms: float,
    per_rubric_details: dict[str, tuple[float, float, bool]] | None = None,
) -> None:
    """Log a detailed example of reward computation."""
    logger.info("=" * 60)
    logger.info("RLVR Reward Example:")
    logger.info(f"  Prompt: {prompt}")
    logger.info(f"  Completion: {completion}")
    logger.info(f"  Rubrics: {rubrics}")

    logger.info(f"  Reward: {reward:.3f}")
    logger.info(f"  Judge time: {judge_time_ms:.0f}ms")

    # Log per-rubric details if available
    if per_rubric_details:
        logger.info("  Per-rubric scores:")
        for name, details in per_rubric_details.items():
            # Handle old format (score, weight) and new (score, weight, is_pitfall)
            if len(details) == 3:
                score, weight, is_pitfall = details
                if is_pitfall:
                    # For pitfall, score=1 means avoided (good), score=0 means hit (bad)
                    status = "✓ avoided" if score > 0.5 else "✗ HIT"
                    logger.info(
                        f"    {status} [PITFALL] {name}: {score:.0f} (weight={weight})"
                    )
                else:
                    status = "✓" if score > 0.5 else "✗"
                    logger.info(f"    {status} {name}: {score:.0f} (weight={weight})")
            else:
                # Old format for backward compatibility
                score, weight = details[0], details[1]
                status = "✓" if score > 0.5 else "✗"
                logger.info(f"    {status} {name}: {score:.0f} (weight={weight})")

    logger.info("=" * 60)


@register("rubric_reward", RegistryType.REWARD_FUNCTION)
def rubric_reward(
    completions: list[list[dict[str, Any]]],
    prompts: list[str] | None = None,
    rubrics: list[list] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Rubric-based reward function for GRPO training.

    This reward function evaluates model completions against a set of rubrics
    using an LLM judge (or panel of judges). It's designed for RLVR (RL from
    Verifiable Rewards) where the reward signal comes from rubric-based evaluation.

    Supports two rubric formats:
    1. Simple: list of strings, e.g., ["Is clear", "Is concise"]
    2. Weighted: list of dicts with name, description, weight, evaluation_type

    Supports judge panels for more robust evaluation:
    - Pass judge_panel config with multiple judges and aggregation strategy
    - Each judge evaluates independently, scores are aggregated
    - Supports mean, weighted_mean, median, min, max, majority_vote

    For more details on custom reward functions used in trl's GRPOTrainer, see:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function

    Args:
        completions: List of completions from the LLM. Each completion is a list
            of message dicts with 'role' and 'content' keys.
        prompts: List of original prompts/tasks (passed by TRL).
        rubrics: List of rubric lists from the dataset. Each rubric list contains
            either string criteria or weighted rubric dicts.
        kwargs: Additional keyword arguments:
            - judge_model: Model to use for single judge (default: gpt-4o-mini)
            - judge_config_path: Path to judge config file
            - judge_panel: Dict or JudgePanelConfig for panel evaluation
            - system_prompt: System prompts from dataset
            - prompt_id: Prompt IDs from dataset
            - metadata: Metadata from dataset

    Returns:
        List of float rewards in [0.0, 1.0] for each completion.
    """
    # TRL passes 'prompts' (plural), dataset may have 'prompt' (singular)
    prompt_list = prompts
    if prompt_list is None:
        prompt_list = kwargs.get("prompt", [])

    # Rubrics come from the dataset
    rubric_list = rubrics
    if rubric_list is None:
        rubric_list = kwargs.get("rubrics", [])

    # System prompts (optional) for context
    system_prompts: list[Any] = kwargs.get("system_prompt", [])

    # Validate we have the required data
    if not prompt_list or not rubric_list:
        prompt_count = len(prompt_list) if prompt_list else 0
        rubric_count = len(rubric_list) if rubric_list else 0
        logger.warning(
            f"Missing prompts or rubrics. prompts={prompt_count}, "
            f"rubrics={rubric_count}. Returning zero rewards."
        )
        return [0.0] * len(completions)

    # Detect if we have weighted rubrics (check first non-empty rubric list)
    has_weighted = False
    for r in rubric_list:
        if r and isinstance(r, list) and len(r) > 0:
            has_weighted = _is_weighted_rubrics(r)
            break

    # Check for judge panel configuration from multiple sources:
    # 1. kwargs (from dataset or programmatic call)
    # 2. Global config (from environment or set_panel_config)
    judge_panel_config = kwargs.get("judge_panel")
    use_panel = False
    panel = None
    panel_config = None

    if judge_panel_config:
        if isinstance(judge_panel_config, dict):
            panel_config = JudgePanelConfig.from_dict(judge_panel_config)
        elif isinstance(judge_panel_config, JudgePanelConfig):
            panel_config = judge_panel_config

    # Fall back to global/environment config
    if panel_config is None:
        panel_config = load_panel_config()

    if panel_config and panel_config.judges:
        use_panel = True
        panel = _get_or_create_panel(panel_config, weighted_rubrics=has_weighted)

    # Get or create single judge if not using panel
    judge = None
    if not use_panel:
        judge_model = kwargs.get("judge_model", "gpt-4o-mini")
        judge_config_path = kwargs.get("judge_config_path")
        judge = _get_or_create_judge(
            judge_model=str(judge_model) if judge_model else "gpt-4o-mini",
            judge_config_path=str(judge_config_path) if judge_config_path else None,
            weighted_rubrics=has_weighted,
        )

    # Extract completion strings - handle different formats from TRL
    completion_strs = _extract_completion_strings(completions)

    # Log batch start
    batch_size = len(completion_strs)
    batch_start_time = time.time()
    rubric_type = "weighted" if has_weighted else "simple"
    if use_panel and panel_config is not None:
        judge_info = f"panel of {len(panel_config.judges)} judges"
    else:
        judge_info = "single judge"
    logger.info(
        f"[RLVR] Processing batch of {batch_size} completions "
        f"({rubric_type} rubrics, {judge_info})..."
    )

    # Compute rewards for each completion
    rewards = []
    for i, (comp, p, r) in enumerate(zip(completion_strs, prompt_list, rubric_list)):
        # Prepend system prompt to the prompt if available
        full_prompt = p
        if system_prompts and i < len(system_prompts) and system_prompts[i]:
            full_prompt = f"[System: {system_prompts[i]}]\n\n{p}"

        # Log first example in each batch for visibility
        log_example = i == 0

        if use_panel and panel is not None and panel_config is not None:
            reward = compute_rubric_reward_panel(
                prompt=full_prompt,
                completion=comp,
                rubrics=r if isinstance(r, list) else [r],
                panel=panel,
                panel_config=panel_config,
                log_example=log_example,
            )
        elif judge is not None:
            reward = compute_rubric_reward(
                prompt=full_prompt,
                completion=comp,
                rubrics=r if isinstance(r, list) else [r],
                judge=judge,
                log_example=log_example,
            )
        else:
            reward = 0.0
        rewards.append(reward)

    # Log batch summary
    batch_time_ms = (time.time() - batch_start_time) * 1000
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    non_zero_rewards = sum(1 for r in rewards if r > 0)
    logger.info(
        f"[RLVR] Batch complete: "
        f"size={batch_size}, "
        f"avg_reward={avg_reward:.3f}, "
        f"non_zero={non_zero_rewards}/{batch_size}, "
        f"time={batch_time_ms:.0f}ms"
    )
    # Log cumulative cost summary
    logger.info(f"[RLVR] {_REWARD_STATS.get_cost_summary()}")

    return rewards
