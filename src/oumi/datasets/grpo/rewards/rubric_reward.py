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
import re
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

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

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    MAJORITY_VOTE = "majority_vote"


@dataclass
class JudgePanelMember:
    """Configuration for a single judge in a panel."""

    model: str = "gpt-4o-mini"
    weight: float = 1.0
    name: str | None = None
    temperature: float = 0.0
    role: str | None = None

    def __post_init__(self) -> None:
        """Set default name to model if not provided."""
        if self.name is None:
            self.name = self.model


@dataclass
class JudgePanelConfig:
    """Configuration for a panel of judges."""

    judges: list[JudgePanelMember] = field(default_factory=list)
    aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "JudgePanelConfig":
        """Create a JudgePanelConfig from a dictionary."""
        judges = []
        for j in config.get("judges", []):
            if isinstance(j, str):
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
        )


@dataclass
class RubricRewardStats:
    """Statistics tracker for rubric-based reward computation."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_reward: float = 0.0
    total_judge_time_ms: float = 0.0
    log_interval: int = 100
    panel_variance_history: list[float] = field(default_factory=list)

    def record_success(self, reward: float, judge_time_ms: float) -> None:
        """Record a successful judge call with its reward and timing."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_reward += reward
        self.total_judge_time_ms += judge_time_ms

    def record_failure(self) -> None:
        """Record a failed judge call."""
        self.total_calls += 1
        self.failed_calls += 1

    def record_panel_variance(self, variance: float) -> None:
        """Record variance from a panel evaluation."""
        self.panel_variance_history.append(variance)

    @property
    def avg_reward(self) -> float:
        """Return average reward across successful calls."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_reward / self.successful_calls

    @property
    def avg_judge_time_ms(self) -> float:
        """Return average judge time in milliseconds."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_judge_time_ms / self.successful_calls

    @property
    def success_rate(self) -> float:
        """Return success rate as a fraction."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    def should_log(self) -> bool:
        """Return True if stats should be logged based on log interval."""
        return self.total_calls > 0 and self.total_calls % self.log_interval == 0

    def get_summary(self) -> str:
        """Return a summary string of the statistics."""
        return (
            f"RubricReward: calls={self.total_calls}, "
            f"success={self.success_rate:.1%}, "
            f"avg_reward={self.avg_reward:.3f}, "
            f"avg_time={self.avg_judge_time_ms:.0f}ms"
        )


class RubricRewardEvaluator:
    """Evaluator for rubric-based rewards using LLM judges.

    Encapsulates judge caching, panel configuration, and statistics tracking.
    """

    def __init__(self, panel_config: JudgePanelConfig | None = None):
        """Initialize the evaluator with optional panel configuration."""
        self._judges: dict[str, SimpleJudge] = {}
        self._stats = RubricRewardStats()
        self._panel_config = panel_config

    @property
    def stats(self) -> RubricRewardStats:
        """Return the current statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset the statistics to initial state."""
        self._stats = RubricRewardStats()

    def set_panel_config(self, config: JudgePanelConfig | None) -> None:
        """Set the panel configuration for multi-judge evaluation."""
        self._panel_config = config

    def _build_judge_config(
        self,
        model: str,
        temperature: float,
        weighted_rubrics: bool,
        group_rubrics: bool,
        role: str | None = None,
    ) -> JudgeConfig:
        """Build a JudgeConfig for rubric evaluation."""
        from oumi.core.configs.inference_config import InferenceConfig
        from oumi.core.configs.inference_engine_type import InferenceEngineType
        from oumi.core.configs.params.generation_params import GenerationParams
        from oumi.core.configs.params.model_params import ModelParams

        role_suffix = f"\n\nYour specific focus is: {role}" if role else ""

        if group_rubrics:
            if weighted_rubrics:
                prompt_template = (
                    "You are evaluating a response based on specific weighted "
                    "rubrics.\n\n"
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
                    f"sum(weight*score)/sum(weights). Return valid JSON only."
                    f"{role_suffix}"
                )
                response_format = JudgeResponseFormat.RAW
                judgment_type = JudgeOutputType.TEXT
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
                response_format = JudgeResponseFormat.XML
                judgment_type = JudgeOutputType.FLOAT
                max_tokens = 100
        else:
            prompt_template = (
                "You are evaluating a response based on a single rubric.\n\n"
                "## Task\n{prompt}\n\n"
                "## Response to Evaluate\n{response}\n\n"
                "## Rubric\n{rubrics}\n\n"
                "Determine if the response satisfies the rubric.\n"
                "Return 1 if satisfied, 0 if not."
            )
            system_instruction = (
                "You are an expert evaluator. Assess responses against a rubric "
                "fairly and consistently. Return 1 if satisfied, 0 if not."
                f"{role_suffix}"
            )
            response_format = JudgeResponseFormat.XML
            judgment_type = JudgeOutputType.FLOAT
            max_tokens = 100

        return JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=prompt_template,
                response_format=response_format,
                judgment_type=judgment_type,
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

    def _get_judge(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        weighted_rubrics: bool = False,
        group_rubrics: bool = False,
        role: str | None = None,
    ) -> SimpleJudge:
        """Get or create a cached judge instance."""
        cache_key = f"{model}_{temperature}_{weighted_rubrics}_{group_rubrics}_{role}"
        if cache_key not in self._judges:
            config = self._build_judge_config(
                model, temperature, weighted_rubrics, group_rubrics, role
            )
            self._judges[cache_key] = SimpleJudge(config)
        return self._judges[cache_key]

    def _is_weighted_rubrics(self, rubrics: list) -> bool:
        """Check if rubrics are in the weighted format."""
        if not rubrics:
            return False
        are_dicts = [isinstance(rubric, dict) for rubric in rubrics]
        if any(are_dicts) and not all(are_dicts):
            raise ValueError(
                "Rubrics must be all dicts (weighted) or all strings. "
                f"Actual types: {[type(r).__name__ for r in rubrics]}"
            )
        if all(are_dicts):
            for i, rubric in enumerate(rubrics, 1):
                if "description" not in rubric and "weight" not in rubric:
                    raise ValueError(
                        "Weighted rubrics must include 'description' or 'weight'. "
                        f"Missing in rubric_{i}: {rubric}"
                    )
            return True
        return False

    def _format_rubrics(self, rubrics: list, weighted: bool) -> str:
        """Format rubrics for the judge prompt."""
        if weighted:
            lines = []
            for i, rubric in enumerate(rubrics, 1):
                name = rubric.get("name") or f"rubric_{i}"
                description = rubric.get("description", "")
                weight = rubric.get("weight", 1.0)
                eval_type = rubric.get("evaluation_type", "binary")
                lines.append(
                    f"{i}. [{name}] (weight={weight}, type={eval_type}): {description}"
                )
            return "\n".join(lines)
        else:
            return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubrics))

    def _extract_json_object(self, response_text: str) -> str | None:
        """Extract the first JSON object found in the response text."""
        start = response_text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(response_text)):
            ch = response_text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return response_text[start : i + 1]
        return None

    def _coerce_score_map(self, data: Any) -> dict[str, float]:
        """Coerce a mapping of rubric scores into float values."""
        if not isinstance(data, dict):
            return {}
        scores: dict[str, float] = {}
        for key, value in data.items():
            try:
                scores[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return scores

    def _parse_weighted_response(
        self, response_text: str
    ) -> tuple[dict[str, float], float | None]:
        """Parse per-rubric scores and weighted score from judge response."""
        scores: dict[str, float] = {}
        weighted_score: float | None = None

        json_text = self._extract_json_object(response_text)
        if json_text:
            try:
                data = json.loads(json_text)
                if isinstance(data, dict):
                    scores = self._coerce_score_map(data.get("scores", data))
                    try:
                        if "weighted_score" in data:
                            weighted_score = float(data["weighted_score"])
                    except (TypeError, ValueError):
                        weighted_score = None
                    return scores, weighted_score
            except json.JSONDecodeError:
                pass

        try:
            data = json.loads(response_text.strip())
            if isinstance(data, dict):
                scores = self._coerce_score_map(data.get("scores", data))
                try:
                    if "weighted_score" in data:
                        weighted_score = float(data["weighted_score"])
                except (TypeError, ValueError):
                    weighted_score = None
                return scores, weighted_score
        except json.JSONDecodeError:
            pass

        weighted_match = re.search(
            r'"?weighted_score"?\s*:\s*([+-]?\d+(?:\.\d+)?)',
            response_text,
        )
        if weighted_match:
            try:
                weighted_score = float(weighted_match.group(1))
            except ValueError:
                weighted_score = None

        score_pattern = r'"([^"]+)"\s*:\s*([+-]?\d+(?:\.\d+)?)'
        for match in re.finditer(score_pattern, response_text):
            name, score = match.groups()
            if name not in ("scores", "weighted_score"):
                scores[name] = float(score)

        if not scores:
            score_pattern = r"(?:^|[,{])\s*([A-Za-z0-9_\-]+)\s*:\s*([+-]?\d+(?:\.\d+)?)"
            for match in re.finditer(score_pattern, response_text):
                name, score = match.groups()
                if name not in ("scores", "weighted_score"):
                    scores[name] = float(score)

        return scores, weighted_score

    def _compute_weighted_score(
        self,
        rubrics: list[dict[str, Any]],
        per_rubric_scores: dict[str, float],
    ) -> float:
        """Compute weighted score from per-rubric scores.

        Supports negative weights for Pitfall criteria from the RaR paper.
        If any weights are negative, scores are normalized to [0, 1] after
        applying pitfall semantics. Otherwise uses a standard weighted mean.
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

            raw_score = float(per_rubric_scores.get(name, 0))
            raw_score = max(0.0, min(1.0, raw_score))

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
            return max(0.0, min(1.0, weighted_sum / total_weight))

        # Normalize to [0, 1] for pitfall-style scoring.
        normalized = (weighted_sum + total_weight) / (2 * total_weight)
        return max(0.0, min(1.0, normalized))

    def _aggregate_scores(
        self,
        scores: list[float],
        weights: list[float],
        strategy: AggregationStrategy,
    ) -> tuple[float, float]:
        """Aggregate scores from multiple judges. Returns (score, variance)."""
        if not scores:
            return 0.0, 0.0

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
            votes = [1 if s >= 0.5 else 0 for s in scores]
            result = 1.0 if sum(votes) > len(votes) / 2 else 0.0
        else:
            result = sum(scores) / len(scores)

        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        return result, variance

    def _evaluate_single_judge(
        self,
        prompt: str,
        completion: str,
        rubrics: list,
        judge: SimpleJudge,
        is_weighted: bool,
        group_rubrics: bool,
    ) -> tuple[float, float]:
        """Core evaluation logic for a single judge. Returns (score, time_ms)."""
        rubrics_text = self._format_rubrics(rubrics, is_weighted)
        judge_inputs = [
            {"prompt": prompt, "response": completion, "rubrics": rubrics_text}
        ]

        start_time = time.time()
        try:
            outputs = judge.judge(judge_inputs)
            judge_time_ms = (time.time() - start_time) * 1000

            if not outputs or len(outputs) == 0:
                return 0.0, judge_time_ms

            output = outputs[0]
            score = 0.0

            if is_weighted:
                if group_rubrics:
                    raw_response = output.raw_output or ""
                    per_rubric_scores, weighted_score = self._parse_weighted_response(
                        raw_response
                    )
                    if weighted_score is not None:
                        score = float(weighted_score)
                    elif per_rubric_scores:
                        missing = []
                        for i, rubric in enumerate(rubrics, 1):
                            name = rubric.get("name") or f"rubric_{i}"
                            if name not in per_rubric_scores:
                                missing.append(name)
                        if missing:
                            logger.warning(
                                "Weighted rubric scores missing keys: "
                                f"{missing}. Response: {raw_response}"
                            )
                        score = self._compute_weighted_score(rubrics, per_rubric_scores)
                    else:
                        try:
                            score = float(raw_response.strip())
                        except (TypeError, ValueError):
                            score = 0.0
                    return max(0.0, min(1.0, float(score))), judge_time_ms

                raw_response = output.field_values.get("judgment", "")
                if isinstance(raw_response, str):
                    per_rubric_scores, weighted_score = self._parse_weighted_response(
                        raw_response
                    )
                    if weighted_score is not None:
                        score = float(weighted_score)
                    elif per_rubric_scores:
                        score = self._compute_weighted_score(rubrics, per_rubric_scores)
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

            return score, judge_time_ms

        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            return 0.0, (time.time() - start_time) * 1000

    def evaluate(
        self,
        prompt: str,
        completion: str,
        rubrics: list,
        judge_model: str = "gpt-4o-mini",
        group_rubrics: bool = False,
    ) -> float:
        """Evaluate a single completion against rubrics.

        Uses panel if configured, otherwise single judge.
        """
        is_weighted = self._is_weighted_rubrics(rubrics)

        if group_rubrics:
            if self._panel_config and self._panel_config.judges:
                return self._evaluate_with_panel(
                    prompt, completion, rubrics, is_weighted, group_rubrics=True
                )
            return self._evaluate_with_single_judge(
                prompt,
                completion,
                rubrics,
                judge_model,
                is_weighted,
                group_rubrics=True,
            )

        if self._panel_config and self._panel_config.judges:
            return self._evaluate_per_rubric_with_panel(
                prompt, completion, rubrics, is_weighted
            )
        return self._evaluate_per_rubric_with_single_judge(
            prompt, completion, rubrics, judge_model, is_weighted
        )

    def _evaluate_with_single_judge(
        self,
        prompt: str,
        completion: str,
        rubrics: list,
        judge_model: str,
        is_weighted: bool,
        group_rubrics: bool,
    ) -> float:
        """Evaluate using a single judge."""
        judge = self._get_judge(
            judge_model, 0.0, is_weighted, group_rubrics=group_rubrics
        )
        score, judge_time_ms = self._evaluate_single_judge(
            prompt, completion, rubrics, judge, is_weighted, group_rubrics
        )

        if score > 0:
            self._stats.record_success(score, judge_time_ms)
        else:
            self._stats.record_failure()

        if self._stats.should_log():
            logger.info(self._stats.get_summary())

        return score

    def _evaluate_with_panel(
        self,
        prompt: str,
        completion: str,
        rubrics: list,
        is_weighted: bool,
        group_rubrics: bool,
    ) -> float:
        """Evaluate using a panel of judges."""
        assert self._panel_config is not None  # Guaranteed by caller

        scores = []
        weights = []
        total_time_ms = 0.0

        for member in self._panel_config.judges:
            judge = self._get_judge(
                member.model,
                member.temperature,
                is_weighted,
                group_rubrics=group_rubrics,
                role=member.role,
            )
            score, judge_time_ms = self._evaluate_single_judge(
                prompt, completion, rubrics, judge, is_weighted, group_rubrics
            )
            scores.append(score)
            weights.append(member.weight)
            total_time_ms += judge_time_ms

        if not scores:
            self._stats.record_failure()
            return 0.0

        reward, variance = self._aggregate_scores(
            scores, weights, self._panel_config.aggregation
        )

        if len(scores) > 1:
            self._stats.record_panel_variance(variance)

        self._stats.record_success(reward, total_time_ms)

        if self._stats.should_log():
            logger.info(self._stats.get_summary())

        return reward

    def _evaluate_per_rubric_with_single_judge(
        self,
        prompt: str,
        completion: str,
        rubrics: list,
        judge_model: str,
        is_weighted: bool,
    ) -> float:
        """Evaluate by scoring each rubric independently with one judge."""
        if not rubrics:
            self._stats.record_failure()
            return 0.0

        judge = self._get_judge(judge_model, 0.0, is_weighted, group_rubrics=False)
        scores = []
        per_rubric_scores = {}
        total_time_ms = 0.0

        for i, rubric in enumerate(rubrics, 1):
            score, judge_time_ms = self._evaluate_single_judge(
                prompt, completion, [rubric], judge, is_weighted, group_rubrics=False
            )
            scores.append(score)
            total_time_ms += judge_time_ms

            if is_weighted:
                if isinstance(rubric, dict):
                    name = rubric.get("name") or f"rubric_{i}"
                else:
                    name = f"rubric_{i}"
                per_rubric_scores[name] = score

        if not scores:
            self._stats.record_failure()
            return 0.0

        if is_weighted:
            reward = self._compute_weighted_score(rubrics, per_rubric_scores)
        else:
            reward = sum(scores) / len(scores)

        if reward > 0:
            self._stats.record_success(reward, total_time_ms)
        else:
            self._stats.record_failure()

        if self._stats.should_log():
            logger.info(self._stats.get_summary())

        return reward

    def _evaluate_per_rubric_with_panel(
        self,
        prompt: str,
        completion: str,
        rubrics: list,
        is_weighted: bool,
    ) -> float:
        """Evaluate by scoring each rubric independently with a judge panel."""
        assert self._panel_config is not None  # Guaranteed by caller

        if not rubrics:
            self._stats.record_failure()
            return 0.0

        scores = []
        per_rubric_scores = {}
        total_time_ms = 0.0

        for i, rubric in enumerate(rubrics, 1):
            panel_scores = []
            panel_weights = []
            rubric_time_ms = 0.0

            for member in self._panel_config.judges:
                judge = self._get_judge(
                    member.model,
                    member.temperature,
                    is_weighted,
                    group_rubrics=False,
                    role=member.role,
                )
                score, judge_time_ms = self._evaluate_single_judge(
                    prompt,
                    completion,
                    [rubric],
                    judge,
                    is_weighted,
                    group_rubrics=False,
                )
                panel_scores.append(score)
                panel_weights.append(member.weight)
                rubric_time_ms += judge_time_ms

            if not panel_scores:
                continue

            rubric_score, variance = self._aggregate_scores(
                panel_scores, panel_weights, self._panel_config.aggregation
            )
            if len(panel_scores) > 1:
                self._stats.record_panel_variance(variance)

            scores.append(rubric_score)
            total_time_ms += rubric_time_ms

            if is_weighted:
                if isinstance(rubric, dict):
                    name = rubric.get("name") or f"rubric_{i}"
                else:
                    name = f"rubric_{i}"
                per_rubric_scores[name] = rubric_score

        if not scores:
            self._stats.record_failure()
            return 0.0

        if is_weighted:
            reward = self._compute_weighted_score(rubrics, per_rubric_scores)
        else:
            reward = sum(scores) / len(scores)

        if reward > 0:
            self._stats.record_success(reward, total_time_ms)
        else:
            self._stats.record_failure()

        if self._stats.should_log():
            logger.info(self._stats.get_summary())

        return reward


# Module-level evaluator instance (lazily used by the reward function)
_EVALUATOR: RubricRewardEvaluator | None = None


def get_evaluator() -> RubricRewardEvaluator:
    """Get or create the module-level evaluator instance."""
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = RubricRewardEvaluator()
    return _EVALUATOR


def get_rubric_reward_stats() -> RubricRewardStats:
    """Get the current rubric reward statistics."""
    return get_evaluator().stats


def reset_rubric_reward_stats() -> None:
    """Reset the rubric reward statistics."""
    get_evaluator().reset_stats()


def _extract_completion_strings(completions: list) -> list[str]:
    """Extract string content from completions in various formats."""
    completion_strs = []
    for c in completions:
        if isinstance(c, str):
            completion_strs.append(c)
        elif isinstance(c, list) and len(c) > 0:
            if isinstance(c[0], dict):
                completion_strs.append(c[0].get("content", str(c[0])))
            else:
                completion_strs.append(str(c[0]))
        elif isinstance(c, dict):
            completion_strs.append(c.get("content", str(c)))
        else:
            completion_strs.append(str(c))
    return completion_strs


@register("rubric_reward", RegistryType.REWARD_FUNCTION)
def rubric_reward(
    completions: list[list[dict[str, Any]]],
    prompts: list[str] | None = None,
    rubrics: list[list] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Rubric-based reward function for GRPO training.

    Evaluates model completions against rubrics using an LLM judge.
    Supports both simple string rubrics and weighted rubric dicts.
    Optionally uses a panel of judges for more robust evaluation.

    Args:
        completions: List of completions from the LLM.
        prompts: List of original prompts/tasks.
        rubrics: List of rubric lists from the dataset.
        kwargs: Additional arguments:
            - judge_model: Model to use (default: gpt-4o-mini)
            - judge_panel: Dict or JudgePanelConfig for panel evaluation
            - judge_panel_path: Path to YAML file with panel configuration
            - group_rubrics: If True, a single judge processes all rubrics
            - system_prompt: System prompts from dataset

    Returns:
        List of float rewards in [0.0, 1.0] for each completion.
    """
    prompt_list = prompts or kwargs.get("prompt", [])
    rubric_list = rubrics or kwargs.get("rubrics", [])
    system_prompts: list[Any] = kwargs.get("system_prompt", [])

    if not prompt_list or not rubric_list:
        logger.warning(
            f"Missing prompts ({len(prompt_list) if prompt_list else 0}) or "
            f"rubrics ({len(rubric_list) if rubric_list else 0}). Returning zeros."
        )
        return [0.0] * len(completions)

    # Get evaluator and configure panel if provided
    evaluator = get_evaluator()

    judge_panel_config = kwargs.get("judge_panel")
    judge_panel_path = kwargs.get("judge_panel_path")

    # Load from file path if provided
    if judge_panel_path and not judge_panel_config:
        with open(judge_panel_path) as f:
            judge_panel_config = yaml.safe_load(f)

    if judge_panel_config:
        if isinstance(judge_panel_config, dict):
            evaluator.set_panel_config(JudgePanelConfig.from_dict(judge_panel_config))
        elif isinstance(judge_panel_config, JudgePanelConfig):
            evaluator.set_panel_config(judge_panel_config)
    else:
        evaluator.set_panel_config(None)

    judge_model = str(kwargs.get("judge_model", "gpt-4o-mini"))
    group_rubrics = kwargs.get("group_rubrics", False)
    if isinstance(group_rubrics, str):
        group_rubrics = group_rubrics.strip().lower() in ("1", "true", "yes", "y")
    else:
        group_rubrics = bool(group_rubrics)
    completion_strs = _extract_completion_strings(completions)

    lengths = {
        "completions": len(completion_strs),
        "prompts": len(prompt_list),
        "rubrics": len(rubric_list),
    }
    min_len = min(lengths.values()) if lengths else 0
    if len(set(lengths.values())) > 1:
        logger.warning(f"Mismatched input lengths: {lengths}. Truncating to {min_len}.")
        completion_strs = completion_strs[:min_len]
        prompt_list = prompt_list[:min_len]
        rubric_list = rubric_list[:min_len]
        if system_prompts:
            system_prompts = system_prompts[:min_len]

    # Compute rewards
    rewards = []
    for i, (comp, p, r) in enumerate(zip(completion_strs, prompt_list, rubric_list)):
        full_prompt = p
        if system_prompts and i < len(system_prompts) and system_prompts[i]:
            full_prompt = f"[System: {system_prompts[i]}]\n\n{p}"

        rubric_list_for_prompt = r if isinstance(r, list) else [r]
        reward = evaluator.evaluate(
            full_prompt,
            comp,
            rubric_list_for_prompt,
            judge_model,
            group_rubrics=group_rubrics,
        )
        rewards.append(reward)

    # Log batch summary
    batch_size = len(completion_strs)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info(f"[RLVR] Batch: size={batch_size}, avg_reward={avg_reward:.3f}")

    return rewards
