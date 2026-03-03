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

"""Core utilities for rubric-based reward functions.

This module provides shared components used by all rubric reward functions:
- RubricJudge: Wrapper around SimpleJudge for rubric evaluation
- Completion extraction utilities
- Statistics tracking
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeParams,
    JudgeResponseFormat,
)
from oumi.core.configs.params.model_params import ModelParams
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger


# Response parsing constants
SCORES_KEY = "scores"
WEIGHTED_SCORE_KEY = "weighted_score"


@dataclass
class RubricStats:
    """Statistics tracker for rubric-based reward computation."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_reward: float = 0.0
    total_judge_time_ms: float = 0.0
    log_interval: int = 100

    def record_success(self, reward: float, judge_time_ms: float) -> None:
        """Record a successful judge call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_reward += reward
        self.total_judge_time_ms += judge_time_ms

    def record_failure(self, judge_time_ms: float = 0.0) -> None:
        """Record a failed judge call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.total_judge_time_ms += judge_time_ms

    @property
    def avg_reward(self) -> float:
        """Average reward across successful calls."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_reward / self.successful_calls

    @property
    def avg_judge_time_ms(self) -> float:
        """Average judge time in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_judge_time_ms / self.total_calls

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    def should_log(self) -> bool:
        """Return True if stats should be logged based on interval."""
        return self.total_calls > 0 and self.total_calls % self.log_interval == 0

    def get_summary(self) -> str:
        """Return a summary string of the statistics."""
        return (
            f"RubricReward: calls={self.total_calls}, "
            f"success={self.success_rate:.1%}, "
            f"avg_reward={self.avg_reward:.3f}, "
            f"avg_time={self.avg_judge_time_ms:.0f}ms"
        )


@dataclass
class JudgeResult:
    """Result from a single judge evaluation."""

    score: float
    time_ms: float
    success: bool
    raw_response: str = ""


class RubricJudge:
    """Wrapper around SimpleJudge for rubric evaluation.

    Provides a simple interface for evaluating completions against rubrics
    using an LLM judge.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        response_format: JudgeResponseFormat = JudgeResponseFormat.RAW,
        max_tokens: int = 500,
        system_instruction: str | None = None,
        prompt_template: str | None = None,
    ):
        """Initialize the RubricJudge.

        Args:
            model: The LLM model to use for judging.
            temperature: Sampling temperature (0.0 for deterministic).
            response_format: Expected response format (RAW for JSON, XML for tags).
            max_tokens: Maximum tokens in judge response.
            system_instruction: Custom system instruction for the judge.
            prompt_template: Custom prompt template for evaluation.
        """
        self.model = model
        self.temperature = temperature
        self._judge: SimpleJudge | None = None
        self._response_format = response_format
        self._max_tokens = max_tokens
        self._system_instruction = system_instruction or self._default_system()
        self._prompt_template = prompt_template or self._default_prompt()

    def _default_system(self) -> str:
        """Default system instruction for rubric evaluation."""
        return (
            "You are an expert evaluator. Assess responses against rubrics "
            "fairly and consistently. Return valid JSON only."
        )

    def _default_prompt(self) -> str:
        """Default prompt template for rubric evaluation."""
        return (
            "Evaluate the response against the rubrics.\n\n"
            "## Task\n{prompt}\n\n"
            "## Response to Evaluate\n{response}\n\n"
            "## Rubrics\n{rubrics}\n\n"
            "For each rubric, determine if it is satisfied (1) or not (0).\n"
            "Return a JSON object with:\n"
            '- "scores": dict mapping rubric names to 0 or 1\n'
            '- "total_score": fraction of rubrics satisfied (0.0 to 1.0)\n\n'
            "Output only valid JSON."
        )

    def _build_config(self) -> JudgeConfig:
        """Build a JudgeConfig for the judge."""
        return JudgeConfig(
            judge_params=JudgeParams(
                prompt_template=self._prompt_template,
                response_format=self._response_format,
                judgment_type=JudgeOutputType.TEXT,
                include_explanation=False,
                system_instruction=self._system_instruction,
            ),
            inference_config=InferenceConfig(
                engine=InferenceEngineType.OPENAI,
                model=ModelParams(model_name=self.model),
                generation=GenerationParams(
                    max_new_tokens=self._max_tokens,
                    temperature=self.temperature,
                ),
            ),
        )

    def _get_judge(self) -> SimpleJudge:
        """Get or create the judge instance."""
        if self._judge is None:
            self._judge = SimpleJudge(self._build_config())
        return self._judge

    def evaluate(
        self,
        prompt: str,
        completion: str,
        rubrics_text: str,
    ) -> JudgeResult:
        """Evaluate a completion against rubrics.

        Args:
            prompt: The original task/prompt.
            completion: The model's completion to evaluate.
            rubrics_text: Formatted rubrics text.

        Returns:
            JudgeResult with score, timing, and success status.
        """
        judge = self._get_judge()
        judge_input = {
            "prompt": prompt,
            "response": completion,
            "rubrics": rubrics_text,
        }

        start_time = time.time()
        try:
            outputs = judge.judge([judge_input])
            time_ms = (time.time() - start_time) * 1000

            if not outputs:
                return JudgeResult(score=0.0, time_ms=time_ms, success=False)

            raw_response = outputs[0].raw_output or ""
            score = self._parse_score(raw_response)

            return JudgeResult(
                score=score,
                time_ms=time_ms,
                success=True,
                raw_response=raw_response,
            )

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            logger.warning(f"Judge evaluation failed: {e}")
            return JudgeResult(score=0.0, time_ms=time_ms, success=False)

    def _parse_score(self, response: str) -> float:
        """Parse the score from judge response."""
        # Try to extract JSON
        json_obj = extract_json_object(response)
        if json_obj:
            try:
                data = json.loads(json_obj)
                if isinstance(data, dict):
                    # Look for total_score or weighted_score
                    for key in ("total_score", WEIGHTED_SCORE_KEY, "score"):
                        if key in data:
                            return clamp(float(data[key]))
                    # Fall back to computing from scores dict
                    scores = data.get(SCORES_KEY, {})
                    if scores and isinstance(scores, dict):
                        values = [float(v) for v in scores.values()]
                        if values:
                            return clamp(sum(values) / len(values))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Fallback: try to extract a float from the response
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            try:
                value = float(match.group(1))
                # If it looks like a count, normalize
                if value > 1.0:
                    return 0.0  # Can't normalize without knowing total
                return clamp(value)
            except ValueError:
                pass

        return 0.0


def extract_json_object(text: str) -> str | None:
    """Extract the first JSON object from text.

    Handles nested braces and strings correctly.

    Args:
        text: Text potentially containing a JSON object.

    Returns:
        The JSON object string, or None if not found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
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
                return text[start : i + 1]

    return None


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))


def extract_completion_strings(completions: list) -> list[str]:
    """Extract string content from completions in various formats.

    Handles:
    - str: Used directly
    - list[dict]: First item's "content" field
    - list[str]: First item
    - dict: "content" field

    Args:
        completions: List of completions in various formats.

    Returns:
        List of completion strings.
    """
    result = []
    for c in completions:
        if isinstance(c, str):
            result.append(c)
        elif isinstance(c, list) and len(c) > 0:
            first = c[0]
            if isinstance(first, dict):
                result.append(first.get("content", str(first)))
            else:
                result.append(str(first))
        elif isinstance(c, dict):
            result.append(c.get("content", str(c)))
        else:
            result.append(str(c))
    return result


def format_string_rubrics(rubrics: list[str]) -> str:
    """Format string rubrics for the judge prompt.

    Args:
        rubrics: List of rubric strings.

    Returns:
        Formatted rubrics text with numbering.
    """
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rubrics, 1))


def validate_inputs(
    completions: list,
    prompts: list,
    rubrics: list,
    function_name: str,
) -> tuple[list[str], list, list, int]:
    """Validate and align inputs for reward functions.

    Args:
        completions: List of completions.
        prompts: List of prompts.
        rubrics: List of rubrics.
        function_name: Name of the calling function for logging.

    Returns:
        Tuple of (completion_strs, prompts, rubrics, count).
    """
    completion_strs = extract_completion_strings(completions)

    if not prompts or not rubrics:
        logger.warning(
            f"[{function_name}] Missing prompts ({len(prompts) if prompts else 0}) "
            f"or rubrics ({len(rubrics) if rubrics else 0}). Returning zeros."
        )
        return [], [], [], 0

    lengths = {
        "completions": len(completion_strs),
        "prompts": len(prompts),
        "rubrics": len(rubrics),
    }
    min_len = min(lengths.values())

    if len(set(lengths.values())) > 1:
        logger.warning(
            f"[{function_name}] Mismatched lengths: {lengths}. Truncating to {min_len}."
        )
        completion_strs = completion_strs[:min_len]
        prompts = prompts[:min_len]
        rubrics = rubrics[:min_len]

    return completion_strs, prompts, rubrics, min_len
