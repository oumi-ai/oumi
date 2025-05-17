# Copyright 2025 - Jiayi Pan
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

"""Derived from https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py.

This file was slightly modified to be an Oumi reward registry function.
"""

import re
from typing import Any, Optional

from oumi.core.registry import RegistryType, register


def _extract_solution(solution_str: str) -> Optional[str]:
    """Extracts the equation from the solution string.

    Args:
        solution_str: The response from the LLM.

    Returns:
        The equation from the solution string, or None if not found.
    """
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def _validate_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """Validates that equation only uses available numbers and each number once.

    Args:
        equation_str: The equation to validate.
        available_numbers: The list of available numbers.

    Returns:
        True if the equation uses each available number exactly once, else False.
    """
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def _evaluate_equation(equation_str: str) -> Optional[float]:
    """Safely evaluates the arithmetic equation using eval() with precautions."""
    try:
        # Regex that only allows numbers, operators, parentheses and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


@register("countdown", RegistryType.REWARD_FUNCTION)
def countdown_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    format_score=0.0,
    score=1.0,
) -> float:
    """Custom reward function for the Countdown task.

    Currently, this function only works with the VERL_GRPO trainer.

    Args:
        data_source: The data source.
        solution_str: The response from the LLM.
        ground_truth: Dictionary containing target number and available numbers
        extra_info: Extra information about the sample.
        format_score: The score for correct format but wrong answer.
        score: The score for the correct answer.

    Returns:
        `score` if the equation is valid and correct,
        `format_score` if the answer was parsed properly but the equation is incorrect,
        `0` if the answer was not parsed properly.
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = _extract_solution(solution_str=solution_str)

    if equation is None:
        return 0

    # Validate equation uses correct numbers
    if not _validate_equation(equation, numbers):
        return format_score

    # Evaluate equation
    try:
        result = _evaluate_equation(equation)
        if result is None:
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            return score
        else:
            return format_score
    except Exception:
        return format_score
