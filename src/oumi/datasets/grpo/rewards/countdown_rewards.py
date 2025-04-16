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

import random
import re
from typing import Any

from oumi.core.registry import RegistryType, register


def _extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def _validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
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


def _evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Regex that only allows numbers, operators, parentheses and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        # TODO: Remove log
        print(f"Error evaluating equation: {equation_str}. Exception: {e}")
        return None


@register("countdown", RegistryType.REWARD_FUNCTION)
def countdown(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    format_score=0.1,
    score=1.0,
) -> float:
    """Custom reward function for the Countdown task.

    Currently, this function only works with the VERL_PPO trainer.
    Derived from https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py.

    Args:
        data_source: The data source.
        solution_str: The response from the LLM.
        ground_truth: Dictionary containing target number and available numbers
        extra_info: Extra information about the sample.
        format_score: The score for correct format but wrong answer.
        score: The score for the correct answer.

    Returns:
        The reward value.
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = _extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0

    # Validate equation uses correct numbers
    if not _validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = _evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if do_print:
            print("Error evaluating equation")
        return format_score
