"""Custom Oumi evaluation function for the Countdown task.

Oumi ships an evaluation function for letter counting (`count_letters`) but not
for Countdown -- Countdown only has a *reward* function, which is used during
training and cannot be used to measure before/after accuracy. This module fills
that gap so a Countdown GRPO run can be evaluated the same way the letter
counting tutorial evaluates its model.

The scoring helpers are reused from `oumi.datasets.grpo.rewards.countdown_rewards`
so training and evaluation stay in agreement about what counts as correct.

Usage -- import this module once so the decorator runs, then reference it by
name in an EvaluationConfig:

    import countdown_eval_task  # noqa: F401  (registers "countdown")

    tasks:
      - evaluation_backend: custom
        task_name: countdown
"""

from typing import Any

from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.datasets.grpo.countdown import CountdownGrpoDataset
from oumi.datasets.grpo.rewards.countdown_rewards import (
    _evaluate_equation,
    _extract_solution,
    _validate_equation,
)
from oumi.utils.logging import logger


@register_evaluation_function("countdown")
def countdown(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
) -> dict[str, Any]:
    """Custom evaluation function registered as `countdown`.

    Scores each response by extracting the equation from its `<answer>` tags,
    checking it uses each provided number exactly once, and evaluating it
    against the target.

    Returns:
        A dict of metrics. `num_correct_answers`, `num_incorrect_answers` and
        `num_invalid_answers` sum to `num_samples`.
    """
    dataset = CountdownGrpoDataset(dataset="d1shs0ap/countdown", split="test")
    num_samples = task_params.num_samples
    if num_samples is None:
        num_samples = len(dataset)

    input_conversations = [dataset.conversation(i) for i in range(num_samples)]
    conversations = inference_engine.infer(input_conversations)
    logger.info(f"Finished inference on {len(conversations)} conversations!")
    if len(conversations) > 0:
        logger.info(f"Sample conversation: {conversations[0]}")

    total = 0  # All examples.
    valid_count = 0  # Responses with an equation that parses and evaluates.
    count = 0  # Responses that are valid and hit the target.
    bad_numbers = 0  # Valid equations that misuse the available numbers.

    for conversation in conversations:
        total += 1
        response = conversation.last_message()
        # Ignore cases where the model didn't respond, or responded with a
        # multimodal message. For now, we focus on text-only responses.
        if not response or not isinstance(response.content, str):
            continue

        equation = _extract_solution(solution_str=response.content)
        if equation is None:
            continue

        result = _evaluate_equation(equation)
        if result is None:
            # Present but not arithmetic -- counts as unparseable, matching the
            # reward function, which returns 0 for this case.
            continue
        valid_count += 1

        ground_truth = conversation.metadata["reward_model"]["ground_truth"]
        if not _validate_equation(equation, ground_truth["numbers"]):
            bad_numbers += 1
            continue

        if abs(result - ground_truth["target"]) < 1e-5:
            count += 1

    return {
        # Accuracy across all examples.
        "accuracy": count / total if total > 0 else 0,
        # Accuracy when only counting examples with a usable equation.
        "properly_extracted_accuracy": count / valid_count if valid_count > 0 else 0,
        "num_samples": num_samples,
        # These three values sum up to num_samples.
        "num_correct_answers": count,
        "num_incorrect_answers": valid_count - count,
        "num_invalid_answers": total - valid_count,
        # Subset of num_incorrect_answers: equation evaluated, but reused or
        # invented numbers rather than using each one exactly once.
        "num_invalid_number_usage": bad_numbers,
    }