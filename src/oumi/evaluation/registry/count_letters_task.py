import re
from typing import Optional

from oumi.core.configs.evaluation_config import EvaluationConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.evaluation import EvaluationResult
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.datasets.grpo.letter_count import LetterCountGrpoDataset


def _extract_prediction(response: str) -> Optional[int]:
    r"""Extracts the numeric answer from within `\boxed{...}`, or None."""
    regex_result = re.findall(r"\\boxed\{(\d+)\}", response)
    if not regex_result or len(regex_result) != 1:
        return None
    number_str = regex_result[0]
    # Except clause shouldn't trigger because the regex should only find ints.
    try:
        return int(number_str)
    except ValueError:
        return None


@register_evaluation_function("count_letters")
def count_letters(
    task_params: EvaluationTaskParams,
    config: EvaluationConfig,
    inference_engine: BaseInferenceEngine,
):
    """Custom evaluation function registered as `count_letters`."""
    dataset = LetterCountGrpoDataset()
    # dataset = build_dataset("oumi-ai/oumi-letter-count", tokenizer=None, sample_count=10)  # noqa: E501
    # dataset = build_dataset("oumi-ai/berrybench-v0.1.0", tokenizer=None, sample_count=10)  # noqa: E501
    # num_samples = task_params.num_samples
    # dataset = dataset[:num_samples]
    print(dataset)
    print(next(iter(dataset)))
    print(type(dataset))
    print(dataset.conversations())
    conversations = inference_engine.infer(dataset.conversations())

    count = 0
    total = 0
    for i, conversation in enumerate(conversations):
        total += 1
        response = conversation.last_message()
        prediction = _extract_prediction(response.content)  # type: ignore
        if (
            prediction is not None
            and prediction == conversation.metadata["letter_count_integer"]
        ):
            count += 1

    return EvaluationResult(task_result={"accuracy": count / total})
