import re
from typing import Any, Optional

from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.datasets.grpo.letter_count import LetterCountGrpoDataset
from oumi.utils.logging import logger


def _extract_prediction(response: str) -> Optional[int]:
    r"""Returns the numeric answer extracted from `\boxed{...}`, or returns None."""
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
    inference_engine: BaseInferenceEngine,
) -> dict[str, Any]:
    """Custom evaluation function registered as `count_letters`."""
    dataset = LetterCountGrpoDataset(split="test")
    # TODO: OPE-1155: Add support for using Oumi dataset code to create the dataset.
    # dataset = build_dataset("oumi-ai/oumi-letter-count", tokenizer=None, sample_count=10)  # noqa: E501
    # dataset = build_dataset("oumi-ai/berrybench-v0.1.0", tokenizer=None, sample_count=10)  # noqa: E501
    num_samples = task_params.num_samples
    if num_samples is None:
        num_samples = len(dataset)
    input_conversations = [dataset.conversation(i) for i in range(num_samples)]
    conversations = inference_engine.infer(input_conversations)
    logger.info(f"Finished inference on {len(conversations)} conversations!")
    if len(conversations) > 0:
        logger.info(f"Sample conversation: {conversations[0]}")

    count = 0
    total = 0
    for i, conversation in enumerate(conversations):
        total += 1
        response = conversation.last_message()
        if not response or not isinstance(response.content, str):
            continue
        prediction = _extract_prediction(response.content)
        if (
            prediction is not None
            and prediction == conversation.metadata["letter_count_integer"]
        ):
            count += 1

    return {"accuracy": count / total}
