from typing import Optional

import datasets
from sklearn.metrics import balanced_accuracy_score

from oumi.core.configs import EvaluationConfig, EvaluationTaskParams
from oumi.core.evaluation import EvaluationResult
from oumi.core.inference import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.datasets import TextSftJsonLinesDataset

_prompt_template = """You will be given a premise and a hypothesis.
Determine if the hypothesis is <|supported|> or <|unsupported|> based on the premise.

premise: {premise}

hypothesis: {hypothesis}

You are allowed to think out loud.
Ensure that your final answer is formatted as <|supported|> or <|unsupported|>.
"""


def _load_dataset(num_examples: Optional[int]) -> TextSftJsonLinesDataset:
    """Load the facebook/anli dataset, formatted as a classification problem."""
    anli_dataset = datasets.load_dataset("facebook/anli", split="test_r3")
    evaluation_dataset = []
    for anli_example in anli_dataset:
        assert isinstance(anli_example, dict)
        prompt = _prompt_template.format(
            premise=anli_example["premise"], hypothesis=anli_example["hypothesis"]
        )
        message = {"role": "user", "content": prompt}
        metadata = {"label": _convert_label_to_binary(anli_example["label"])}
        conversation = {
            "conversation_id": anli_example["uid"],
            "messages": [message],
            "metadata": metadata,
        }
        evaluation_dataset.append(conversation)

    # Limit the number of examples, if requested.
    if num_examples is not None:
        evaluation_dataset = evaluation_dataset[:num_examples]
    return TextSftJsonLinesDataset(data=evaluation_dataset)


def _convert_label_to_binary(label: int) -> int:
    """Clamps labels to the set [0,1]."""
    if label == 0:
        return 0  # supported
    elif label in [1, 2]:
        return 1  # unsupported
    else:
        raise ValueError(f"Invalid label: {label}")


def _extract_prediction(response: str) -> int:
    """Converts a response to a label: [0, 1], or -1 if inconclusive."""
    is_unsupported = "<|unsupported|>" in response
    is_supported = "<|supported|>" in response
    if is_unsupported == is_supported:
        return -1
    return 0 if is_supported else 1


@register_evaluation_function("classifier_benchmark")
def classifier_benchmark(
    task_params: EvaluationTaskParams,
    config: EvaluationConfig,
    inference_engine: BaseInferenceEngine,
):
    """Custom evaluation function registered as `my_custom_evaluation`."""
    num_examples: Optional[int] = task_params.eval_kwargs.get("num_examples", None)
    dataset = _load_dataset(num_examples)
    # Run inference to generate the model responses.
    conversations = inference_engine.infer(dataset.conversations())

    y_true, y_pred = [], []
    for conversation in conversations:
        # Extract the assistant's (LLM's) response from the conversation.
        response = conversation.last_message()
        assert response and isinstance(response.content, str)
        # Extract the prediction from the response.
        prediction = _extract_prediction(response.content)
        if prediction != -1:
            y_pred.append(prediction)
            y_true.append(conversation.metadata["label"])

    # Compute any relevant metrics (such as Balanced Accuracy).
    bacc = balanced_accuracy_score(y_true, y_pred)

    return EvaluationResult(
        task_result={
            "results": {
                "classifier_benchmark": {"alias": "classifier_benchmark", "bacc": bacc}
            }
        }
    )
