import string
from typing import List, Optional

import numpy as np

from lema.utils.logging import logger


def compute_multiple_choice_accuracy(
    label_probs_by_model: List[List[float]],
    label_indices: List[int],
    prompts: Optional[List[str]] = None,
) -> float:
    """Computes the `accuracy` metric when evaluating multiple-choice benchmarks.

    Args:
        label_probs_by_model: List of lists; Shape is (number_answers, number_labels).
          For each answer, the model-generated probability of each label (corresponding
          to each multiple choice option) is provided.
        label_indices: A list of number_answers items, where each item is the index of
          the correct label, i.e. the index of the multiple choice answer that
          correctly answers the question.
        prompts: The list of prompts which generated these answers. If provided, this
          function will execute in debug mode, logging all the prompts and their
          corresponding answers.

    Returns:
        The accuracy (correct / total answers).
    """

    def diplay_prompt_and_answers(prompt, label_index_correct, label_index_by_model):
        answer_letters = list(string.ascii_uppercase)
        answer = answer_letters[label_index_by_model]
        correct_answer = answer_letters[label_index_correct]
        logger.info(f"PROMPT:\n{prompt}<end>\nANSWERS={answer}, {correct_answer}")

    total_answers_count = len(label_indices)
    correct_answers_count = 0
    for answer_index in range(total_answers_count):
        label_index_by_model = np.argmax(label_probs_by_model[answer_index])
        label_index_correct = label_indices[answer_index]
        if label_index_by_model == label_index_correct:
            correct_answers_count += 1
        if prompts:
            diplay_prompt_and_answers(
                prompts[answer_index], label_index_correct, label_index_by_model
            )

    accuracy = float(correct_answers_count) / total_answers_count
    logger.info(
        f"Correct answers: {correct_answers_count} of {total_answers_count}. "
        f"Accuracy is {accuracy:.3f}."
    )
    return accuracy
