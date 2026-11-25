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

"""LLM-judge reward for the RaR-Medicine dataset."""

import logging
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any

from oumi.core.registry import RegistryType, register

if TYPE_CHECKING:
    from oumi.judges.simple_judge import SimpleJudge

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_CONFIG = "configs/examples/grpo_verl_medqa/judge.yaml"
_MAX_CONCURRENCY = int(os.environ.get("RAR_JUDGE_MAX_CONCURRENCY", "16"))
_MAX_ATTEMPTS = int(os.environ.get("RAR_JUDGE_MAX_ATTEMPTS", "2"))

_judge: "SimpleJudge | None" = None
_judge_init_lock = threading.Lock()
_judge_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENCY)


def _get_judge(judge_config_path: str) -> "SimpleJudge":
    """Build and return the process-wide judge."""
    global _judge
    with _judge_init_lock:
        if _judge is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY must be set to use the RaR-Medicine reward."
                )
            from oumi.judges.simple_judge import SimpleJudge

            _judge = SimpleJudge(judge_config_path)
    return _judge


def score_rar_medicine_response(
    question: str,
    reference_answer: str,
    response: str,
    judge_config_path: str = _DEFAULT_JUDGE_CONFIG,
) -> float:
    """Return the medical judge's integer score normalized to ``[0, 1]``."""
    judge_config_path = os.environ.get("RAR_JUDGE_CONFIG", judge_config_path)
    judge_input = {
        "question": question,
        "reference_answer": reference_answer,
        "response": response,
    }

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with _judge_semaphore:
                outputs = _get_judge(judge_config_path).judge([judge_input])
            judgment = outputs[0].field_values.get("judgment")
            if judgment is None:
                raise ValueError("The judge returned no parseable judgment.")
            return min(max(float(judgment), 0.0), 10.0) / 10.0
        except Exception as error:
            last_error = error
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(2**attempt + random.random())

    logger.warning(
        "RaR-Medicine judge failed after %d attempts (%s); assigning reward 0.",
        _MAX_ATTEMPTS,
        last_error,
    )
    return 0.0


@register("rar_medicine_verl", RegistryType.REWARD_FUNCTION)
def rar_medicine_verl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    judge_config_path: str = _DEFAULT_JUDGE_CONFIG,
    **kwargs: Any,
) -> float:
    """Score a VERL completion against its RaR-Medicine reference answer."""
    del data_source, kwargs
    if not solution_str or not solution_str.strip():
        return 0.0
    question = str((extra_info or {}).get("question", ""))
    return score_rar_medicine_response(
        question, ground_truth, solution_str, judge_config_path
    )


def _message_text(message: Any) -> str:
    """Return a message's text, which may be a string or typed content blocks."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict)
            )
    return ""


def _completion_text(completion: Any) -> str:
    """Return a TRL completion's text, which arrives as a str or a message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return _message_text(completion[-1])
    return str(completion)


def _question_text(prompt: Any) -> str:
    """Return the question from a TRL prompt, which is a str or a message list."""
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        for message in reversed(prompt):
            text = _message_text(message).strip()
            if text:
                return text
    return ""


def _normalize_judgment(output: Any) -> float | None:
    """Return a judge output's score in ``[0, 1]``, or None if it is unusable.

    A judge that is configured for FLOAT or TEXT rather than INT can put a value
    that does not convert to a number in ``judgment``. Returning None for it
    keeps that row on the retry path instead of raising into the training step.
    """
    field_values = getattr(output, "field_values", None) or {}
    judgment = field_values.get("judgment")
    if judgment is None:
        return None
    try:
        score = float(judgment)
    except (TypeError, ValueError):
        return None
    return min(max(score, 0.0), 10.0) / 10.0


def score_rar_medicine_batch(
    questions: list[str],
    reference_answers: list[str],
    responses: list[str],
    judge_config_path: str = _DEFAULT_JUDGE_CONFIG,
) -> list[float]:
    """Score a whole batch with one judge call, retrying only the rows that fail.

    Uses ``BaseJudge.judge_partial()`` rather than ``judge()``: one row whose
    inference fails must not zero the rest of the group. Successful rows keep
    their scores, and only the failed or unparseable rows are resubmitted.
    """
    judge_config_path = os.environ.get("RAR_JUDGE_CONFIG", judge_config_path)
    if not (len(questions) == len(reference_answers) == len(responses)):
        raise ValueError(
            "questions, reference_answers and responses must have the same length. "
            f"Got {len(questions)}, {len(reference_answers)}, {len(responses)}."
        )

    scores: list[float | None] = [None] * len(responses)
    pending = [
        index
        for index, response in enumerate(responses)
        if response and response.strip()
    ]
    last_error: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        if not pending:
            break
        inputs = [
            {
                "question": questions[index],
                "reference_answer": reference_answers[index],
                "response": responses[index],
            }
            for index in pending
        ]
        try:
            with _judge_semaphore:
                result = _get_judge(judge_config_path).judge_partial(inputs)
        except Exception as error:
            last_error = error
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(2**attempt + random.random())
            continue

        outputs = dict(result.successful)
        still_pending = []
        for position, index in enumerate(pending):
            score = (
                _normalize_judgment(outputs[position]) if position in outputs else None
            )
            if score is None:
                still_pending.append(index)
            else:
                scores[index] = score
        pending = still_pending

    if pending:
        logger.warning(
            "RaR-Medicine judge failed after %d attempts (%s); assigning reward 0 "
            "to %d of %d completions.",
            _MAX_ATTEMPTS,
            last_error or "no parseable judgment",
            len(pending),
            len(responses),
        )

    return [0.0 if score is None else score for score in scores]


def _reference_answers(
    metadata: list[Any] | None, reference_answer: list[Any] | None, expected: int
) -> list[str]:
    """Return one reference answer per completion, from a column or ``metadata``."""
    if reference_answer is not None:
        answers = [str(answer).strip() for answer in reference_answer]
    elif metadata is not None:
        answers = []
        for row in metadata:
            if not isinstance(row, dict) or not row.get("reference_answer"):
                keys = sorted(row) if isinstance(row, dict) else type(row).__name__
                raise ValueError(
                    "rar_medicine_trl needs a reference answer per row. Expected "
                    "`metadata['reference_answer']`, but got metadata row with "
                    f"keys {keys}. Use a dataset whose transform() sets it, such as "
                    "`rar-medicine`, or pass a top-level `reference_answer` column."
                )
            answers.append(str(row["reference_answer"]).strip())
    else:
        raise ValueError(
            "rar_medicine_trl needs a reference answer per row. Expected "
            "`metadata['reference_answer']` or a top-level `reference_answer` "
            "column, but neither was passed. Check that `data.train.datasets` "
            "loads `rar-medicine` and that `remove_unused_columns` is False."
        )

    if len(answers) != expected:
        raise ValueError(
            "rar_medicine_trl got a different number of reference answers than "
            f"completions: {len(answers)} vs {expected}."
        )
    return answers


@register("rar_medicine_trl", RegistryType.REWARD_FUNCTION)
def rar_medicine_trl(
    completions: list[Any],
    prompts: list[Any] | None = None,
    metadata: list[Any] | None = None,
    reference_answer: list[Any] | None = None,
    judge_config_path: str = _DEFAULT_JUDGE_CONFIG,
    **kwargs: Any,
) -> list[float]:
    """Score a TRL_GRPO batch of completions against RaR-Medicine references.

    TRL hands a reward function the whole batch, so this version sends every
    completion to the judge at once instead of one call per completion.

    ``prompts`` and ``metadata`` are the dataset columns TRL passes by name.
    ``RaRMedicineDataset.transform()`` puts the question in ``prompt`` and the
    answer in ``metadata['reference_answer']``, so both are read from there. A
    top-level ``reference_answer`` column is also accepted and takes precedence.
    """
    del kwargs
    if prompts is None:
        raise ValueError(
            "rar_medicine_trl needs the `prompts` column to build the judge "
            "prompt. Check that `remove_unused_columns` is False and that the "
            "reward is called through TRL_GRPO."
        )

    responses = [_completion_text(completion) for completion in completions]
    questions = [_question_text(prompt) for prompt in prompts]
    if len(questions) != len(responses):
        raise ValueError(
            "rar_medicine_trl got a different number of prompts than completions: "
            f"{len(questions)} vs {len(responses)}."
        )
    if any(not question for question in questions):
        raise ValueError(
            "rar_medicine_trl could not read a question from at least one prompt. "
            "Each prompt must be a string or a non-empty message list."
        )

    return score_rar_medicine_batch(
        questions=questions,
        reference_answers=_reference_answers(
            metadata, reference_answer, len(responses)
        ),
        responses=responses,
        judge_config_path=judge_config_path,
    )
