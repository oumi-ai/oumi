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


def _completion_text(completion: Any) -> str:
    """Return a TRL completion's text, which arrives as a str or a message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, str):
            return last
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def _normalize_judgment(output: Any) -> float | None:
    """Return a judge output's score in ``[0, 1]``, or None if it is unusable."""
    judgment = output.field_values.get("judgment")
    if judgment is None:
        return None
    return min(max(float(judgment), 0.0), 10.0) / 10.0


def score_rar_medicine_batch(
    questions: list[str],
    reference_answers: list[str],
    responses: list[str],
    judge_config_path: str = _DEFAULT_JUDGE_CONFIG,
) -> list[float]:
    """Score a whole batch with one judge call, retrying only the rows that fail."""
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
                outputs = _get_judge(judge_config_path).judge(inputs)
            if len(outputs) != len(pending):
                raise ValueError(
                    f"The judge returned {len(outputs)} outputs for "
                    f"{len(pending)} inputs."
                )
        except Exception as error:
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(2**attempt + random.random())
                continue
            logger.warning(
                "RaR-Medicine judge failed after %d attempts (%s); assigning "
                "reward 0 to %d of %d completions.",
                _MAX_ATTEMPTS,
                error,
                len(pending),
                len(responses),
            )
            break

        still_pending = []
        for index, output in zip(pending, outputs):
            score = _normalize_judgment(output)
            if score is None:
                still_pending.append(index)
            else:
                scores[index] = score
        pending = still_pending

    return [0.0 if score is None else score for score in scores]


@register("rar_medicine_trl", RegistryType.REWARD_FUNCTION)
def rar_medicine_trl(
    completions: list[Any],
    reference_answer: list[str],
    question: list[str] | None = None,
    judge_config_path: str = _DEFAULT_JUDGE_CONFIG,
    **kwargs: Any,
) -> list[float]:
    """Score a TRL_GRPO batch of completions against RaR-Medicine references.

    TRL hands a reward function the whole batch, so this version sends every
    completion to the judge at once instead of one call per completion.
    """
    del kwargs
    responses = [_completion_text(completion) for completion in completions]
    answers = list(reference_answer)
    if question is None:
        questions = [""] * len(responses)
    else:
        questions = list(question)
    return score_rar_medicine_batch(
        questions=questions,
        reference_answers=answers,
        responses=responses,
        judge_config_path=judge_config_path,
    )
