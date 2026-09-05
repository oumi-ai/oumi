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
