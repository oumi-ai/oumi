"""RaR-Medicine GRPO: Variant B meta-rubric judge via oumi SimpleJudge, as a verl reward.

Registers two objects in the oumi registry:
  - dataset  ``anisha2102/RaR-Medicine``: transforms the HF dataset into
    verl-format rows (chat ``prompt``, ``reward_model.ground_truth`` =
    reference answer, ``extra_info.question`` for the judge).
  - reward   ``rar_medicine_verl``: a verl-style reward function that runs one
    oumi ``SimpleJudge`` (gpt-5-mini, config in judge_config.yaml next to this
    file) for a single holistic 0-10 score against the fixed meta rubric
    (Variant B / RaR-Implicit, see ../META_RUBRIC.md), returning ``score / 10``
    in [0, 1].

This module must be importable in two places:
  1. The oumi driver — via ``OUMI_EXTRA_DEPS_FILE`` (which appends this
     directory to ``sys.path`` and imports the module, so the ``@register``
     decorators run before the config is resolved).
  2. verl's reward-loop Ray actors — the oumi trainer points verl at
     ``pkg://rar_medicine_grpo``, so this directory must be on ``PYTHONPATH``
     when launching (Ray actors inherit the driver's environment). See run.sh.

Concurrency: the reward function is sync, and verl 0.7.1's reward loop runs
sync reward functions in each reward worker's thread pool (one thread per
in-flight sample), so judge calls overlap. SimpleJudge is safe to share across
those threads — each ``infer`` call spins up its own event loop and HTTP
session — and a module-level semaphore caps in-flight judge requests per
worker (there are ``reward.num_workers`` workers, 8 by default).

Environment knobs (all optional except OPENAI_API_KEY):
  OPENAI_API_KEY               required for the gpt-5-mini judge.
  RAR_JUDGE_CONFIG             path to a SimpleJudge YAML; default:
                               judge_config.yaml next to this file. Model,
                               generation, and API-retry settings live there.
  RAR_JUDGE_MAX_CONCURRENCY    default 16 (per reward-loop worker).
  RAR_JUDGE_MAX_RETRIES        default 2 (outer retries around the judge call,
                               on top of the engine's own API retries).
"""

import logging
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

# Imported first on purpose: when this module is loaded standalone (verl's
# reward-loop Ray actors import it via `pkg://rar_medicine_grpo`), the
# @register calls below trigger oumi's lazy registry init, which imports
# oumi.datasets while oumi.core.datasets may still be mid-import — a circular
# import. Fully initializing oumi.datasets up front avoids that.
import oumi.datasets  # noqa: F401  # isort: skip
from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import RegistryType, register, register_dataset

if TYPE_CHECKING:
    from oumi.judges.simple_judge import SimpleJudge

logger = logging.getLogger("rar_medicine_grpo")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_DATA_SOURCE = "anisha2102/RaR-Medicine"

# Elicits exactly what the meta rubric rewards: explicit single final answer
# (E1/E2), key reasoning (I1/I2), concision (O1), no hedging (E2/P1).
_SYSTEM_PROMPT = (
    "You are a medical expert answering medical questions. Think through the "
    "problem, then give a clear, concise answer that: (1) explicitly commits to "
    "a single final answer (e.g. 'The final answer is ...'), (2) explains the "
    "key medical reasoning behind it, and (3) avoids irrelevant details and "
    "hedging between alternatives."
)


@register_dataset(_DATA_SOURCE)
class RarMedicineGrpoDataset(BaseExperimentalGrpoDataset):
    """`anisha2102/RaR-Medicine` in verl format for the VERL_GRPO trainer.

    Splits on the hub are named ``train`` / ``val`` / ``test``.
    The per-sample rubric columns are intentionally dropped: the reward is the
    fixed meta rubric applied by one judge against ``reference_answer``.
    """

    default_dataset = _DATA_SOURCE

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Transforms a raw sample into a verl-format dict."""
        question = str(sample["question"]).strip()
        reference = str(sample["reference_answer"]).strip()
        return {
            "prompt": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "data_source": _DATA_SOURCE,
            "ability": "medicine",
            "reward_model": {"style": "rule", "ground_truth": reference},
            # The judge needs the question too; ground_truth only carries the
            # reference answer.
            "extra_info": {
                "split": self.split if self.split else "",
                "question": question,
            },
        }


# ---------------------------------------------------------------------------
# Variant B judge (oumi SimpleJudge; prompt + model live in judge_config.yaml)
# ---------------------------------------------------------------------------

_JUDGE_CONFIG_PATH = os.environ.get(
    "RAR_JUDGE_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_config.yaml"),
)
_JUDGE_MAX_CONCURRENCY = int(os.environ.get("RAR_JUDGE_MAX_CONCURRENCY", "16"))
_JUDGE_MAX_RETRIES = int(os.environ.get("RAR_JUDGE_MAX_RETRIES", "2"))

# Lazy singletons, shared across the reward worker's threads. SimpleJudge is
# safe for that: each infer call runs in its own event loop and HTTP session.
_judge: "SimpleJudge | None" = None
_judge_init_lock = threading.Lock()
_judge_semaphore = threading.BoundedSemaphore(_JUDGE_MAX_CONCURRENCY)


def get_judge() -> "SimpleJudge":
    """Returns the process-wide SimpleJudge, building it on first use."""
    global _judge
    with _judge_init_lock:
        if _judge is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; the rar_medicine_verl reward "
                    "needs it to call the gpt-5-mini judge."
                )
            from oumi.judges.simple_judge import SimpleJudge

            logger.info("Building SimpleJudge from %s", _JUDGE_CONFIG_PATH)
            _judge = SimpleJudge(_JUDGE_CONFIG_PATH)
    return _judge


def judge_response(question: str, reference_answer: str, response: str) -> float:
    """Scores one response with the Variant B judge. Returns a reward in [0, 1].

    The judgment is an integer 0-10 (parsed and schema-enforced by SimpleJudge);
    unparseable or failed judgments fall back to 0.0 after retries. The
    inference engine already retries transient API errors internally
    (remote_params.max_retries); the loop here only re-asks on unparseable
    output or exhausted engine retries.
    """
    judge = get_judge()
    judge_input = {
        "question": question,
        "reference_answer": reference_answer,
        "response": response,
    }

    last_error: Exception | None = None
    for attempt in range(_JUDGE_MAX_RETRIES):
        try:
            with _judge_semaphore:
                outputs = judge.judge([judge_input])
            judgment = outputs[0].field_values.get("judgment")
            if judgment is not None:
                score = min(max(float(judgment), 0.0), 10.0)
                return score / 10.0
            raise ValueError(
                f"Judge returned no parseable judgment: {outputs[0].raw_output[:200]!r}"
            )
        except Exception as e:
            last_error = e
            time.sleep(2**attempt + random.random())

    logger.warning(
        "Judge failed after %d attempts (%s); assigning reward 0.0. Question: %.80r",
        _JUDGE_MAX_RETRIES,
        last_error,
        question,
    )
    return 0.0


# ---------------------------------------------------------------------------
# Registered verl reward
# ---------------------------------------------------------------------------


@register("rar_medicine_verl", RegistryType.REWARD_FUNCTION)
def rar_medicine_verl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """verl-style reward: Variant B meta-rubric SimpleJudge score in [0, 1].

    Sync on purpose: verl 0.7's reward loop dispatches sync reward functions to
    each reward worker's thread pool, so judge calls for a chunk of samples
    still run concurrently (capped by the module's semaphore).

    Args:
        data_source: Dataset name for the sample (unused).
        solution_str: The policy model's decoded response.
        ground_truth: The sample's reference answer.
        extra_info: Carries the original question (set by the dataset class).
        kwargs: Unused extras verl may pass (e.g. reward_router_address).

    Returns:
        judge score / 10, in [0, 1]; 0.0 for empty responses or judge failure.
    """
    if not solution_str or not solution_str.strip():
        return 0.0
    question = (extra_info or {}).get("question", "")
    return judge_response(question, ground_truth, solution_str)
