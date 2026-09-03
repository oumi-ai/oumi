# Copyright 2026 - Oumi
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

"""RaR-Medicine evaluation with the Variant B meta-rubric judge.

One holistic 0-10 judge call per sample, conditioned on the sample's reference
answer -- the same judge config and the same input mapping the GRPO reward used
during training (experiments/rar_medicine/variant_b/rar_medicine_grpo.py). The
headline number is therefore the training reward measured on held-out prompts,
not an independent benchmark; see `healthbench_task` for that.

The harness scores an existing file of completed conversations (the output of
`oumi infer`) when `responses_path` is given, and otherwise generates responses
from a prompt-only conversation file with the configured inference engine.
Judgments are cached per conversation so interrupted runs resume.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonlines

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation, Role
from oumi.evaluation.registry.healthbench_common import (
    bootstrap_std,
    clipped_mean,
    load_completed_conversations,
    save_conversations,
    save_jsonlines,
    write_json,
)
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger

__all__ = [
    "CORRECT_CONCLUSION_THRESHOLD",
    "aggregate_rar_scores",
    "build_rar_judge_inputs",
    "is_valid_judgment",
    "rar_medicine",
    "select_conversations",
    "validate_rar_conversations",
]

DEFAULT_JUDGE_CONFIG = "experiments/rar_medicine/variant_b/judge_config.yaml"

JUDGMENT_MIN = 0
JUDGMENT_MAX = 10

# The rubric caps any response whose final conclusion disagrees with the
# reference at 3, so a judgment of 4 or more is the judge's "final answer is
# correct" line. It is a property of the rubric, not an independent accuracy
# measurement.
CORRECT_CONCLUSION_THRESHOLD = 4


def is_valid_judgment(judgment: Any) -> bool:
    """Returns True if `judgment` is an integer on the judge's 0-10 scale."""
    return (
        isinstance(judgment, int)
        and not isinstance(judgment, bool)
        and JUDGMENT_MIN <= judgment <= JUDGMENT_MAX
    )


def select_conversations(
    conversations: Sequence[Conversation], num_samples: int | None, seed: int
) -> list[Conversation]:
    """Returns `num_samples` conversations, drawn deterministically by id.

    Conversations are sorted by `conversation_id` before sampling, so two
    response files over the same prompts yield the same subset for a paired
    pilot regardless of their row order.
    """
    ordered = sorted(conversations, key=lambda c: str(c.conversation_id))
    if num_samples is None or num_samples == len(ordered):
        return ordered
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if num_samples > len(ordered):
        raise ValueError(
            f"Requested {num_samples} samples, but only {len(ordered)} are available"
        )
    selected = random.Random(seed).sample(ordered, num_samples)
    return sorted(selected, key=lambda c: str(c.conversation_id))


def validate_rar_conversations(conversations: Sequence[Conversation]) -> None:
    """Checks that every conversation carries what the judge needs."""
    if not conversations:
        raise ValueError("No conversations to evaluate")
    seen_ids: set[str] = set()
    for index, conversation in enumerate(conversations):
        conversation_id = conversation.conversation_id
        if not conversation_id:
            raise ValueError(f"Conversation at row {index} has no conversation_id")
        if conversation_id in seen_ids:
            raise ValueError(
                f"Duplicate conversation_id at row {index}: {conversation_id}"
            )
        seen_ids.add(conversation_id)

        question = conversation.last_message(Role.USER)
        if question is None or not isinstance(question.content, str):
            raise ValueError(
                f"Row {index} ({conversation_id}) has no text user message"
            )

        response = conversation.last_message()
        if response is None or response.role != Role.ASSISTANT:
            raise ValueError(
                f"Row {index} ({conversation_id}) does not end with an assistant "
                "response"
            )
        if not isinstance(response.content, str):
            raise ValueError(f"Row {index} ({conversation_id}) has a non-text response")

        reference_answer = conversation.metadata.get("reference_answer")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(
                f"Row {index} ({conversation_id}) has no metadata.reference_answer"
            )


def build_rar_judge_inputs(
    conversations: Sequence[Conversation],
) -> list[dict[str, str]]:
    """Maps conversations onto the judge's `{question}/{reference_answer}/{response}`.

    Same mapping as the training reward (`rar_medicine_grpo.judge_response`), so
    evaluation scores are on the reward's scale.
    """
    judge_inputs = []
    for conversation in conversations:
        question = conversation.last_message(Role.USER)
        response = conversation.last_message()
        assert question is not None and isinstance(question.content, str)
        assert response is not None and isinstance(response.content, str)
        judge_inputs.append(
            {
                "question": question.content,
                "reference_answer": str(conversation.metadata["reference_answer"]),
                "response": response.content,
            }
        )
    return judge_inputs


def _build_judge(judge_config_path: str, num_workers: int) -> tuple[SimpleJudge, str]:
    """Builds the judge, overriding only its request concurrency.

    The training judge config keeps `num_workers` at 1 because the reward path
    supplies its own thread pool; a 1,000-row batch here would run serially.
    The rubric, model, and sampling settings are left untouched so evaluation
    and training grade with the same judge.
    """
    if num_workers < 1:
        raise ValueError("judge_num_workers must be positive")
    judge_config = JudgeConfig.from_path(judge_config_path)
    if judge_config.inference_config is None:
        raise ValueError(f"{judge_config_path} has no inference_config")
    if judge_config.inference_config.remote_params is None:
        judge_config.inference_config.remote_params = RemoteParams()
    judge_config.inference_config.remote_params.num_workers = num_workers
    judge_model = judge_config.inference_config.model.model_name or ""
    return SimpleJudge(judge_config), judge_model


def _load_judgment_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with jsonlines.open(cache_path) as reader:
        for row in reader:
            cached[str(row["conversation_id"])] = row
    return cached


def _judgment_row(
    conversation: Conversation,
    *,
    judgment: int,
    explanation: str | None,
    raw_judge_output: str | None,
    judged: bool,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "idx": conversation.metadata.get("idx"),
        "question_source": conversation.metadata.get("question_source"),
        "judgment": judgment,
        "explanation": explanation,
        "raw_judge_output": raw_judge_output,
        "judged": judged,
    }


def _judge_responses(
    *,
    judge: SimpleJudge,
    conversations: Sequence[Conversation],
    judge_inputs: Sequence[dict[str, str]],
    cache_path: Path,
    progress_path: Path,
    batch_size: int,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    """Judges every conversation not already in the cache, in resumable batches."""
    if batch_size < 1:
        raise ValueError("judge_batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("judge_max_attempts must be positive")

    cached = _load_judgment_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Blank responses score 0 without a judge call, as the training reward did.
    blank_rows = []
    for conversation, judge_input in zip(conversations, judge_inputs, strict=True):
        conversation_id = str(conversation.conversation_id)
        if conversation_id in cached or judge_input["response"].strip():
            continue
        row = _judgment_row(
            conversation,
            judgment=JUDGMENT_MIN,
            explanation="Empty response; scored 0 without judging.",
            raw_judge_output=None,
            judged=False,
        )
        cached[conversation_id] = row
        blank_rows.append(row)
    if blank_rows:
        with jsonlines.open(cache_path, mode="a") as writer:
            writer.write_all(blank_rows)
        logger.warning(f"{len(blank_rows)} blank responses scored 0 without judging")

    missing_indices = [
        index
        for index, conversation in enumerate(conversations)
        if str(conversation.conversation_id) not in cached
    ]
    logger.info(
        f"RaR-Medicine judgments: {len(cached)} cached, "
        f"{len(missing_indices)} remaining"
    )

    for batch_start in range(0, len(missing_indices), batch_size):
        pending = missing_indices[batch_start : batch_start + batch_size]
        errors: dict[int, str] = {}
        for attempt in range(1, max_attempts + 1):
            if not pending:
                break
            batch_inputs = [judge_inputs[index] for index in pending]
            partial_result = judge.judge_partial(
                batch_inputs, progress_path=str(progress_path)
            )
            succeeded: set[int] = set()
            rows_to_write = []
            for local_index, judge_output in partial_result.successful:
                index = pending[local_index]
                judgment = judge_output.field_values.get("judgment")
                if not is_valid_judgment(judgment):
                    errors[index] = (
                        f"Judge output did not contain an integer judgment in "
                        f"[{JUDGMENT_MIN}, {JUDGMENT_MAX}]: {judgment!r}"
                    )
                    continue
                assert isinstance(judgment, int)
                explanation = judge_output.field_values.get("explanation")
                row = _judgment_row(
                    conversations[index],
                    judgment=judgment,
                    explanation=str(explanation) if explanation is not None else None,
                    raw_judge_output=judge_output.raw_output,
                    judged=True,
                )
                rows_to_write.append(row)
                cached[str(conversations[index].conversation_id)] = row
                succeeded.add(index)

            errors.update(
                {
                    pending[local_index]: detail.error_message
                    for local_index, detail in partial_result.failures.items()
                }
            )
            if rows_to_write:
                with jsonlines.open(cache_path, mode="a") as writer:
                    writer.write_all(rows_to_write)

            pending = [index for index in pending if index not in succeeded]
            if pending:
                logger.warning(
                    f"Retrying {len(pending)} judgments after attempt "
                    f"{attempt}/{max_attempts}"
                )

        if pending:
            first = pending[0]
            raise RuntimeError(
                f"Judge failed for {len(pending)} responses after {max_attempts} "
                f"attempts. First failure: {conversations[first].conversation_id}: "
                f"{errors.get(first, 'unknown error')}"
            )
        logger.info(
            f"Completed {min(batch_start + batch_size, len(missing_indices))}/"
            f"{len(missing_indices)} remaining judgments"
        )

    return cached


def _build_sample_results(
    conversations: Sequence[Conversation], judgments: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    sample_results = []
    for conversation in conversations:
        conversation_id = str(conversation.conversation_id)
        grade = judgments[conversation_id]
        response = conversation.last_message()
        assert response is not None and isinstance(response.content, str)
        judgment = int(grade["judgment"])
        sample_results.append(
            {
                "conversation_id": conversation_id,
                "idx": conversation.metadata.get("idx"),
                "question_source": conversation.metadata.get("question_source"),
                "judgment": judgment,
                "score": judgment / JUDGMENT_MAX,
                "judged": bool(grade.get("judged", True)),
                "response": response.content,
                "explanation": grade.get("explanation"),
            }
        )
    return sample_results


def aggregate_rar_scores(
    sample_results: Sequence[dict[str, Any]],
    *,
    num_bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregates per-sample judgments into dataset-level metrics.

    `mean_score` is on the training reward's [0, 1] scale (judgment / 10);
    `mean_judgment` is the same number on the judge's 0-10 scale.
    """
    if not sample_results:
        raise ValueError("sample_results must not be empty")

    scores: list[float] = []
    judgments: list[int] = []
    scores_by_source: defaultdict[str, list[float]] = defaultdict(list)
    for sample in sample_results:
        judgment = int(sample["judgment"])
        if not is_valid_judgment(judgment):
            raise ValueError(f"Judgment out of range: {judgment}")
        judgments.append(judgment)
        scores.append(judgment / JUDGMENT_MAX)
        scores_by_source[str(sample.get("question_source"))].append(
            judgment / JUDGMENT_MAX
        )

    histogram = Counter(judgments)
    return {
        "mean_score": clipped_mean(scores),
        "mean_score_bootstrap_std": bootstrap_std(
            scores, num_bootstrap_samples=num_bootstrap_samples, seed=bootstrap_seed
        ),
        "mean_judgment": float(sum(judgments) / len(judgments)),
        "judgment_histogram": {
            str(value): histogram.get(value, 0)
            for value in range(JUDGMENT_MIN, JUDGMENT_MAX + 1)
        },
        "frac_correct_conclusion": sum(
            judgment >= CORRECT_CONCLUSION_THRESHOLD for judgment in judgments
        )
        / len(judgments),
        "correct_conclusion_threshold": CORRECT_CONCLUSION_THRESHOLD,
        "num_samples": len(sample_results),
        "num_blank_responses": sum(
            not sample.get("judged", True) for sample in sample_results
        ),
        "by_question_source": {
            source: {
                "mean_score": clipped_mean(source_scores),
                "num_samples": len(source_scores),
                "bootstrap_std": bootstrap_std(
                    source_scores,
                    num_bootstrap_samples=num_bootstrap_samples,
                    seed=bootstrap_seed,
                ),
            }
            for source, source_scores in sorted(scores_by_source.items())
        },
    }


def _obtain_responses(
    *,
    responses_path: str | None,
    dataset_path: str | None,
    inference_path: Path,
    inference_engine: BaseInferenceEngine,
) -> list[Conversation]:
    """Returns completed conversations: an existing file, the cache, or fresh."""
    if responses_path is not None:
        logger.info(f"Loading model responses from {responses_path}")
        return load_completed_conversations(Path(responses_path))
    if inference_path.exists():
        logger.info(f"Loading cached model responses from {inference_path}")
        return load_completed_conversations(inference_path)
    if dataset_path is None:
        raise ValueError(
            "Provide `responses_path` (completed conversations to score) or "
            "`dataset_path` (prompt-only conversations to generate from)"
        )
    with jsonlines.open(dataset_path) as reader:
        prompts = [Conversation.from_dict(row) for row in reader]
    if not prompts:
        raise ValueError(f"No conversations found in {dataset_path}")
    conversations = inference_engine.infer(prompts)
    save_conversations(conversations, inference_path)
    logger.info(f"Saved {len(conversations)} model responses to {inference_path}")
    return conversations


@register_evaluation_function("rar_medicine")
def rar_medicine(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    config: EvaluationConfig | None = None,
    judge_config: str = DEFAULT_JUDGE_CONFIG,
    responses_path: str | None = None,
    dataset_path: str | None = None,
    artifact_dir: str | None = None,
    judge_num_workers: int = 16,
    judge_batch_size: int = 250,
    judge_max_attempts: int = 3,
    sample_seed: int = 0,
    num_bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Scores RaR-Medicine responses with the Variant B meta-rubric judge.

    Args:
        task_params: Standard Oumi task params; `num_samples` selects a seeded
            subset of the conversations (the same subset for any file over the
            same prompts).
        inference_engine: Used only when neither `responses_path` nor a cached
            `model_responses.jsonl` exists. Regrade-only configs should set
            `inference_engine: OPENAI` so no model weights are loaded.
        config: The evaluation config; supplies the default artifact location.
        judge_config: SimpleJudge YAML with `{question}`, `{reference_answer}`
            and `{response}` placeholders and an INT judgment.
        responses_path: Completed-conversation jsonl (e.g. `oumi infer` output)
            to score. Each row needs a user turn, a final assistant turn and
            `metadata.reference_answer`.
        dataset_path: Prompt-only conversation jsonl to generate from when no
            responses exist.
        artifact_dir: Where judgments, per-sample results and the summary go.
        judge_num_workers: Concurrent judge requests.
        judge_batch_size: Conversations per judge batch (each batch is
            retried and cached independently).
        judge_max_attempts: Attempts per batch before failing the run.
        sample_seed: Seed for `num_samples` selection and the bootstrap.
        num_bootstrap_samples: Bootstrap resamples for the standard deviation.

    Returns:
        The summary dict, also written to `<artifact_dir>/summary.json`.
    """
    if artifact_dir is None:
        base_output_dir = (
            config.output_dir if config and config.output_dir else "output"
        )
        artifact_dir = str(Path(base_output_dir) / "rar_medicine_artifacts")
    resolved_artifact_dir = Path(artifact_dir)
    inference_path = resolved_artifact_dir / "model_responses.jsonl"
    judgment_path = resolved_artifact_dir / "judgments.jsonl"
    progress_path = resolved_artifact_dir / "judge_progress.json"
    sample_results_path = resolved_artifact_dir / "sample_results.jsonl"
    summary_path = resolved_artifact_dir / "summary.json"

    conversations = _obtain_responses(
        responses_path=responses_path,
        dataset_path=dataset_path,
        inference_path=inference_path,
        inference_engine=inference_engine,
    )
    conversations = select_conversations(
        conversations, task_params.num_samples, sample_seed
    )
    validate_rar_conversations(conversations)

    judge_inputs = build_rar_judge_inputs(conversations)
    judge, judge_model = _build_judge(judge_config, judge_num_workers)
    judgments = _judge_responses(
        judge=judge,
        conversations=conversations,
        judge_inputs=judge_inputs,
        cache_path=judgment_path,
        progress_path=progress_path,
        batch_size=judge_batch_size,
        max_attempts=judge_max_attempts,
    )
    sample_results = _build_sample_results(conversations, judgments)
    save_jsonlines(sample_results, sample_results_path)

    summary = aggregate_rar_scores(
        sample_results,
        num_bootstrap_samples=num_bootstrap_samples,
        bootstrap_seed=sample_seed,
    )
    summary.update(
        {
            "judge_config": judge_config,
            "judge_model": judge_model,
            "responses_path": responses_path,
            "model_name": config.model.model_name if config else None,
            "artifact_dir": str(resolved_artifact_dir),
        }
    )
    write_json(summary_path, summary)
    logger.info(
        f"RaR-Medicine judge score: {summary['mean_score']:.4f} "
        f"± {summary['mean_score_bootstrap_std']:.4f} "
        f"(mean judgment {summary['mean_judgment']:.2f}/10, "
        f"n={summary['num_samples']})"
    )
    return summary
