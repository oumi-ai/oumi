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

"""HealthBench evaluation with sample-specific weighted rubrics.

This is the faithful benchmark: every example is graded against its own
physician-written rubric, one judge call per rubric item. See
`healthbench_global_task` for the consolidated dataset-level rubric variant.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonlines

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation
from oumi.evaluation.registry.healthbench_common import (
    DEFAULT_DATASET_URL,
    bootstrap_std,
    calculate_healthbench_score,
    clipped_mean,
    conversation_to_text,
    obtain_responses,
    resolve_dataset,
    save_jsonlines,
    write_json,
)
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger

__all__ = [
    "aggregate_healthbench_scores",
    "build_healthbench_judge_inputs",
    "calculate_healthbench_score",
    "healthbench",
]

DEFAULT_JUDGE_CONFIG = "experiments/rar_medicine/variant_b/healthbench/judge_gpt4o.yaml"


def aggregate_healthbench_scores(
    sample_results: Sequence[dict[str, Any]],
    *,
    num_bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregates example and tag scores into dataset-level HealthBench metrics."""
    if not sample_results:
        raise ValueError("sample_results must not be empty")

    overall_scores: list[float] = []
    tag_scores: defaultdict[str, list[float]] = defaultdict(list)
    rubric_count = 0
    criteria_met_count = 0

    for sample in sample_results:
        score = float(sample["score"])
        overall_scores.append(score)
        for tag in sample.get("example_tags", []):
            tag_scores[tag].append(score)

        rubrics = sample["rubrics"]
        rubric_count += len(rubrics)
        criteria_met_count += sum(bool(rubric["criteria_met"]) for rubric in rubrics)

        rubrics_by_tag: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for rubric in rubrics:
            for tag in rubric.get("tags", []):
                rubrics_by_tag[tag].append(rubric)
        for tag, tagged_rubrics in rubrics_by_tag.items():
            tagged_score = calculate_healthbench_score(
                tagged_rubrics,
                [bool(rubric["criteria_met"]) for rubric in tagged_rubrics],
            )
            if tagged_score is not None:
                tag_scores[tag].append(tagged_score)

    return {
        "overall_score": clipped_mean(overall_scores),
        "overall_score_bootstrap_std": bootstrap_std(
            overall_scores,
            num_bootstrap_samples=num_bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "num_samples": len(sample_results),
        "num_rubric_items": rubric_count,
        "num_criteria_met": criteria_met_count,
        "criteria_met_rate": criteria_met_count / rubric_count if rubric_count else 0.0,
        "tag_scores": {
            tag: {
                "score": clipped_mean(scores),
                "num_samples": len(scores),
                "bootstrap_std": bootstrap_std(
                    scores,
                    num_bootstrap_samples=num_bootstrap_samples,
                    seed=bootstrap_seed,
                ),
            }
            for tag, scores in sorted(tag_scores.items())
        },
    }


def build_healthbench_judge_inputs(
    conversations: Sequence[Conversation],
) -> tuple[list[dict[str, str]], list[tuple[int, int]]]:
    """Flattens sample-specific rubrics into one dataset for Oumi SimpleJudge."""
    judge_inputs: list[dict[str, str]] = []
    locations: list[tuple[int, int]] = []
    for sample_index, conversation in enumerate(conversations):
        conversation_text = conversation_to_text(conversation)
        for rubric_index, rubric in enumerate(conversation.metadata["rubrics"]):
            judge_inputs.append(
                {
                    "conversation": conversation_text,
                    "rubric_item": f"[{rubric['points']}] {rubric['criterion']}",
                }
            )
            locations.append((sample_index, rubric_index))
    return judge_inputs, locations


def _load_judgment_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cached: dict[int, dict[str, Any]] = {}
    with jsonlines.open(cache_path) as reader:
        for row in reader:
            cached[int(row["flat_index"])] = row
    return cached


def _judge_rubrics(
    *,
    judge: SimpleJudge,
    judge_inputs: Sequence[dict[str, str]],
    locations: Sequence[tuple[int, int]],
    conversations: Sequence[Conversation],
    cache_path: Path,
    progress_path: Path,
    batch_size: int,
    max_attempts: int,
) -> dict[int, dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("judge_batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("judge_max_attempts must be positive")

    cached = _load_judgment_cache(cache_path)
    missing_indices = [
        index for index in range(len(judge_inputs)) if index not in cached
    ]
    logger.info(
        f"HealthBench rubric judgments: {len(cached)} cached, "
        f"{len(missing_indices)} remaining"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

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
            succeeded_global_indices = set()
            rows_to_write = []
            for local_index, judge_output in partial_result.successful:
                flat_index = pending[local_index]
                judgment = judge_output.field_values.get("judgment")
                if not isinstance(judgment, bool):
                    errors[flat_index] = (
                        "Judge output did not contain a boolean judgment"
                    )
                    continue
                sample_index, rubric_index = locations[flat_index]
                rubric = conversations[sample_index].metadata["rubrics"][rubric_index]
                row = {
                    "flat_index": flat_index,
                    "sample_index": sample_index,
                    "rubric_index": rubric_index,
                    "prompt_id": conversations[sample_index].metadata["prompt_id"],
                    "criterion": rubric["criterion"],
                    "points": rubric["points"],
                    "tags": rubric["tags"],
                    "criteria_met": judgment,
                    "explanation": judge_output.field_values.get("explanation"),
                    "raw_judge_output": judge_output.raw_output,
                }
                rows_to_write.append(row)
                cached[flat_index] = row
                succeeded_global_indices.add(flat_index)

            errors.update(
                {
                    pending[local_index]: detail.error_message
                    for local_index, detail in partial_result.failures.items()
                }
            )
            if rows_to_write:
                with jsonlines.open(cache_path, mode="a") as writer:
                    writer.write_all(rows_to_write)

            pending = [
                index for index in pending if index not in succeeded_global_indices
            ]
            if pending:
                logger.warning(
                    f"Retrying {len(pending)} rubric judgments after attempt "
                    f"{attempt}/{max_attempts}"
                )

        if pending:
            first_index = pending[0]
            raise RuntimeError(
                f"GPT judge failed for {len(pending)} rubric items after "
                f"{max_attempts} attempts. First failure at flat index "
                f"{first_index}: {errors.get(first_index, 'unknown error')}"
            )
        logger.info(
            f"Completed {min(batch_start + batch_size, len(missing_indices))}/"
            f"{len(missing_indices)} remaining rubric judgments"
        )

    return cached


def _build_sample_results(
    conversations: Sequence[Conversation],
    locations: Sequence[tuple[int, int]],
    judgments: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rubrics_by_sample: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for flat_index, (sample_index, rubric_index) in enumerate(locations):
        grade = judgments[flat_index]
        rubric = conversations[sample_index].metadata["rubrics"][rubric_index]
        rubrics_by_sample[sample_index].append(
            {
                **rubric,
                "criteria_met": grade["criteria_met"],
                "explanation": grade["explanation"],
            }
        )

    sample_results = []
    for sample_index, conversation in enumerate(conversations):
        response = conversation.last_message()
        assert response is not None and isinstance(response.content, str)
        graded_rubrics = rubrics_by_sample[sample_index]
        score = calculate_healthbench_score(
            graded_rubrics,
            [bool(rubric["criteria_met"]) for rubric in graded_rubrics],
        )
        if score is None:
            raise ValueError(
                f"HealthBench sample {sample_index} has no positive-point rubrics"
            )
        sample_results.append(
            {
                "sample_index": sample_index,
                "prompt_id": conversation.metadata["prompt_id"],
                "score": score,
                "example_tags": conversation.metadata["example_tags"],
                "response": response.content,
                "rubrics": graded_rubrics,
            }
        )
    return sample_results


@register_evaluation_function("healthbench")
def healthbench(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    config: EvaluationConfig | None = None,
    dataset_url: str = DEFAULT_DATASET_URL,
    dataset_path: str = "output/healthbench/data/healthbench_test.jsonl",
    judge_config: str = DEFAULT_JUDGE_CONFIG,
    artifact_dir: str | None = None,
    judge_batch_size: int = 500,
    judge_max_attempts: int = 3,
    sample_seed: int = 0,
    num_bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Runs HealthBench generation, per-rubric judging, and dataset aggregation.

    ``num_samples`` comes from the standard Oumi task params. If it is smaller than
    the full dataset, examples are selected with the HealthBench reference seed and
    sampling procedure.
    """
    examples = resolve_dataset(
        dataset_path, dataset_url, task_params.num_samples, sample_seed
    )

    if artifact_dir is None:
        base_output_dir = (
            config.output_dir if config and config.output_dir else "output"
        )
        artifact_dir = str(Path(base_output_dir) / "healthbench_artifacts")
    resolved_artifact_dir = Path(artifact_dir)
    inference_path = resolved_artifact_dir / "model_responses.jsonl"
    judgment_path = resolved_artifact_dir / "rubric_judgments.jsonl"
    progress_path = resolved_artifact_dir / "judge_progress.json"
    sample_results_path = resolved_artifact_dir / "sample_results.jsonl"
    summary_path = resolved_artifact_dir / "summary.json"

    conversations = obtain_responses(
        examples=examples,
        inference_engine=inference_engine,
        inference_path=inference_path,
    )

    judge_inputs, locations = build_healthbench_judge_inputs(conversations)
    judge = SimpleJudge(judge_config)
    judgments = _judge_rubrics(
        judge=judge,
        judge_inputs=judge_inputs,
        locations=locations,
        conversations=conversations,
        cache_path=judgment_path,
        progress_path=progress_path,
        batch_size=judge_batch_size,
        max_attempts=judge_max_attempts,
    )
    sample_results = _build_sample_results(conversations, locations, judgments)
    save_jsonlines(sample_results, sample_results_path)
    summary = aggregate_healthbench_scores(
        sample_results,
        num_bootstrap_samples=num_bootstrap_samples,
        bootstrap_seed=sample_seed,
    )
    summary.update(
        {
            "dataset_url": dataset_url,
            "judge_config": judge_config,
            "model_name": config.model.model_name if config else None,
            "artifact_dir": str(resolved_artifact_dir),
        }
    )
    write_json(summary_path, summary)
    logger.info(
        f"HealthBench score: {summary['overall_score']:.4f} "
        f"± {summary['overall_score_bootstrap_std']:.4f}"
    )
    return summary
