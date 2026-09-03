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

"""HealthBench evaluated against one consolidated, dataset-level rubric.

HealthBench ships a bespoke rubric per example -- 57,237 rubric items over 5,000
examples -- which makes per-criterion statistics incomparable across the dataset
and costs one judge call per rubric item. This harness grades every example
against a single shared rubric (see `consolidate_rubrics.py`), so grading is one
call per sample and each criterion has a dataset-wide distribution.

Two design points are driven by measurements on the cached ground truth rather
than by preference:

- Criteria are graded 0-4, not binary. HealthBench's own 33 sample-agnostic
  `level:cluster` criteria score 398/436 for both models under comparison --
  bit-identical -- so a binary shared rubric reports a delta of zero regardless
  of the models being compared.
- Every judgment record carries the judge model, engine, temperature and a hash
  of the prompt template, and aggregation refuses to mix them. Two judge
  endpoints with documented failover exist in this project, and grading one
  model through each would fabricate a difference larger than the effect being
  measured.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonlines

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.judge_config import JudgeConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation
from oumi.evaluation.registry.healthbench_common import (
    DEFAULT_DATASET_URL,
    bootstrap_std,
    clipped_mean,
    conversation_to_text,
    obtain_responses,
    resolve_dataset,
    save_jsonlines,
    write_json,
)
from oumi.judges.simple_judge import SimpleJudge
from oumi.utils.logging import logger

DEFAULT_RUBRIC_PATH = (
    "experiments/rar_medicine/variant_b/healthbench/global_rubric_v1.json"
)
DEFAULT_JUDGE_CONFIG = (
    "experiments/rar_medicine/variant_b/healthbench/judge_gpt4o_mini_global.yaml"
)

__all__ = [
    "aggregate_global_scores",
    "build_global_judge_inputs",
    "healthbench_global",
    "load_global_rubric",
    "parse_grades",
    "render_rubric_block",
    "score_sample",
]


def load_global_rubric(rubric_path: str | Path) -> dict[str, Any]:
    """Loads and validates the consolidated rubric document."""
    path = Path(rubric_path)
    rubric = json.loads(path.read_text())

    criteria = rubric.get("criteria")
    if not criteria:
        raise ValueError(f"Rubric at {path} has no criteria")
    scale_max = int(rubric["scale"]["max"])
    if scale_max < 1:
        raise ValueError(f"Rubric scale max must be >= 1, got {scale_max}")

    ids = [criterion["id"] for criterion in criteria]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Rubric at {path} has duplicate criterion ids")
    for criterion in criteria:
        for key in ("id", "axis", "text", "weight", "anchors"):
            if key not in criterion:
                raise ValueError(f"Criterion {criterion.get('id')} is missing '{key}'")
        if float(criterion["weight"]) <= 0:
            raise ValueError(f"Criterion {criterion['id']} has a non-positive weight")
    return rubric


def rubric_fingerprint(rubric: dict[str, Any]) -> str:
    """Returns a short hash of the scoring-relevant parts of the rubric."""
    payload = [
        {
            "id": criterion["id"],
            "text": criterion["text"],
            "weight": criterion["weight"],
            "anchors": criterion["anchors"],
        }
        for criterion in rubric["criteria"]
    ]
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def render_rubric_block(rubric: dict[str, Any]) -> str:
    """Renders the criteria as the constant text block shown to the judge.

    This block is byte-identical for every sample and is placed at the front of
    the prompt, so the shared prefix is long enough for the provider's automatic
    prompt caching to apply across the run.
    """
    scale_max = int(rubric["scale"]["max"])
    lines = []
    for criterion in rubric["criteria"]:
        lines.append(f"### {criterion['id']} - {criterion['title']}")
        lines.append(criterion["text"])
        for grade in sorted(criterion["anchors"], key=int):
            lines.append(f"  {grade} = {criterion['anchors'][grade]}")
        if criterion.get("kind") == "harm_avoidance":
            lines.append(
                f"  (harm-avoidance: {scale_max} means the response avoids this; "
                "0 means it clearly exhibits it)"
            )
        lines.append("")
    return "\n".join(lines).strip()


def build_global_judge_inputs(
    conversations: Sequence[Conversation], rubric: dict[str, Any]
) -> list[dict[str, str]]:
    """Builds one judge input per sample: the shared rubric plus the conversation."""
    rubric_block = render_rubric_block(rubric)
    return [
        {"rubric": rubric_block, "conversation": conversation_to_text(conversation)}
        for conversation in conversations
    ]


# The judge is asked for digits, but occasionally writes the word from its own
# explanation into the judgment field ("AC3=absent"). For a quality criterion the
# grade-0 anchor is literally "the property is absent", and the judge's paired
# explanation grades such cases 0, so the word is accepted as 0 there. It is NOT
# accepted for a harm-avoidance criterion, where "absent" is ambiguous in the
# dangerous direction: the absent thing could be the harmful behaviour, which is
# grade 4, not grade 0.
_ZERO_WORDS = frozenset({"absent", "none", "n/a", "na", "missing", "not present"})


def parse_grades(judgment: str, rubric: dict[str, Any]) -> dict[str, int]:
    """Parses an ``AC1=3,AC2=2,...`` judgment string into per-criterion grades.

    Grades are keyed by criterion id rather than by position so that a dropped,
    duplicated or reordered criterion is an error instead of a silent shift of
    every downstream grade.
    """
    expected = {criterion["id"] for criterion in rubric["criteria"]}
    kinds = {c["id"]: c.get("kind", "quality") for c in rubric["criteria"]}
    scale_max = int(rubric["scale"]["max"])

    grades: dict[str, int] = {}
    for token in judgment.replace(";", ",").replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        key, separator, value = token.partition("=")
        if not separator:
            key, separator, value = token.partition(":")
        if not separator:
            raise ValueError(f"Malformed grade token {token!r}; expected 'ID=grade'")
        key = key.strip().upper()
        raw = value.strip()
        try:
            grade = int(raw)
        except ValueError as error:
            word = raw.lower().strip(".\"'")
            if word in _ZERO_WORDS and kinds.get(key) != "harm_avoidance":
                grade = 0
            else:
                raise ValueError(f"Non-integer grade in token {token!r}") from error
        if key in grades:
            raise ValueError(f"Duplicate grade for criterion {key}")
        if not 0 <= grade <= scale_max:
            raise ValueError(f"Grade {grade} for {key} is outside 0..{scale_max}")
        grades[key] = grade

    if set(grades) != expected:
        missing = sorted(expected - set(grades))
        unexpected = sorted(set(grades) - expected)
        raise ValueError(
            f"Grade set mismatch; missing={missing} unexpected={unexpected}"
        )
    return grades


def score_sample(grades: dict[str, int], rubric: dict[str, Any]) -> float:
    """Returns the weighted score in [0, 1] for one sample."""
    scale_max = int(rubric["scale"]["max"])
    total_weight = 0.0
    achieved = 0.0
    for criterion in rubric["criteria"]:
        weight = float(criterion["weight"])
        total_weight += weight
        achieved += weight * grades[criterion["id"]] / scale_max
    if total_weight <= 0:
        raise ValueError("Rubric weights sum to zero")
    return achieved / total_weight


def _axis_score(
    grades: dict[str, int], rubric: dict[str, Any], axis: str
) -> float | None:
    scale_max = int(rubric["scale"]["max"])
    total_weight = 0.0
    achieved = 0.0
    for criterion in rubric["criteria"]:
        if criterion["axis"] != axis:
            continue
        weight = float(criterion["weight"])
        total_weight += weight
        achieved += weight * grades[criterion["id"]] / scale_max
    return achieved / total_weight if total_weight > 0 else None


def aggregate_global_scores(
    sample_results: Sequence[dict[str, Any]],
    rubric: dict[str, Any],
    *,
    num_bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregates per-sample grades into dataset-level metrics.

    Reports per-criterion saturation alongside the score: a criterion that
    almost every response maxes out contributes an identical constant to every
    model and cannot participate in a comparison, which is the failure mode this
    rubric was built to avoid.
    """
    if not sample_results:
        raise ValueError("sample_results must not be empty")

    scale_max = int(rubric["scale"]["max"])
    overall_scores = [float(sample["score"]) for sample in sample_results]
    grades_by_criterion: defaultdict[str, list[int]] = defaultdict(list)
    axis_scores: defaultdict[str, list[float]] = defaultdict(list)
    theme_scores: defaultdict[str, list[float]] = defaultdict(list)

    for sample in sample_results:
        grades = sample["grades"]
        for criterion_id, grade in grades.items():
            grades_by_criterion[criterion_id].append(int(grade))
        for axis in sorted({c["axis"] for c in rubric["criteria"]}):
            value = _axis_score(grades, rubric, axis)
            if value is not None:
                axis_scores[axis].append(value)
        for tag in sample.get("example_tags", []):
            if tag.startswith("theme:"):
                theme_scores[tag].append(float(sample["score"]))

    criterion_stats = {}
    for criterion in rubric["criteria"]:
        values = grades_by_criterion[criterion["id"]]
        count = len(values)
        criterion_stats[criterion["id"]] = {
            "axis": criterion["axis"],
            "kind": criterion.get("kind", "quality"),
            "title": criterion.get("title", ""),
            "weight": float(criterion["weight"]),
            "mean_grade": sum(values) / count,
            "normalized_score": sum(values) / (count * scale_max),
            "ceiling_rate": sum(1 for v in values if v == scale_max) / count,
            "floor_rate": sum(1 for v in values if v == 0) / count,
            "grade_histogram": {
                str(g): sum(1 for v in values if v == g) for g in range(scale_max + 1)
            },
        }

    return {
        "overall_score": clipped_mean(overall_scores),
        "overall_score_bootstrap_std": bootstrap_std(
            overall_scores,
            num_bootstrap_samples=num_bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "num_samples": len(sample_results),
        "rubric_version": rubric.get("version"),
        "rubric_sha256": rubric_fingerprint(rubric),
        "criterion_stats": criterion_stats,
        "axis_scores": {
            axis: {
                "score": clipped_mean(values),
                "num_samples": len(values),
                "bootstrap_std": bootstrap_std(
                    values,
                    num_bootstrap_samples=num_bootstrap_samples,
                    seed=bootstrap_seed,
                ),
            }
            for axis, values in sorted(axis_scores.items())
        },
        "theme_scores": {
            theme: {"score": clipped_mean(values), "num_samples": len(values)}
            for theme, values in sorted(theme_scores.items())
        },
    }


def _judge_provenance(
    judge_config: JudgeConfig, rubric: dict[str, Any]
) -> dict[str, Any]:
    inference_config = judge_config.inference_config
    if inference_config is None:
        raise ValueError("Judge config must define an inference_config")
    generation = inference_config.generation
    template = judge_config.judge_params.prompt_template
    system = judge_config.judge_params.system_instruction or ""
    return {
        "judge_model": inference_config.model.model_name,
        "judge_engine": str(
            getattr(inference_config.engine, "value", inference_config.engine)
        ),
        "judge_temperature": float(generation.temperature) if generation else None,
        "prompt_sha256": hashlib.sha256((template + system).encode()).hexdigest()[:16],
        "rubric_sha256": rubric_fingerprint(rubric),
    }


def _load_grade_cache(
    cache_path: Path, provenance: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Loads cached grades, refusing rows graded under a different setup.

    Keyed by prompt_id, not by position: `num_samples` smaller than the dataset
    draws a random subset, so row N of a 200-sample pilot and row N of the full
    run are different prompts. A positional cache would silently serve one
    prompt's grades for another.
    """
    if not cache_path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with jsonlines.open(cache_path) as reader:
        for row in reader:
            row_provenance = {key: row.get(key) for key in provenance}
            if row_provenance != provenance:
                raise ValueError(
                    f"{cache_path} contains judgments from a different setup "
                    f"({row_provenance} != {provenance}). Grading a comparison "
                    "under mixed judges invalidates it; move the file aside to "
                    "regrade from scratch."
                )
            cached[str(row["prompt_id"])] = row
    return cached


def _judge_samples(
    *,
    judge: SimpleJudge,
    judge_inputs: Sequence[dict[str, str]],
    conversations: Sequence[Conversation],
    rubric: dict[str, Any],
    provenance: dict[str, Any],
    cache_path: Path,
    progress_path: Path,
    batch_size: int,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    """Grades every sample against the shared rubric, with resume and retry."""
    if batch_size < 1:
        raise ValueError("judge_batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("judge_max_attempts must be positive")

    prompt_ids = [
        str(conversation.metadata["prompt_id"]) for conversation in conversations
    ]
    cached = _load_grade_cache(cache_path, provenance)
    missing = [
        index for index in range(len(judge_inputs)) if prompt_ids[index] not in cached
    ]
    logger.info(
        f"HealthBench global rubric: {len(cached)} cached, {len(missing)} remaining"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(missing), batch_size):
        pending = missing[batch_start : batch_start + batch_size]
        errors: dict[int, str] = {}
        for attempt in range(1, max_attempts + 1):
            if not pending:
                break
            partial = judge.judge_partial(
                [judge_inputs[index] for index in pending],
                progress_path=str(progress_path),
            )
            succeeded: set[int] = set()
            rows: list[dict[str, Any]] = []
            for local_index, judge_output in partial.successful:
                sample_index = pending[local_index]
                judgment = judge_output.field_values.get("judgment")
                if not isinstance(judgment, str):
                    errors[sample_index] = "Judge output has no text judgment"
                    continue
                try:
                    grades = parse_grades(judgment, rubric)
                except ValueError as error:
                    errors[sample_index] = str(error)
                    continue
                row = {
                    "sample_index": sample_index,
                    "prompt_id": prompt_ids[sample_index],
                    "grades": grades,
                    "score": score_sample(grades, rubric),
                    "explanation": judge_output.field_values.get("explanation"),
                    **provenance,
                }
                rows.append(row)
                cached[prompt_ids[sample_index]] = row
                succeeded.add(sample_index)

            errors.update(
                {
                    pending[local_index]: detail.error_message
                    for local_index, detail in partial.failures.items()
                }
            )
            if rows:
                with jsonlines.open(cache_path, mode="a") as writer:
                    writer.write_all(rows)

            pending = [index for index in pending if index not in succeeded]
            if pending:
                logger.warning(
                    f"Retrying {len(pending)} samples after attempt "
                    f"{attempt}/{max_attempts}"
                )

        if pending:
            first = pending[0]
            raise RuntimeError(
                f"Judge failed for {len(pending)} samples after {max_attempts} "
                f"attempts. First failure at sample {first}: "
                f"{errors.get(first, 'unknown error')}"
            )
        logger.info(
            f"Graded {min(batch_start + batch_size, len(missing))}/{len(missing)} "
            "remaining samples"
        )
    return cached


@register_evaluation_function("healthbench_global")
def healthbench_global(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    config: EvaluationConfig | None = None,
    dataset_url: str = DEFAULT_DATASET_URL,
    dataset_path: str = "output/healthbench/data/healthbench_test.jsonl",
    global_rubric_path: str = DEFAULT_RUBRIC_PATH,
    judge_config: str = DEFAULT_JUDGE_CONFIG,
    artifact_dir: str | None = None,
    judge_batch_size: int = 250,
    judge_max_attempts: int = 3,
    sample_seed: int = 0,
    num_bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Runs HealthBench generation and grading against one consolidated rubric.

    Reuses `model_responses.jsonl` under `artifact_dir` when present, so a run
    that already generated responses under the per-sample-rubric harness can be
    regraded without touching a GPU.
    """
    examples = resolve_dataset(
        dataset_path, dataset_url, task_params.num_samples, sample_seed
    )
    rubric = load_global_rubric(global_rubric_path)

    if artifact_dir is None:
        base_output_dir = (
            config.output_dir if config and config.output_dir else "output"
        )
        artifact_dir = str(Path(base_output_dir) / "healthbench_global_artifacts")
    resolved_artifact_dir = Path(artifact_dir)
    inference_path = resolved_artifact_dir / "model_responses.jsonl"
    grades_path = resolved_artifact_dir / "criterion_grades.jsonl"
    progress_path = resolved_artifact_dir / "judge_progress.json"
    sample_results_path = resolved_artifact_dir / "sample_results.jsonl"
    summary_path = resolved_artifact_dir / "summary.json"

    conversations = obtain_responses(
        examples=examples,
        inference_engine=inference_engine,
        inference_path=inference_path,
    )

    resolved_judge_config = JudgeConfig.from_path(judge_config)
    provenance = _judge_provenance(resolved_judge_config, rubric)
    logger.info(f"HealthBench global rubric judge provenance: {provenance}")

    judge_inputs = build_global_judge_inputs(conversations, rubric)
    graded = _judge_samples(
        judge=SimpleJudge(resolved_judge_config),
        judge_inputs=judge_inputs,
        conversations=conversations,
        rubric=rubric,
        provenance=provenance,
        cache_path=grades_path,
        progress_path=progress_path,
        batch_size=judge_batch_size,
        max_attempts=judge_max_attempts,
    )

    sample_results = []
    for sample_index, conversation in enumerate(conversations):
        row = graded[str(conversation.metadata["prompt_id"])]
        response = conversation.last_message()
        assert response is not None and isinstance(response.content, str)
        sample_results.append(
            {
                "sample_index": sample_index,
                "prompt_id": conversation.metadata["prompt_id"],
                "score": row["score"],
                "grades": row["grades"],
                "example_tags": conversation.metadata["example_tags"],
                "response": response.content,
                "explanation": row.get("explanation"),
            }
        )
    save_jsonlines(sample_results, sample_results_path)

    summary = aggregate_global_scores(
        sample_results,
        rubric,
        num_bootstrap_samples=num_bootstrap_samples,
        bootstrap_seed=sample_seed,
    )
    summary.update(
        {
            "dataset_url": dataset_url,
            "global_rubric_path": global_rubric_path,
            "judge_config": judge_config,
            "model_name": config.model.model_name if config else None,
            "artifact_dir": str(resolved_artifact_dir),
            **provenance,
        }
    )
    write_json(summary_path, summary)
    logger.info(
        f"HealthBench global-rubric score: {summary['overall_score']:.4f} "
        f"± {summary['overall_score_bootstrap_std']:.4f}"
    )
    return summary
