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

"""Consolidates HealthBench's 57,237 per-sample rubrics into one shared rubric.

HealthBench writes a bespoke rubric for every example (48,562 unique criterion
strings across 5,000 examples), which makes per-criterion statistics
incomparable across the dataset and costs one judge call per rubric item. This
script abstracts that corpus into a single ~15-criterion rubric that every
sample is graded against, so grading is one call per sample and each criterion
has a dataset-wide pass rate.

Two findings from the corpus drive the design, and both are enforced here:

1. Sample-agnostic criteria saturate. HealthBench already ships 33 shared
   `level:cluster` criteria, and on the two models under evaluation they score
   398/436 -- bit-identical. Binary criteria would therefore report a delta of
   zero regardless of the models. The synthesized criteria are graded 0-4 with
   explicit anchors, and the reduce prompt states the calibration target.
2. Axis weights must be macro-averaged over examples, not pooled over rubric
   items, because HealthBench macro-averages. The two differ materially
   (completeness 0.428 pooled vs 0.397 macro).

Usage:
    python consolidate_rubrics.py --stats-only
    python consolidate_rubrics.py --out global_rubric_v1.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hb_env import DEFAULT_KEY_VARNAME, load_api_key  # noqa: E402

AXES = [
    "accuracy",
    "completeness",
    "context_awareness",
    "communication_quality",
    "instruction_following",
]
TOTAL_CRITERIA = 15
MIN_PER_AXIS = 2
MAX_PER_AXIS = 4
NUM_HARM_CRITERIA = 3
SCALE_MAX = 4

DEFAULT_DATASET = "artifacts/data/healthbench_test.jsonl"


@dataclass
class CriterionRecord:
    """One unique criterion string from the HealthBench corpus."""

    text: str
    axis: str
    is_negative: bool
    count: int = 0
    points: list[int] = field(default_factory=list)
    themes: set[str] = field(default_factory=set)


@dataclass
class CorpusStats:
    """Aggregate statistics over the HealthBench rubric corpus."""

    num_examples: int
    num_rubrics: int
    macro_weights: dict[str, float]
    pooled_weights: dict[str, float]
    positive_mass: dict[str, int]
    negative_mass: dict[str, int]
    item_counts: dict[str, int]
    unique_counts: dict[str, int]
    dataset_sha256: str


def load_examples(path: Path) -> list[dict[str, Any]]:
    """Loads the HealthBench jsonl."""
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _axis_of(rubric: dict[str, Any]) -> str | None:
    return next((t[5:] for t in rubric["tags"] if t.startswith("axis:")), None)


def _themes_of(example: dict[str, Any]) -> set[str]:
    return {t for t in example["example_tags"] if t.startswith("theme:")}


def analyze_corpus(
    examples: list[dict[str, Any]], dataset_path: Path
) -> tuple[CorpusStats, dict[tuple[str, bool], list[CriterionRecord]]]:
    """Computes axis weights and buckets unique criteria by (axis, sign).

    The macro weight of an axis is the mean over examples of that axis's share
    of the example's positive point mass. HealthBench divides by positive points
    only and macro-averages over examples, so this is the leading term of that
    score's axis decomposition -- but not the whole of it: an axis whose items in
    a given example are all negative-point has zero positive mass, so it gets
    zero weight while its points still count in the benchmark's numerator. 34.7%
    of examples have at least one such axis. Negative items otherwise stay inside
    their axis's sub-score rather than forming a separate axis.

    The choice is immaterial in practice: recomputing the headline delta from the
    same grades under |points|-mass, pooled, or equal weights moves it by at most
    0.0008 (see compare_merge_paths.py and the report).
    """
    positive_mass: collections.Counter = collections.Counter()
    negative_mass: collections.Counter = collections.Counter()
    item_counts: collections.Counter = collections.Counter()
    per_example_shares: dict[str, list[float]] = {axis: [] for axis in AXES}
    records: dict[tuple[str, str, bool], CriterionRecord] = {}
    num_rubrics = 0

    for example in examples:
        themes = _themes_of(example)
        example_positive: collections.Counter = collections.Counter()
        for rubric in example["rubrics"]:
            num_rubrics += 1
            axis = _axis_of(rubric)
            if axis is None:
                continue
            points = int(rubric["points"])
            is_negative = points <= 0
            item_counts[axis] += 1
            if is_negative:
                negative_mass[axis] += -points
            else:
                positive_mass[axis] += points
                example_positive[axis] += points

            text = " ".join(rubric["criterion"].split())
            key = (text, axis, is_negative)
            record = records.get(key)
            if record is None:
                record = CriterionRecord(text=text, axis=axis, is_negative=is_negative)
                records[key] = record
            record.count += 1
            record.points.append(points)
            record.themes |= themes

        total_positive = sum(example_positive.values())
        if total_positive <= 0:
            continue
        for axis in AXES:
            per_example_shares[axis].append(example_positive[axis] / total_positive)

    macro_raw = {
        axis: sum(values) / len(values) if values else 0.0
        for axis, values in per_example_shares.items()
    }
    macro_total = sum(macro_raw.values())
    macro_weights = {axis: value / macro_total for axis, value in macro_raw.items()}
    pooled_total = sum(positive_mass.values())
    pooled_weights = {axis: positive_mass[axis] / pooled_total for axis in AXES}

    buckets: dict[tuple[str, bool], list[CriterionRecord]] = collections.defaultdict(
        list
    )
    for record in records.values():
        buckets[(record.axis, record.is_negative)].append(record)

    unique_counts = {axis: 0 for axis in AXES}
    for (axis, _), bucket in buckets.items():
        unique_counts[axis] += len(bucket)

    stats = CorpusStats(
        num_examples=len(examples),
        num_rubrics=num_rubrics,
        macro_weights=macro_weights,
        pooled_weights=pooled_weights,
        positive_mass=dict(positive_mass),
        negative_mass=dict(negative_mass),
        item_counts=dict(item_counts),
        unique_counts=unique_counts,
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    )
    return stats, buckets


def allocate_criteria(macro_weights: dict[str, float]) -> dict[str, int]:
    """Splits TOTAL_CRITERIA across axes, proportional to weight.

    Clamped to [MIN_PER_AXIS, MAX_PER_AXIS]: the floor keeps a diagnostic
    foothold on the light axes, the cap stops the heaviest axis from being
    split into near-duplicates. Influence is carried by the weights, not the
    counts, so clamping changes granularity only.
    """
    raw = {axis: macro_weights[axis] * TOTAL_CRITERIA for axis in AXES}
    alloc = {
        axis: min(MAX_PER_AXIS, max(MIN_PER_AXIS, int(value)))
        for axis, value in raw.items()
    }
    # Spare slots go to the axis with the largest unmet demand (raw - allocated),
    # not the largest fractional part: an axis pushed up to MIN_PER_AXIS is
    # already over-served, and its fractional part says nothing about need.
    for _ in range(TOTAL_CRITERIA):
        if sum(alloc.values()) >= TOTAL_CRITERIA:
            break
        eligible = [axis for axis in AXES if alloc[axis] < MAX_PER_AXIS]
        if not eligible:
            raise RuntimeError("Could not allocate criteria: every axis is at the cap")
        alloc[max(eligible, key=lambda axis: raw[axis] - alloc[axis])] += 1
    for _ in range(TOTAL_CRITERIA):
        if sum(alloc.values()) <= TOTAL_CRITERIA:
            break
        eligible = [axis for axis in AXES if alloc[axis] > MIN_PER_AXIS]
        if not eligible:
            raise RuntimeError(
                "Could not allocate criteria: every axis is at the floor"
            )
        alloc[min(eligible, key=lambda axis: raw[axis] - alloc[axis])] -= 1
    if sum(alloc.values()) != TOTAL_CRITERIA:
        raise RuntimeError(
            f"Allocation sums to {sum(alloc.values())}, expected {TOTAL_CRITERIA}"
        )
    return alloc


def allocate_harm_criteria(negative_mass: dict[str, int]) -> dict[str, int]:
    """Assigns the harm-avoidance slots to the axes carrying negative mass."""
    ranked = sorted(AXES, key=lambda axis: negative_mass.get(axis, 0), reverse=True)
    return {axis: (1 if axis in ranked[:NUM_HARM_CRITERIA] else 0) for axis in AXES}


def sample_criteria(
    bucket: list[CriterionRecord], budget: int, rng: random.Random
) -> list[CriterionRecord]:
    """Samples criteria from a bucket, stratified by theme.

    Round-robins across themes so the low-frequency themes (complex_responses,
    health_data_tasks) are represented rather than swamped by global_health.
    """
    if len(bucket) <= budget:
        return list(bucket)
    by_theme: dict[str, list[CriterionRecord]] = collections.defaultdict(list)
    for record in bucket:
        for theme in record.themes or {"theme:none"}:
            by_theme[theme].append(record)
    for records in by_theme.values():
        rng.shuffle(records)

    chosen: list[CriterionRecord] = []
    seen: set[str] = set()
    themes = sorted(by_theme)
    cursor = {theme: 0 for theme in themes}
    while len(chosen) < budget:
        progressed = False
        for theme in themes:
            records = by_theme[theme]
            while cursor[theme] < len(records):
                record = records[cursor[theme]]
                cursor[theme] += 1
                if record.text in seen:
                    continue
                seen.add(record.text)
                chosen.append(record)
                progressed = True
                break
            if len(chosen) >= budget:
                break
        if not progressed:
            break
    return chosen


AXIS_DEFINITIONS = {
    "accuracy": (
        "factual correctness of the medical content: claims match current evidence and "
        "clinical consensus, numbers and dosing are right, nothing is fabricated"
    ),
    "completeness": (
        "coverage: the response includes the information the user actually needs and "
        "omits nothing important for acting safely on it"
    ),
    "context_awareness": (
        "use of the specific situation: seeking missing context when it changes the "
        "answer, and tailoring to the user's setting, resources and constraints"
    ),
    "communication_quality": (
        "how the content is delivered: clarity, structure, tone, register appropriate "
        "to the reader, and hedging calibrated to genuine uncertainty"
    ),
    "instruction_following": (
        "doing what was actually asked: the requested task, format, language, length "
        "and role, without substituting a different task"
    ),
}

MAP_PROMPT = """\
You are analysing the rubric corpus of HealthBench, a benchmark in which physicians \
wrote a bespoke grading rubric for every medical conversation.

Below are {count} rubric criteria from the "{axis}" axis, which covers {axis_definition}.

Each was written for one specific conversation, so each names specifics -- a drug, a \
symptom, a guideline, a population.{sign_note}

Abstract them into {target} GENERALIZED criteria: sample-agnostic statements, \
applicable to ANY medical conversation, that capture the recurring requirements these \
specific criteria are instances of.

Rules:
- Never name a specific condition, drug, dose, test, or population.
- Stay inside the "{axis}" axis as defined above. Do not produce a criterion that \
really belongs to a different axis.
- Each criterion must be judgeable from the conversation and the response alone.
- Favour criteria that DISCRIMINATE between a mediocre and an excellent response over \
criteria that almost every competent response already satisfies.
- Write each criterion as one complete sentence describing an observable property of \
the response, not a noun phrase.

Return JSON: {{"criteria": [{{"title": "3-6 words", "text": "one complete sentence"}}]}}

# Criteria
{criteria}
"""

REDUCE_PROMPT = """\
You are finalising the "{axis}" axis of a shared rubric for grading medical chat \
responses. It will be applied unchanged to all 5,000 HealthBench conversations.

The "{axis}" axis covers {axis_definition}.

Below are {count} candidate criteria, produced by abstracting {source_count} \
physician-written criteria from this axis.

Merge, deduplicate and prune them into EXACTLY {target} final criteria{harm_note}.

Every criterion must sit squarely inside the "{axis}" axis as defined above -- discard \
any candidate that really belongs to accuracy, completeness, context awareness, \
communication quality or instruction following when that is not this axis.

Each criterion is graded 0-{scale_max} by an LLM judge. Write explicit anchors for \
grades 0, 2 and {scale_max}.

CALIBRATION -- this is the most important requirement. These criteria grade strong \
instruction-tuned models. A previous attempt used binary sample-agnostic criteria and \
91% of responses passed every one, so the rubric could not distinguish two different \
models at all. Write anchors that spread responses out:
- {scale_max} = genuinely excellent; only the strongest ~20% of responses should earn it. \
The anchor must demand something concrete and specific, not merely the absence of faults.
- 2 = competent but unremarkable; the typical fluent, well-formatted response lands here.
- 0 = the property is absent, or clearly violated.
Do not write a {scale_max} anchor that any fluent, well-organised response satisfies by \
default.

{harm_phrasing}

Return JSON:
{{"criteria": [{{"title": "3-6 words", "text": "one complete sentence stating the property \
being graded", "kind": "quality" or "harm_avoidance", \
"anchors": {{"0": "...", "2": "...", "{scale_max}": "..."}}}}]}}

# Candidates
{candidates}
"""

HARM_PHRASING = """\
The harm-avoidance criterion is different: its text names an UNDESIRABLE behaviour that \
a good response avoids (phrase it as "The response does not ..."). Its anchors invert \
accordingly -- {scale_max} = the response fully avoids the behaviour, 2 = a mild or \
borderline instance, 0 = the response clearly exhibits it."""


class OpenAIChat:
    """Thin retrying wrapper around the OpenAI chat completions JSON mode."""

    def __init__(self, model: str, api_key: str, max_attempts: int = 5) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_attempts = max_attempts
        self.num_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def json_call(self, prompt: str, max_tokens: int = 4096) -> dict[str, Any]:
        """Sends one prompt and parses the JSON object it returns."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                self.num_calls += 1
                if response.usage:
                    self.input_tokens += response.usage.prompt_tokens
                    self.output_tokens += response.usage.completion_tokens
                return json.loads(response.choices[0].message.content or "{}")
            except Exception as error:  # noqa: BLE001 - retry any transport/parse error
                last_error = error
                if attempt < self._max_attempts:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"OpenAI call failed after {self._max_attempts} attempts: {last_error}"
        )


def _format_criteria(records: list[CriterionRecord], limit: int = 400) -> str:
    lines = []
    for index, record in enumerate(records, start=1):
        text = record.text
        if len(text) > limit:
            text = text[:limit] + " [...]"
        lines.append(f"{index}. {text}")
    return "\n".join(lines)


def map_stage(
    client: OpenAIChat,
    buckets: dict[tuple[str, bool], list[CriterionRecord]],
    sample_total: int,
    batch_size: int,
    per_batch_target: int,
    workers: int,
    rng: random.Random,
    cache_path: Path | None,
) -> dict[str, list[dict[str, str]]]:
    """Abstracts sampled criteria into per-axis candidate criteria."""
    if cache_path and cache_path.exists():
        print(f"  reusing map-stage cache at {cache_path}")
        return json.loads(cache_path.read_text())

    total_unique = sum(len(bucket) for bucket in buckets.values())
    jobs: list[tuple[str, bool, list[CriterionRecord]]] = []
    for (axis, is_negative), bucket in sorted(buckets.items()):
        budget = max(batch_size, round(sample_total * len(bucket) / total_unique))
        sampled = sample_criteria(bucket, budget, rng)
        for start in range(0, len(sampled), batch_size):
            jobs.append((axis, is_negative, sampled[start : start + batch_size]))

    print(
        f"  {len(jobs)} map calls over {sum(len(j[2]) for j in jobs)} sampled criteria"
    )

    def run(
        job: tuple[str, bool, list[CriterionRecord]],
    ) -> tuple[str, list[dict[str, str]]]:
        axis, is_negative, records = job
        sign_note = (
            " These are NEGATIVE criteria: they describe undesirable behaviour and "
            "carry negative points, so a good response does NOT exhibit them."
            if is_negative
            else ""
        )
        prompt = MAP_PROMPT.format(
            count=len(records),
            axis=axis,
            axis_definition=AXIS_DEFINITIONS[axis],
            sign_note=sign_note,
            target=per_batch_target,
            criteria=_format_criteria(records),
        )
        payload = client.json_call(prompt)
        out = []
        for item in payload.get("criteria", []):
            if isinstance(item, dict) and item.get("text"):
                out.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "text": str(item["text"]).strip(),
                        "kind": "harm_avoidance" if is_negative else "quality",
                    }
                )
        return axis, out

    candidates: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for axis, items in pool.map(run, jobs):
            candidates[axis].extend(items)

    result = {axis: candidates.get(axis, []) for axis in AXES}
    if cache_path:
        cache_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def reduce_stage(
    client: OpenAIChat,
    candidates: dict[str, list[dict[str, str]]],
    allocation: dict[str, int],
    harm_allocation: dict[str, int],
    unique_counts: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Merges per-axis candidates into the final criteria for each axis."""
    final: dict[str, list[dict[str, Any]]] = {}
    for axis in AXES:
        items = candidates[axis]
        harm = harm_allocation[axis]
        harm_note = (
            f", of which EXACTLY {harm} must be a harm-avoidance criterion derived "
            "from the candidates describing undesirable behaviour"
            if harm
            else ", all of them quality criteria"
        )
        listing = "\n".join(
            f"{i}. [{item['kind']}] {item['title']}: {item['text']}"
            for i, item in enumerate(items, start=1)
        )
        prompt = REDUCE_PROMPT.format(
            axis=axis,
            axis_definition=AXIS_DEFINITIONS[axis],
            harm_phrasing=(HARM_PHRASING.format(scale_max=SCALE_MAX) if harm else ""),
            count=len(items),
            source_count=unique_counts[axis],
            target=allocation[axis],
            harm_note=harm_note,
            scale_max=SCALE_MAX,
            candidates=listing,
        )
        payload = client.json_call(prompt, max_tokens=4096)
        criteria = payload.get("criteria", [])
        if len(criteria) != allocation[axis]:
            raise RuntimeError(
                f"Axis {axis}: reduce returned {len(criteria)} criteria, "
                f"expected {allocation[axis]}"
            )
        final[axis] = criteria
        print(f"  {axis}: {len(items)} candidates -> {len(criteria)} final")
    return final


def assemble_rubric(
    final: dict[str, list[dict[str, Any]]],
    allocation: dict[str, int],
    stats: CorpusStats,
    model: str,
    dataset_path: Path,
) -> dict[str, Any]:
    """Builds the frozen rubric document, with per-criterion weights."""
    prefix = {
        "accuracy": "AC",
        "completeness": "CP",
        "context_awareness": "CX",
        "communication_quality": "CM",
        "instruction_following": "IF",
    }
    criteria = []
    for axis in AXES:
        axis_weight = stats.macro_weights[axis]
        per_criterion = axis_weight / allocation[axis]
        for index, item in enumerate(final[axis], start=1):
            anchors = {str(k): str(v) for k, v in (item.get("anchors") or {}).items()}
            missing = {"0", "2", str(SCALE_MAX)} - set(anchors)
            if missing:
                raise RuntimeError(
                    f"{axis} criterion {index} missing anchors {missing}"
                )
            criteria.append(
                {
                    "id": f"{prefix[axis]}{index}",
                    "axis": axis,
                    "kind": item.get("kind", "quality"),
                    "title": str(item.get("title", "")).strip(),
                    "text": str(item["text"]).strip(),
                    "anchors": anchors,
                    "weight": round(per_criterion, 6),
                }
            )
    if len(criteria) != TOTAL_CRITERIA:
        raise RuntimeError(
            f"Assembled {len(criteria)} criteria, expected {TOTAL_CRITERIA}"
        )

    return {
        "version": "v1",
        "scale": {"min": 0, "max": SCALE_MAX},
        "score_formula": (
            "sum(weight_c * grade_c / scale_max) / sum(weight_c), per sample; "
            "dataset score is the mean over samples"
        ),
        "criteria": criteria,
        "axis_weights": {axis: round(stats.macro_weights[axis], 6) for axis in AXES},
        "allocation": allocation,
        "provenance": {
            "source_dataset": str(dataset_path),
            "dataset_sha256": stats.dataset_sha256,
            "num_examples": stats.num_examples,
            "num_rubric_items": stats.num_rubrics,
            "unique_criteria_per_axis": stats.unique_counts,
            "item_counts_per_axis": stats.item_counts,
            "positive_point_mass": stats.positive_mass,
            "negative_point_mass": stats.negative_mass,
            "pooled_axis_weights": {
                axis: round(stats.pooled_weights[axis], 6) for axis in AXES
            },
            "weighting": "macro-averaged per-example share of positive point mass",
            "synthesis_model": model,
        },
    }


def print_stats(
    stats: CorpusStats, allocation: dict[str, int], harm: dict[str, int]
) -> None:
    """Prints the corpus statistics that drive the rubric shape."""
    print(f"examples={stats.num_examples}  rubric items={stats.num_rubrics}")
    print(f"unique criteria={sum(stats.unique_counts.values())}")
    print(f"dataset sha256={stats.dataset_sha256[:16]}...")
    header = f"{'axis':24s} {'items':>7s} {'unique':>7s} {'pooled':>8s} {'macro':>8s} {'crit':>5s} {'harm':>5s}"
    print(header)
    for axis in AXES:
        print(
            f"{axis:24s} {stats.item_counts[axis]:7d} {stats.unique_counts[axis]:7d} "
            f"{stats.pooled_weights[axis]:8.4f} {stats.macro_weights[axis]:8.4f} "
            f"{allocation[axis]:5d} {harm[axis]:5d}"
        )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default="global_rubric_v1.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sample-total", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--per-batch-target", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--key-varname", default=DEFAULT_KEY_VARNAME)
    parser.add_argument("--map-cache", default="artifacts/rubric_map_candidates.json")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = here / dataset_path

    print(f"Loading {dataset_path}")
    examples = load_examples(dataset_path)
    stats, buckets = analyze_corpus(examples, dataset_path)
    allocation = allocate_criteria(stats.macro_weights)
    harm_allocation = allocate_harm_criteria(stats.negative_mass)
    print_stats(stats, allocation, harm_allocation)
    if args.stats_only:
        return

    client = OpenAIChat(args.model, load_api_key(args.key_varname))
    rng = random.Random(args.seed)

    map_cache = Path(args.map_cache)
    if not map_cache.is_absolute():
        map_cache = here / map_cache
    map_cache.parent.mkdir(parents=True, exist_ok=True)

    print("Map stage: abstracting sampled criteria")
    candidates = map_stage(
        client,
        buckets,
        args.sample_total,
        args.batch_size,
        args.per_batch_target,
        args.workers,
        rng,
        map_cache,
    )
    for axis in AXES:
        print(f"  {axis}: {len(candidates[axis])} candidates")

    print("Reduce stage: merging into the final rubric")
    final = reduce_stage(
        client, candidates, allocation, harm_allocation, stats.unique_counts
    )

    rubric = assemble_rubric(final, allocation, stats, args.model, dataset_path)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = here / out_path
    out_path.write_text(json.dumps(rubric, indent=2) + "\n")

    print(f"\nWrote {out_path}")
    print(
        f"OpenAI calls={client.num_calls} in={client.input_tokens} out={client.output_tokens}"
    )
    for criterion in rubric["criteria"]:
        kind = "HARM" if criterion["kind"] == "harm_avoidance" else "    "
        print(
            f"  {criterion['id']:5s} {kind} w={criterion['weight']:.4f}  {criterion['title']}"
        )
        print(
            textwrap.fill(
                criterion["text"],
                96,
                initial_indent=" " * 16,
                subsequent_indent=" " * 16,
            )
        )


if __name__ == "__main__":
    main()
