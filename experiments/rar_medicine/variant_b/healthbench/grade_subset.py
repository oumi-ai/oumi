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

"""Grades an explicit set of prompt_ids against the consolidated rubric.

The validation compares the consolidated score against the true per-sample
rubric score, which exists only for the prefix of the dataset the expensive
per-item run got through. A random 1,000-sample draw overlaps that prefix by
only ~60 examples, too few to bound the agreement. This tops the consolidated
cache up so both metrics exist on the same ~306 samples.

Writes into the same `criterion_grades.jsonl` as the main run: the cache is
keyed by prompt_id and stamped with judge provenance, so the extra rows merge
cleanly and are ignored by any run that did not select them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oumi.core.configs.judge_config import JudgeConfig
from oumi.evaluation.registry.healthbench_common import (
    load_completed_conversations,
    subset_cached_conversations,
)
from oumi.evaluation.registry.healthbench_global_task import (
    _judge_provenance,
    _judge_samples,
    build_global_judge_inputs,
    load_global_rubric,
)
from oumi.judges.simple_judge import SimpleJudge

HERE = Path(__file__).resolve().parent


def fully_judged_prompt_ids(artifact_dir: Path, dataset: Path) -> list[str]:
    """Returns prompt_ids whose per-sample rubrics are all judged."""
    examples = [json.loads(line) for line in dataset.open()]
    by_sample: dict[int, set[int]] = {}
    with (artifact_dir / "rubric_judgments.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            by_sample.setdefault(int(row["sample_index"]), set()).add(
                int(row["rubric_index"])
            )
    return [
        examples[index]["prompt_id"]
        for index, judged in sorted(by_sample.items())
        if len(judged) == len(examples[index]["rubrics"])
    ]


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-dir", required=True)
    parser.add_argument("--truth-dir", required=True)
    parser.add_argument("--rubric", default=str(HERE / "global_rubric_v2.json"))
    parser.add_argument(
        "--judge-config", default=str(HERE / "judge_gpt4o_mini_global.yaml")
    )
    parser.add_argument(
        "--dataset", default=str(HERE / "artifacts/data/healthbench_test.jsonl")
    )
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()

    global_dir = Path(args.global_dir)
    prompt_ids = fully_judged_prompt_ids(Path(args.truth_dir), Path(args.dataset))
    print(f"{len(prompt_ids)} fully-judged ground-truth prompt_ids")

    conversations = subset_cached_conversations(
        load_completed_conversations(global_dir / "model_responses.jsonl"),
        [{"prompt_id": pid} for pid in prompt_ids],
    )
    rubric = load_global_rubric(args.rubric)
    judge_config = JudgeConfig.from_path(args.judge_config)
    provenance = _judge_provenance(judge_config, rubric)
    print(f"provenance: {provenance}")

    graded = _judge_samples(
        judge=SimpleJudge(judge_config),
        judge_inputs=build_global_judge_inputs(conversations, rubric),
        conversations=conversations,
        rubric=rubric,
        provenance=provenance,
        cache_path=global_dir / "criterion_grades.jsonl",
        progress_path=global_dir / "judge_progress_subset.json",
        batch_size=args.batch_size,
        max_attempts=3,
    )
    covered = sum(1 for pid in prompt_ids if pid in graded)
    print(f"cache now covers {covered}/{len(prompt_ids)} ground-truth prompt_ids")


if __name__ == "__main__":
    main()
