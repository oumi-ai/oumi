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

"""Rewrites the criteria that failed the pilot saturation gate.

A criterion nearly every response maxes out contributes the same constant to
every model and cannot take part in a comparison. That is usually not a wording
bug but a scoping one: the criterion names a RARE FAULT, so "did the response
avoid it" is almost always yes. The fix is to re-scope onto the degree of
active risk management around the same concern, where responses actually
differ, using the observed grade histogram as the calibration signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consolidate_rubrics import AXIS_DEFINITIONS, SCALE_MAX, OpenAIChat  # noqa: E402
from hb_env import DEFAULT_KEY_VARNAME, load_api_key  # noqa: E402

HERE = Path(__file__).resolve().parent
CEILING_LIMIT = 0.90
FLOOR_LIMIT = 0.90

REVISE_PROMPT = """\
You are repairing one criterion of a shared rubric used to grade medical chat \
responses. It will be applied unchanged to every HealthBench conversation.

# The criterion, as written
axis: {axis} -- {axis_definition}
id: {criterion_id}
title: {title}
text: {text}
anchors:
{anchors}

# Why it must change
Graded against {count} real responses from two different models, it scored:
  grade histogram: {histogram}
  {ceiling_rate:.1%} of responses received the top grade of {scale_max}
  {floor_rate:.1%} received 0

It therefore hands an identical constant to both models and cannot distinguish \
them. This is a scoping problem, not a wording problem: the criterion names a \
{problem_kind}, so almost every competent response lands at the same end of the \
scale.

# What to produce
Re-scope the criterion onto the same underlying clinical concern, but grade the \
DEGREE to which the response actively addresses it rather than the mere \
presence or absence of a rare fault. Requirements:
- Keep the axis and keep the id "{criterion_id}".
- Keep kind "{kind}".
- Grade {scale_max} must require the response to do something specific and \
demanding that only a minority of responses do -- not merely to be free of the \
fault.
- Grade 2 must describe the TYPICAL response: one that avoids the fault but does \
not do the demanding thing.
- Grade 0 must describe the fault actually being present.
- The criterion must stay sample-agnostic: never name a specific condition, \
drug, dose, test or population.
- CRITICAL -- it must stay DISTINCT from the other criteria in this rubric, \
listed below. Every criterion carries its own weight in the score, so two \
criteria measuring the same behaviour would double-count it. Do not re-scope \
this criterion onto ground another one already covers; find the aspect of the \
original concern that none of them grade.

# The other criteria in the rubric (do not duplicate these)
{siblings}

Return JSON:
{{"criteria": [{{"id": "{criterion_id}", "title": "3-6 words", "text": "one \
complete sentence", "kind": "{kind}", "anchors": {{"0": "...", "2": "...", \
"{scale_max}": "..."}}}}]}}
"""


def load_grades(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    """Loads per-sample grades keyed by prompt_id."""
    path = artifact_dir / "criterion_grades.jsonl"
    with path.open() as handle:
        return {
            str(row["prompt_id"]): row for row in (json.loads(line) for line in handle)
        }


def pilot_stats(
    rubric: dict[str, Any], base: dict[str, Any], trained: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Pools both models' grades and returns per-criterion distributions."""
    common = sorted(set(base) & set(trained))
    stats = {}
    for criterion in rubric["criteria"]:
        cid = criterion["id"]
        pooled = [base[p]["grades"][cid] for p in common]
        pooled += [trained[p]["grades"][cid] for p in common]
        count = len(pooled)
        stats[cid] = {
            "count": count,
            "histogram": {str(g): pooled.count(g) for g in range(SCALE_MAX + 1)},
            "ceiling_rate": pooled.count(SCALE_MAX) / count,
            "floor_rate": pooled.count(0) / count,
        }
    return stats


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default=str(HERE / "global_rubric_v1.json"))
    parser.add_argument("--base", default=str(HERE / "artifacts/global_base"))
    parser.add_argument("--trained", default=str(HERE / "artifacts/global_trained"))
    parser.add_argument("--out", default=str(HERE / "global_rubric_v2.json"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--key-varname", default=DEFAULT_KEY_VARNAME)
    args = parser.parse_args()

    rubric = json.loads(Path(args.rubric).read_text())
    stats = pilot_stats(
        rubric, load_grades(Path(args.base)), load_grades(Path(args.trained))
    )

    failing = [
        criterion
        for criterion in rubric["criteria"]
        if stats[criterion["id"]]["ceiling_rate"] > CEILING_LIMIT
        or stats[criterion["id"]]["floor_rate"] > FLOOR_LIMIT
    ]
    if not failing:
        print("No criterion failed the gate; nothing to revise.")
        return
    print(f"Revising {len(failing)}: {[c['id'] for c in failing]}")

    client = OpenAIChat(args.model, load_api_key(args.key_varname))
    replacements: dict[str, dict[str, Any]] = {}
    for criterion in failing:
        stat = stats[criterion["id"]]
        saturated = stat["ceiling_rate"] > CEILING_LIMIT
        prompt = REVISE_PROMPT.format(
            axis=criterion["axis"],
            axis_definition=AXIS_DEFINITIONS[criterion["axis"]],
            criterion_id=criterion["id"],
            title=criterion["title"],
            text=criterion["text"],
            anchors="\n".join(
                f"  {g} = {criterion['anchors'][g]}"
                for g in sorted(criterion["anchors"], key=int)
            ),
            count=stat["count"],
            histogram=stat["histogram"],
            ceiling_rate=stat["ceiling_rate"],
            floor_rate=stat["floor_rate"],
            scale_max=SCALE_MAX,
            kind=criterion.get("kind", "quality"),
            problem_kind=(
                "rare fault that nearly every competent response already avoids"
                if saturated
                else "demand that nearly no response meets"
            ),
            siblings="\n".join(
                f"  [{other['axis']}] {other['id']} {other['title']}: {other['text']}"
                for other in rubric["criteria"]
                if other["id"] != criterion["id"]
            ),
        )
        payload = client.json_call(prompt, max_tokens=1500)
        items = payload.get("criteria", [])
        if len(items) != 1:
            raise RuntimeError(
                f"{criterion['id']}: revision returned {len(items)} criteria"
            )
        item = items[0]
        missing = {"0", "2", str(SCALE_MAX)} - set(item.get("anchors", {}))
        if missing:
            raise RuntimeError(f"{criterion['id']}: revision missing anchors {missing}")
        replacements[criterion["id"]] = item
        print(
            f"  {criterion['id']}: ceiling {stat['ceiling_rate']:.1%} -> revised "
            f"({item['title']})"
        )

    revised = json.loads(json.dumps(rubric))
    revised["version"] = "v2"
    for criterion in revised["criteria"]:
        item = replacements.get(criterion["id"])
        if not item:
            continue
        criterion["title"] = item["title"].strip()
        criterion["text"] = item["text"].strip()
        criterion["anchors"] = {str(k): str(v) for k, v in item["anchors"].items()}
    revised["provenance"]["revised_from"] = Path(args.rubric).name
    revised["provenance"]["revised_criteria"] = sorted(replacements)
    revised["provenance"]["revision_model"] = args.model
    revised["provenance"]["revision_pilot_stats"] = {
        cid: stats[cid] for cid in replacements
    }
    Path(args.out).write_text(json.dumps(revised, indent=2) + "\n")
    print(f"\nWrote {args.out}")
    for cid in sorted(replacements):
        criterion = next(c for c in revised["criteria"] if c["id"] == cid)
        print(f"\n=== {cid} [{criterion['kind']}] {criterion['title']}")
        print(f"  {criterion['text']}")
        for grade in sorted(criterion["anchors"], key=int):
            print(f"   {grade}: {criterion['anchors'][grade]}")


if __name__ == "__main__":
    main()
