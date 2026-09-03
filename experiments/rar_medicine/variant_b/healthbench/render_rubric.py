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

"""Renders a consolidated rubric JSON as human-readable Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

AXIS_ORDER = [
    "accuracy",
    "completeness",
    "context_awareness",
    "communication_quality",
    "instruction_following",
]


def render(rubric: dict[str, Any]) -> str:
    """Returns the rubric as Markdown."""
    provenance = rubric.get("provenance", {})
    scale_max = rubric["scale"]["max"]
    lines = [
        f"# HealthBench consolidated rubric ({rubric['version']})",
        "",
        "One rubric shared by every sample, abstracted from HealthBench's "
        f"{provenance.get('num_rubric_items', '?')} per-sample rubric items "
        f"({sum(provenance.get('unique_criteria_per_axis', {}).values())} unique "
        f"criteria) over {provenance.get('num_examples', '?')} examples.",
        "",
        f"- **Scale**: every criterion graded 0-{scale_max}.",
        f"- **Score**: `{rubric['score_formula']}`",
        "- **Weights**: axis share of positive point mass, macro-averaged over "
        "examples, matching HealthBench's own averaging. This is the leading term "
        "of the benchmark score's axis decomposition, not an exact reproduction: an "
        "axis whose items in an example are all negative-point gets zero weight "
        "while its points still count in the numerator (34.7% of examples). The "
        "headline delta moves by <= 0.0008 under |points|-mass, pooled or equal "
        "weights, so the convention is not load-bearing.",
        f"- **Synthesised by**: {provenance.get('synthesis_model', '?')}; "
        f"dataset sha256 `{provenance.get('dataset_sha256', '?')[:16]}`.",
        "",
        "## Axis weights",
        "",
        "| axis | macro weight | pooled weight | criteria |",
        "| --- | ---: | ---: | ---: |",
    ]
    pooled = provenance.get("pooled_axis_weights", {})
    for axis in AXIS_ORDER:
        if axis in rubric["axis_weights"]:
            lines.append(
                f"| {axis} | {rubric['axis_weights'][axis]:.4f} | "
                f"{pooled.get(axis, float('nan')):.4f} | "
                f"{rubric['allocation'][axis]} |"
            )
    lines += ["", "## Criteria", ""]
    for axis in AXIS_ORDER:
        criteria = [c for c in rubric["criteria"] if c["axis"] == axis]
        if not criteria:
            continue
        lines.append(f"### {axis.replace('_', ' ')}")
        lines.append("")
        for criterion in criteria:
            kind = (
                " *(harm-avoidance)*"
                if criterion.get("kind") == "harm_avoidance"
                else ""
            )
            lines.append(
                f"**{criterion['id']} - {criterion['title']}**{kind} "
                f"(weight {criterion['weight']:.4f})"
            )
            lines.append("")
            lines.append(f"> {criterion['text']}")
            lines.append("")
            for grade in sorted(criterion["anchors"], key=int):
                lines.append(f"- `{grade}` - {criterion['anchors'][grade]}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rubric")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rubric = json.loads(Path(args.rubric).read_text())
    Path(args.out).write_text(render(rubric))
    print(f"Wrote {args.out} ({len(rubric['criteria'])} criteria)")


if __name__ == "__main__":
    main()
