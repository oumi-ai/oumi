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

"""Creates a Markdown comparison report from two HealthBench summaries."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _format_score(score: float) -> str:
    return f"{100 * score:.2f}%"


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"HealthBench summary not found: {path}")
    return json.loads(path.read_text())


def make_report(base: dict[str, Any], trained: dict[str, Any]) -> str:
    """Builds the model comparison report."""
    base_score = float(base["overall_score"])
    trained_score = float(trained["overall_score"])
    delta = trained_score - base_score
    shared_tags = set(base["tag_scores"]) & set(trained["tag_scores"])
    tag_deltas = sorted(
        (
            (
                tag,
                float(trained["tag_scores"][tag]["score"]),
                float(base["tag_scores"][tag]["score"]),
            )
            for tag in shared_tags
        ),
        key=lambda row: row[1] - row[2],
        reverse=True,
    )

    result_delta = 100 * delta
    lines = [
        "# HealthBench evaluation report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Result",
        "",
        "| Model | HealthBench score | Bootstrap std. | Samples | Rubric items |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Untrained `{base['model_name']}` | {_format_score(base_score)} | "
            f"{_format_score(float(base['overall_score_bootstrap_std']))} | "
            f"{base['num_samples']} | {base['num_rubric_items']} |"
        ),
        (
            f"| Trained `{trained['model_name']}` | {_format_score(trained_score)} | "
            f"{_format_score(float(trained['overall_score_bootstrap_std']))} | "
            f"{trained['num_samples']} | {trained['num_rubric_items']} |"
        ),
        "",
        f"The trained-minus-untrained difference is **{result_delta:+.2f} "
        "percentage points**.",
        "",
        "Scores follow the OpenAI HealthBench reference formula: each met rubric adds",
        "its signed point value, each sample is normalized by its positive-point total,",
        "and the dataset result is the clipped mean of the sample scores.",
        "",
        "## Largest tag changes",
        "",
        "| Tag | Trained | Untrained | Difference |",
        "|---|---:|---:|---:|",
    ]
    for tag, trained_tag_score, base_tag_score in tag_deltas[:10]:
        lines.append(
            f"| `{tag}` | {_format_score(trained_tag_score)} | "
            f"{_format_score(base_tag_score)} | "
            f"{100 * (trained_tag_score - base_tag_score):+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Dataset: `{trained['dataset_url']}`",
            f"- Judge config: `{trained['judge_config']}`",
            "- Judge: `gpt-4o`, temperature 0, one binary judgment per rubric item",
            f"- Untrained artifacts: `{base['artifact_dir']}`",
            f"- Trained artifacts: `{trained['artifact_dir']}`",
            "",
            "The artifact directories contain model responses, every raw GPT-4o",
            "judgment and explanation, per-sample scored rubrics, and summary metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/tmp/oumi_healthbench/base_gemma4_e2b_it/summary.json"),
    )
    parser.add_argument(
        "--trained",
        type=Path,
        default=Path("/tmp/oumi_healthbench/trained_merged_model/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/rar_medicine/variant_b/healthbench/HEALTHBENCH_REPORT.md"
        ),
    )
    args = parser.parse_args()
    report = make_report(_load_summary(args.base), _load_summary(args.trained))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
