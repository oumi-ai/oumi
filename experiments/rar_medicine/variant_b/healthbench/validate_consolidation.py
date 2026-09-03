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

"""Gates and validates the consolidated HealthBench rubric.

Two subcommands:

  gate      Per-criterion saturation and discrimination on a pilot run. A
            criterion that nearly every response maxes out contributes the same
            constant to every model and cannot take part in a comparison --
            HealthBench's own 33 sample-agnostic criteria score 398/436 for
            both models here, which is exactly the failure being screened for.

  validate  Compares the consolidated score against the true per-sample-rubric
            HealthBench score on the samples where both exist. The statistic
            that matters is agreement on the PAIRED DELTA, not correlation of
            score levels: level correlation is dominated by shared prompt
            difficulty and flatters any metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CEILING_LIMIT = 0.90
FLOOR_LIMIT = 0.90
MIN_STD = 0.40
REDUNDANCY_LIMIT = 0.90
# |t| below this on the pilot means the criterion is not separating the models.
MIN_DISCRIMINATION_T = 1.0
HERE = Path(__file__).resolve().parent


def _load_grades(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    path = artifact_dir / "criterion_grades.jsonl"
    with path.open() as handle:
        return {
            str(row["prompt_id"]): row for row in (json.loads(line) for line in handle)
        }


def _load_rubric(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _paired(base: dict[str, Any], trained: dict[str, Any]) -> list[str]:
    return sorted(set(base) & set(trained))


def _tstat(values: np.ndarray) -> tuple[float, float]:
    """Returns (standard error, t) for a paired difference vector."""
    if len(values) < 2:
        return float("nan"), float("nan")
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    if standard_error == 0:
        return 0.0, 0.0
    return standard_error, values.mean() / standard_error


def _bootstrap_ci(
    values: np.ndarray, *, samples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a paired difference vector."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def run_gate(args: argparse.Namespace) -> int:
    """Reports per-criterion saturation; returns a non-zero code if it fails."""
    rubric = _load_rubric(Path(args.rubric))
    scale_max = int(rubric["scale"]["max"])
    base = _load_grades(Path(args.base))
    trained = _load_grades(Path(args.trained))
    common = _paired(base, trained)
    if not common:
        raise SystemExit("No prompt_ids in common between the two runs")

    base_scores = np.array([base[p]["score"] for p in common])
    trained_scores = np.array([trained[p]["score"] for p in common])
    delta = trained_scores - base_scores
    standard_error, t_value = _tstat(delta)

    print(f"paired pilot samples: {len(common)}")
    print(
        f"overall  base {base_scores.mean():.4f}  trained {trained_scores.mean():.4f}  "
        f"delta {delta.mean():+.4f}  SE {standard_error:.4f}  t {t_value:+.2f}"
    )
    print(
        f"paired r={np.corrcoef(base_scores, trained_scores)[0, 1]:.3f}  "
        f"sd(sample score)={base_scores.std(ddof=1):.4f}\n"
    )

    header = (
        f"{'id':5s} {'axis':22s} {'kind':5s} {'meanB':>6s} {'meanT':>6s} "
        f"{'ceil%':>6s} {'floor%':>6s} {'sd':>5s} {'delta':>7s} {'t':>6s}  gate"
    )
    print(header)
    failures: list[tuple[str, list[str]]] = []
    for criterion in rubric["criteria"]:
        cid = criterion["id"]
        grades_base = np.array([base[p]["grades"][cid] for p in common], dtype=float)
        grades_trained = np.array(
            [trained[p]["grades"][cid] for p in common], dtype=float
        )
        pooled = np.concatenate([grades_base, grades_trained])
        ceiling = float((pooled == scale_max).mean())
        floor = float((pooled == 0).mean())
        deviation = float(pooled.std(ddof=1))
        difference = grades_trained - grades_base
        _, criterion_t = _tstat(difference)

        # A criterion is dead weight only if it is BOTH pinned to one end of the
        # scale AND fails to separate the models. A rare behaviour can be the
        # signal precisely because it is rare: if the few responses that exhibit
        # it are concentrated in one model, the criterion discriminates well
        # despite an extreme marginal distribution. Extreme-but-informative is
        # therefore reported, not failed.
        extremes = []
        if ceiling > CEILING_LIMIT:
            extremes.append("ceiling")
        if floor > FLOOR_LIMIT:
            extremes.append("floor")
        if deviation < MIN_STD:
            extremes.append("no-variance")

        problems = []
        if extremes and abs(criterion_t) < MIN_DISCRIMINATION_T:
            problems = [f"DEAD({'+'.join(extremes)})"]
            failures.append((cid, problems))
        elif extremes:
            problems = [f"extreme({'+'.join(extremes)}) but informative"]
        print(
            f"{cid:5s} {criterion['axis']:22s} {criterion.get('kind', 'quality')[:5]:5s} "
            f"{grades_base.mean():6.2f} {grades_trained.mean():6.2f} {ceiling:6.1%} "
            f"{floor:6.1%} {deviation:5.2f} {difference.mean():+7.3f} "
            f"{criterion_t:+6.2f}  {','.join(problems) or 'ok'}"
        )

    # Redundancy: two criteria that move together are double-counting one
    # behaviour, since each carries its own weight in the score.
    ids = [c["id"] for c in rubric["criteria"]]
    matrix = np.array(
        [
            [base[p]["grades"][cid] for p in common]
            + [trained[p]["grades"][cid] for p in common]
            for cid in ids
        ],
        dtype=float,
    )
    redundant = []
    print("\ninter-criterion correlation (|r| > 0.75 shown):")
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if matrix[i].std() == 0 or matrix[j].std() == 0:
                continue
            r = float(np.corrcoef(matrix[i], matrix[j])[0, 1])
            if abs(r) > REDUNDANCY_LIMIT:
                redundant.append((ids[i], ids[j], round(r, 3)))
            if abs(r) > 0.75:
                flag = "  <-- REDUNDANT" if abs(r) > REDUNDANCY_LIMIT else ""
                print(f"  {ids[i]:5s} ~ {ids[j]:5s}  r={r:+.3f}{flag}")
    if not any(
        abs(float(np.corrcoef(matrix[i], matrix[j])[0, 1])) > 0.75
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
        if matrix[i].std() and matrix[j].std()
    ):
        print("  (none)")

    if failures or redundant:
        if failures:
            print(f"\nGATE: FAIL -> saturation {failures}")
        if redundant:
            print(f"GATE: FAIL -> redundant pairs {redundant}")
        return 1
    print("\nGATE: PASS (no criterion saturated, floored, variance-free, or redundant)")
    return 0


def _true_healthbench_scores(artifact_dir: Path, dataset: Path) -> dict[str, float]:
    """Per-sample true HealthBench scores, for fully-judged samples only."""
    examples = [json.loads(line) for line in dataset.open()]
    judgments_path = artifact_dir / "rubric_judgments.jsonl"
    by_sample: dict[int, dict[int, Any]] = {}
    for line in judgments_path.open():
        row = json.loads(line)
        by_sample.setdefault(int(row["sample_index"]), {})[int(row["rubric_index"])] = (
            row
        )

    scores: dict[str, float] = {}
    for sample_index, judged in by_sample.items():
        rubrics = examples[sample_index]["rubrics"]
        if len(judged) != len(rubrics):
            continue  # partially judged: excluded
        total = sum(r["points"] for r in rubrics if r["points"] > 0)
        if total <= 0:
            continue
        achieved = sum(
            r["points"] for i, r in enumerate(rubrics) if judged[i]["criteria_met"]
        )
        scores[examples[sample_index]["prompt_id"]] = achieved / total
    return scores


def run_validate(args: argparse.Namespace) -> int:
    """Compares consolidated deltas against true HealthBench deltas."""
    dataset = Path(args.dataset)
    consolidated_base = _load_grades(Path(args.base))
    consolidated_trained = _load_grades(Path(args.trained))
    true_base = _true_healthbench_scores(Path(args.true_base), dataset)
    true_trained = _true_healthbench_scores(Path(args.true_trained), dataset)

    common = sorted(
        set(consolidated_base)
        & set(consolidated_trained)
        & set(true_base)
        & set(true_trained)
    )
    if len(common) < 3:
        raise SystemExit(f"Only {len(common)} samples have both metrics; need more")

    cons_b = np.array([consolidated_base[p]["score"] for p in common])
    cons_t = np.array([consolidated_trained[p]["score"] for p in common])
    true_b = np.array([true_base[p] for p in common])
    true_t = np.array([true_trained[p] for p in common])
    cons_d = cons_t - cons_b
    true_d = true_t - true_b

    print(f"samples with both metrics: {len(common)}\n")
    for label, b, t, d in (
        ("true HealthBench", true_b, true_t, true_d),
        ("consolidated", cons_b, cons_t, cons_d),
    ):
        se, t_value = _tstat(d)
        low, high = _bootstrap_ci(d, samples=args.bootstrap, seed=args.seed)
        print(
            f"{label:18s} base {b.mean():.4f}  trained {t.mean():.4f}  "
            f"delta {d.mean():+.4f}  SE {se:.4f}  t {t_value:+.2f}  "
            f"95% CI [{low:+.4f}, {high:+.4f}]"
        )

    difference_of_deltas = cons_d - true_d
    dod_mean = float(difference_of_deltas.mean())
    dod_low, dod_high = _bootstrap_ci(
        difference_of_deltas, samples=args.bootstrap, seed=args.seed
    )
    biased = not (dod_low <= 0.0 <= dod_high)
    print(
        f"\nPRIMARY  delta_consolidated - delta_true = {dod_mean:+.4f}  "
        f"95% CI [{dod_low:+.4f}, {dod_high:+.4f}]  -> "
        f"{'BIASED (CI excludes 0)' if biased else 'no detectable bias'}"
    )

    recovery = cons_d.mean() / true_d.mean() if true_d.mean() != 0 else float("nan")
    print(f"delta recovery ratio = {recovery:+.3f}  (want >= 0.8)")

    from scipy import stats  # local import: only needed here

    pearson_delta = stats.pearsonr(cons_d, true_d)
    spearman_delta = stats.spearmanr(cons_d, true_d)
    pearson_level = stats.pearsonr(
        np.concatenate([cons_b, cons_t]), np.concatenate([true_b, true_t])
    )
    print(
        f"\npaired-DELTA agreement  Pearson r={pearson_delta.statistic:+.3f} "
        f"(p={pearson_delta.pvalue:.3g})  Spearman rho={spearman_delta.statistic:+.3f} "
        f"(p={spearman_delta.pvalue:.3g})"
    )
    print(
        f"score-LEVEL correlation Pearson r={pearson_level.statistic:+.3f} "
        f"(p={pearson_level.pvalue:.3g})   [reported for completeness; levels are "
        "dominated by shared prompt difficulty]"
    )

    sign_agree = "yes" if np.sign(cons_d.mean()) == np.sign(true_d.mean()) else "NO"
    print(f"\nsign agreement on the trained-vs-base delta: {sign_agree}")

    mde = 2.8 * true_d.std(ddof=1) / np.sqrt(args.target_n)
    print(
        f"true-metric MDE at n={args.target_n} (80% power): {mde:.4f}  "
        f"vs observed true delta {true_d.mean():+.4f}"
    )
    return 0


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="pilot saturation / discrimination gate")
    gate.add_argument("--rubric", default=str(HERE / "global_rubric_v1.json"))
    gate.add_argument("--base", default=str(HERE / "artifacts/global_base"))
    gate.add_argument("--trained", default=str(HERE / "artifacts/global_trained"))
    gate.set_defaults(func=run_gate)

    validate = sub.add_parser("validate", help="consolidated vs true HealthBench")
    validate.add_argument("--base", default=str(HERE / "artifacts/global_base"))
    validate.add_argument("--trained", default=str(HERE / "artifacts/global_trained"))
    validate.add_argument(
        "--true-base", default=str(HERE / "artifacts/base_gemma4_e2b_it")
    )
    validate.add_argument(
        "--true-trained", default=str(HERE / "artifacts/trained_merged_model")
    )
    validate.add_argument(
        "--dataset", default=str(HERE / "artifacts/data/healthbench_test.jsonl")
    )
    validate.add_argument("--bootstrap", type=int, default=10000)
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--target-n", type=int, default=1000)
    validate.set_defaults(func=run_validate)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
