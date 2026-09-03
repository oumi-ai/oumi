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

"""Paired base-vs-trained comparison of two `rar_medicine` evaluation runs.

Reads `sample_results.jsonl` from each arm's artifact dir, joins on
`conversation_id`, and reports the paired delta in the judge score with a
bootstrap 95% CI and a paired t statistic -- overall, on the judge's
"correct final conclusion" line (judgment >= 4), per question source, and as a
shift in the 0-10 judgment histogram. Also checks that the trained arm's
responses actually differ from the base arm's (the adapter-activity check
that caught vLLM's silent LoRA no-op on gemma-4).

Caveat carried over from the HealthBench analysis
(../HEALTHBENCH_SUMMARY.md): the two arms were decoded by different engines
(vLLM vs NATIVE). Re-decoding the *same* model moved 30% of HealthBench
per-sample scores by more than 0.10, so a small delta here needs to clear the
decoder noise floor before it is read as a training effect.

Usage:
    python compare_runs.py [--base-dir DIR] [--trained-dir DIR] [--boots N] [--seed S]
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JUDGMENT_LEVELS = range(11)


def _rows(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        return {
            str(row["conversation_id"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def _paired(
    delta: np.ndarray, boots: int, seed: int
) -> tuple[float, float, float, float]:
    """Mean of a paired delta, bootstrap 95% CI, and paired t."""
    if len(delta) < 2:
        return float(delta.mean()), float("nan"), float("nan"), float("nan")
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (boots, len(delta)))].mean(axis=1)
    return (
        float(delta.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        float(delta.mean() / se) if se else 0.0,
    )


def _fmt_paired(
    label: str, base: np.ndarray, trained: np.ndarray, boots: int, seed: int
) -> dict:
    mean, lo, hi, t = _paired(trained - base, boots, seed)
    flag = "  *" if np.isfinite(lo) and not (lo <= 0 <= hi) else ""
    print(
        f"{label:28s} n={len(base):4d}  base {base.mean():.4f}  "
        f"trained {trained.mean():.4f}  delta {mean:+.4f}  "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]  t {t:+.2f}{flag}"
    )
    return {
        "n": int(len(base)),
        "base": float(base.mean()),
        "trained": float(trained.mean()),
        "delta": mean,
        "ci95": [lo, hi],
        "t": t,
        "significant": bool(flag),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-dir", default=str(HERE / "artifacts/base"))
    parser.add_argument("--trained-dir", default=str(HERE / "artifacts/trained"))
    parser.add_argument("--boots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=str(HERE / "artifacts/comparison.json"),
        help="JSON report path",
    )
    args = parser.parse_args()

    base_dir, trained_dir = Path(args.base_dir), Path(args.trained_dir)
    base = _rows(base_dir / "sample_results.jsonl")
    trained = _rows(trained_dir / "sample_results.jsonl")
    ids = sorted(set(base) & set(trained))
    if not ids:
        raise SystemExit("No shared conversation_ids between the two runs")
    only_base, only_trained = len(base) - len(ids), len(trained) - len(ids)
    if only_base or only_trained:
        print(
            f"note: {only_base} base-only and {only_trained} trained-only prompts "
            "excluded from the paired comparison"
        )

    for label, directory in (("base", base_dir), ("trained", trained_dir)):
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            print(
                f"{label:8s} judge={summary.get('judge_model')}  "
                f"config={summary.get('judge_config')}  "
                f"responses={summary.get('responses_path')}"
            )
    print()

    report: dict = {
        "n": len(ids),
        "base_dir": str(base_dir),
        "trained_dir": str(trained_dir),
    }

    print("=" * 78)
    print("1. DO THE TRAINED RESPONSES DIFFER FROM THE BASE RESPONSES?")
    print("=" * 78)
    identical = sum(base[i]["response"] == trained[i]["response"] for i in ids)
    sample = random.Random(args.seed).sample(ids, min(250, len(ids)))
    similarity = [
        difflib.SequenceMatcher(
            None, base[i]["response"], trained[i]["response"]
        ).ratio()
        for i in sample
    ]
    print(f"paired prompts                : {len(ids)}")
    print(
        f"identical responses           : {identical}/{len(ids)} = {identical / len(ids):.1%}"
    )
    print(f"median char similarity (n={len(sample)}): {np.median(similarity):.3f}")
    report["identical_responses"] = identical
    report["median_char_similarity"] = float(np.median(similarity))
    if identical == len(ids):
        print("\n  *** ALL RESPONSES IDENTICAL -- the arms are the same policy. ***")
    print()

    print("=" * 78)
    print("2. JUDGE SCORE (training reward scale, judgment / 10)")
    print("=" * 78)
    bj = np.array([base[i]["judgment"] for i in ids], dtype=float)
    tj = np.array([trained[i]["judgment"] for i in ids], dtype=float)
    report["score"] = _fmt_paired("mean score", bj / 10, tj / 10, args.boots, args.seed)
    report["judgment"] = _fmt_paired(
        "mean judgment (0-10)", bj, tj, args.boots, args.seed
    )
    report["correct_conclusion"] = _fmt_paired(
        "frac judgment >= 4",
        (bj >= 4).astype(float),
        (tj >= 4).astype(float),
        args.boots,
        args.seed,
    )
    print("  (>= 4 is the rubric's 'final conclusion agrees with the reference' line;")
    print("   wrong conclusions are capped at 3)")
    print()

    print("=" * 78)
    print("3. PER-SAMPLE MOVEMENT")
    print("=" * 78)
    delta = tj - bj
    up, down, same = (
        int((delta > 0).sum()),
        int((delta < 0).sum()),
        int((delta == 0).sum()),
    )
    print(f"improved / worsened / unchanged: {up} / {down} / {same}")
    print(f"mean |delta| (judgment points) : {np.abs(delta).mean():.3f}")
    print(
        f"sd of delta                    : {delta.std(ddof=1) if len(delta) > 1 else 0.0:.3f}"
    )
    print(f"moved by >= 3 points           : {int((np.abs(delta) >= 3).sum())}")
    report["movement"] = {
        "improved": up,
        "worsened": down,
        "unchanged": same,
        "mean_abs_delta": float(np.abs(delta).mean()),
        "moved_ge_3": int((np.abs(delta) >= 3).sum()),
    }
    print()

    print("=" * 78)
    print("4. JUDGMENT HISTOGRAM")
    print("=" * 78)
    hb, ht = Counter(bj.astype(int).tolist()), Counter(tj.astype(int).tolist())
    print(f"{'judgment':>8s} {'base':>6s} {'trained':>8s} {'delta':>6s}")
    for level in JUDGMENT_LEVELS:
        print(
            f"{level:>8d} {hb.get(level, 0):>6d} {ht.get(level, 0):>8d} {ht.get(level, 0) - hb.get(level, 0):>+6d}"
        )
    report["histogram"] = {
        "base": {str(level): hb.get(level, 0) for level in JUDGMENT_LEVELS},
        "trained": {str(level): ht.get(level, 0) for level in JUDGMENT_LEVELS},
    }
    print()

    print("=" * 78)
    print("5. PER QUESTION SOURCE (score)")
    print("=" * 78)
    sources = sorted({str(base[i].get("question_source")) for i in ids})
    report["by_question_source"] = {}
    for source in sources:
        source_ids = [i for i in ids if str(base[i].get("question_source")) == source]
        sb = np.array([base[i]["judgment"] for i in source_ids], dtype=float) / 10
        st = np.array([trained[i]["judgment"] for i in source_ids], dtype=float) / 10
        report["by_question_source"][source] = _fmt_paired(
            source[:28], sb, st, args.boots, args.seed
        )
    print("  (* = bootstrap 95% CI excludes 0)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
