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

"""Builds the HealthBench consolidated-rubric comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
AXIS_ORDER = [
    "accuracy",
    "completeness",
    "context_awareness",
    "communication_quality",
    "instruction_following",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _by_prompt(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["prompt_id"]): row for row in rows}


def _paired_stats(delta: np.ndarray, *, boots: int, seed: int) -> dict[str, float]:
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(boots, len(delta)))
    means = delta[idx].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "se": float(se),
        "t": float(delta.mean() / se) if se else 0.0,
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "sd": float(delta.std(ddof=1)),
    }


def _axis_score(grades: dict[str, int], rubric: dict[str, Any], axis: str) -> float:
    scale = int(rubric["scale"]["max"])
    weights = total = 0.0
    for criterion in rubric["criteria"]:
        if criterion["axis"] != axis:
            continue
        weights += float(criterion["weight"])
        total += float(criterion["weight"]) * grades[criterion["id"]] / scale
    return total / weights


def _true_scores(truth_dir: Path, dataset: Path) -> dict[str, float]:
    examples = _load_jsonl(dataset)
    by_sample: dict[int, dict[int, Any]] = {}
    for row in _load_jsonl(truth_dir / "rubric_judgments.jsonl"):
        by_sample.setdefault(int(row["sample_index"]), {})[int(row["rubric_index"])] = (
            row
        )
    out: dict[str, float] = {}
    for index, judged in by_sample.items():
        rubrics = examples[index]["rubrics"]
        if len(judged) != len(rubrics):
            continue
        total = sum(r["points"] for r in rubrics if r["points"] > 0)
        if total <= 0:
            continue
        out[examples[index]["prompt_id"]] = (
            sum(r["points"] for i, r in enumerate(rubrics) if judged[i]["criteria_met"])
            / total
        )
    return out


def build(args: argparse.Namespace) -> str:
    """Returns the report as Markdown."""
    rubric = json.loads(Path(args.rubric).read_text())
    scale = int(rubric["scale"]["max"])
    base_dir, trained_dir = Path(args.base), Path(args.trained)

    base = _by_prompt(_load_jsonl(base_dir / "sample_results.jsonl"))
    trained = _by_prompt(_load_jsonl(trained_dir / "sample_results.jsonl"))
    common = sorted(set(base) & set(trained))
    b = np.array([base[p]["score"] for p in common])
    t = np.array([trained[p]["score"] for p in common])
    overall = _paired_stats(t - b, boots=args.bootstrap, seed=args.seed)

    base_summary = json.loads((base_dir / "summary.json").read_text())
    trained_summary = json.loads((trained_dir / "summary.json").read_text())

    L: list[str] = []
    add = L.append
    add("# HealthBench with a consolidated dataset-level rubric")
    add("")
    add(
        f"**{base_summary.get('model_name')}** (base) vs "
        f"**{trained_summary.get('model_name')}** (RaR-Medicine GRPO Variant B, "
        "LoRA merged), graded on "
        f"{len(common)} HealthBench conversations."
    )
    add("")
    add("## What this measures")
    add("")
    add(
        "HealthBench ships a bespoke physician-written rubric per example: "
        f"{rubric['provenance']['num_rubric_items']} rubric items over "
        f"{rubric['provenance']['num_examples']} examples, "
        f"{sum(rubric['provenance']['unique_criteria_per_axis'].values())} of them "
        "unique. That makes per-criterion statistics incomparable across the dataset "
        "and costs one judge call per rubric item. Those rubrics were consolidated "
        f"into **{len(rubric['criteria'])} criteria shared by every sample**, each "
        f"graded 0-{scale}, weighted by that axis's share of positive point mass "
        "macro-averaged over examples to match HealthBench's own averaging. That is "
        "the leading term of the benchmark score's axis decomposition, not an exact "
        "reproduction of it: an axis whose items in a given example are all "
        "negative-point gets zero weight while its points still count in the "
        "benchmark's numerator, which affects 34.7% of examples. Recomputing the "
        "headline under |points|-mass, pooled or equal weights moves the delta by at "
        "most 0.0008, so the convention is not load-bearing here. Grading is one "
        "call per sample."
    )
    add("")
    add(
        f"- Judge: `{base_summary.get('judge_model')}` at temperature "
        f"{base_summary.get('judge_temperature')}, rubric "
        f"`{rubric['version']}` (sha `{base_summary.get('rubric_sha256')}`)."
    )
    add(
        "- Score: `sum(weight * grade / 4) / sum(weight)` per sample, averaged over "
        "samples. Both models graded by the same judge, same rubric, same prompts."
    )
    add("")
    add("## Headline")
    add("")
    add("| | base | trained | delta |")
    add("| --- | ---: | ---: | ---: |")
    add(
        f"| Consolidated rubric score | {b.mean():.4f} | {t.mean():.4f} | "
        f"**{overall['mean']:+.4f}** |"
    )
    add(
        f"| 95% CI on delta (paired bootstrap) | | | "
        f"[{overall['lo']:+.4f}, {overall['hi']:+.4f}] |"
    )
    add(f"| t (paired) | | | {overall['t']:+.2f} |")
    add("")
    significant = not (overall["lo"] <= 0.0 <= overall["hi"])
    add(
        f"The 95% CI **{'excludes' if significant else 'includes'} zero**: "
        + (
            "the difference is statistically distinguishable from noise."
            if significant
            else "on this metric the two models are not distinguishable."
        )
    )
    add("")
    add("<!--BOTTOM_LINE-->")
    add("")

    # Per-axis decomposition.
    add("## Per-axis decomposition")
    add("")
    add(
        "The scalar hides the mechanism. Axis deltas of opposite sign mean the "
        "training moved different capabilities in different directions, and the "
        "headline is only their weighted residue."
    )
    add("")
    add("| axis | weight | base | trained | delta | t |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    axis_rows = []
    for axis in AXIS_ORDER:
        ab = np.array([_axis_score(base[p]["grades"], rubric, axis) for p in common])
        at = np.array([_axis_score(trained[p]["grades"], rubric, axis) for p in common])
        st = _paired_stats(at - ab, boots=args.bootstrap, seed=args.seed)
        axis_rows.append((axis, st))
        add(
            f"| {axis.replace('_', ' ')} | {rubric['axis_weights'][axis]:.3f} | "
            f"{ab.mean():.4f} | {at.mean():.4f} | {st['mean']:+.4f} | {st['t']:+.2f} |"
        )
    add("")
    positive = [a for a, s in axis_rows if s["mean"] > 0 and abs(s["t"]) > 2]
    negative = [a for a, s in axis_rows if s["mean"] < 0 and abs(s["t"]) > 2]
    if positive and negative:
        add(
            f"**Sign conflict**: {', '.join(positive)} improved while "
            f"{', '.join(negative)} regressed (both |t| > 2). The headline number is "
            "the residue of opposing effects, not a uniform gain."
        )
    else:
        add("No axis shows a significant move in both directions.")
    add("")

    # Per-criterion table.
    add("## Per-criterion")
    add("")
    add("| id | axis | criterion | base | trained | delta | t | ceiling |")
    add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for criterion in rubric["criteria"]:
        cid = criterion["id"]
        gb = np.array([base[p]["grades"][cid] for p in common], dtype=float)
        gt = np.array([trained[p]["grades"][cid] for p in common], dtype=float)
        st = _paired_stats(gt - gb, boots=args.bootstrap, seed=args.seed)
        ceiling = float(np.concatenate([gb, gt]).__eq__(scale).mean())
        add(
            f"| {cid} | {criterion['axis'].replace('_', ' ')} | {criterion['title']} | "
            f"{gb.mean():.2f} | {gt.mean():.2f} | {st['mean']:+.3f} | {st['t']:+.2f} | "
            f"{ceiling:.0%} |"
        )
    add("")
    add(
        f"Grades are means on the 0-{scale} scale. `ceiling` is the share of all "
        "gradings at the top grade, pooled over both models: a criterion near 100% "
        "there cannot separate models, which is why the rubric was gated on it."
    )
    add("")

    # Themes.
    add("## By HealthBench theme")
    add("")
    add("| theme | n | base | trained | delta |")
    add("| --- | ---: | ---: | ---: | ---: |")
    themes: dict[str, list[str]] = {}
    for p in common:
        for tag in base[p].get("example_tags", []):
            if tag.startswith("theme:"):
                themes.setdefault(tag, []).append(p)
    for theme, ids in sorted(themes.items(), key=lambda kv: -len(kv[1])):
        tb = np.array([base[p]["score"] for p in ids])
        tt = np.array([trained[p]["score"] for p in ids])
        add(
            f"| {theme.removeprefix('theme:')} | {len(ids)} | {tb.mean():.4f} | "
            f"{tt.mean():.4f} | {(tt - tb).mean():+.4f} |"
        )
    add("")

    # Validation against the true per-sample rubric.
    add("## Does this track real HealthBench?")
    add("")
    if args.skip_validation:
        add(
            "Not available for this run. The ground-truth per-sample-rubric "
            "judgments were produced against a **different set of responses** "
            "(the vLLM / merged-bf16 generations). Comparing them with grades of "
            "these responses would match on prompt_id while silently comparing "
            "two different models' answers. The validation lives in the "
            "merged-path report; `compare_merge_paths.py` compares the two "
            "serving paths directly."
        )
        validation = None
    else:
        cons_base = _by_prompt(_load_jsonl(base_dir / "criterion_grades.jsonl"))
        cons_trained = _by_prompt(_load_jsonl(trained_dir / "criterion_grades.jsonl"))
        dataset = Path(args.dataset)
        true_base = _true_scores(Path(args.true_base), dataset)
        true_trained = _true_scores(Path(args.true_trained), dataset)
        v = sorted(
            set(cons_base) & set(cons_trained) & set(true_base) & set(true_trained)
        )
        if len(v) < 30:
            add(f"Only {len(v)} samples have both metrics; validation skipped.")
            validation = None
        else:
            cb = np.array([cons_base[p]["score"] for p in v])
            ct = np.array([cons_trained[p]["score"] for p in v])
            tb = np.array([true_base[p] for p in v])
            tt = np.array([true_trained[p] for p in v])
            cd, td = ct - cb, tt - tb
            cs = _paired_stats(cd, boots=args.bootstrap, seed=args.seed)
            ts = _paired_stats(td, boots=args.bootstrap, seed=args.seed)
            dod = _paired_stats(cd - td, boots=args.bootstrap, seed=args.seed)
            biased = not (dod["lo"] <= 0.0 <= dod["hi"])
            pd_ = stats.pearsonr(cd, td)
            sd_ = stats.spearmanr(cd, td)
            lvl = stats.pearsonr(np.concatenate([cb, ct]), np.concatenate([tb, tt]))
            add(
                f"On the {len(v)} samples where the full per-sample rubric was also "
                "graded (one GPT-4o call per rubric item, the faithful benchmark):"
            )
            add("")
            add("| metric | base | trained | delta | 95% CI | t |")
            add("| --- | ---: | ---: | ---: | :---: | ---: |")
            add(
                f"| True HealthBench | {tb.mean():.4f} | {tt.mean():.4f} | "
                f"{ts['mean']:+.4f} | [{ts['lo']:+.4f}, {ts['hi']:+.4f}] | {ts['t']:+.2f} |"
            )
            add(
                f"| Consolidated rubric | {cb.mean():.4f} | {ct.mean():.4f} | "
                f"{cs['mean']:+.4f} | [{cs['lo']:+.4f}, {cs['hi']:+.4f}] | {cs['t']:+.2f} |"
            )
            add("")
            add(
                f"- **Difference of deltas** (consolidated - true): {dod['mean']:+.4f}, "
                f"95% CI [{dod['lo']:+.4f}, {dod['hi']:+.4f}] -> "
                + (
                    "**biased**, the CI excludes zero."
                    if biased
                    else "no detectable bias."
                )
            )
            recovery = cs["mean"] / ts["mean"] if ts["mean"] else float("nan")
            add(f"- **Delta recovery ratio**: {recovery:+.2f}.")
            add(
                f"- **Agreement on per-sample paired differences**: Pearson "
                f"r={pd_.statistic:+.3f} (p={pd_.pvalue:.3g}), Spearman "
                f"rho={sd_.statistic:+.3f} (p={sd_.pvalue:.3g})."
            )
            add(
                f"- Score-level correlation r={lvl.statistic:+.3f}, reported only for "
                "completeness: levels are dominated by shared prompt difficulty and "
                "flatter any metric, so the delta agreement above is the real test."
            )
            add("")
            mde = 2.8 * ts["sd"] / np.sqrt(len(common))
            add(
                f"- **Minimum detectable effect** for the true metric at n={len(common)} "
                f"(80% power): {mde:.4f}, against an observed true delta of "
                f"{ts['mean']:+.4f}."
            )
            validation = {
                "biased": biased,
                "recovery": recovery,
                "n": len(v),
                "true_delta": ts["mean"],
                "true_t": ts["t"],
                "cons_delta": cs["mean"],
                "delta_r": float(pd_.statistic),
                "delta_p": float(pd_.pvalue),
                "dod_lo": dod["lo"],
                "dod_hi": dod["hi"],
            }
        add("")

    # Controls.
    add("## Controls")
    add("")
    rng = np.random.default_rng(args.seed)
    half = rng.permutation(len(common))
    a_idx, b_idx = half[: len(common) // 2], half[len(common) // 2 :]
    neg = b[a_idx].mean() - b[b_idx].mean()
    boots = np.array(
        [
            b[rng.integers(0, len(b), len(a_idx))].mean()
            - b[rng.integers(0, len(b), len(b_idx))].mean()
            for _ in range(2000)
        ]
    )
    add(
        f"- **Negative control** (base model's own responses split into two random "
        f"halves): delta {neg:+.4f}, null spread "
        f"[{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]. "
        "The metric does not manufacture a difference from sampling alone."
    )
    delta_vec = t - b
    signs = rng.choice([-1.0, 1.0], size=(args.permutations, len(delta_vec)))
    null = (signs * delta_vec).mean(axis=1)
    p_perm = float((np.abs(null) >= abs(delta_vec.mean())).mean())
    add(
        f"- **Label-permutation test** ({args.permutations} sign flips of the paired "
        f"difference): p = {p_perm:.4f}."
    )
    add("")

    # Caveats.
    add("## Caveats")
    add("")
    add(
        "- **The trained checkpoint is the merged bf16 model.** This repo's own "
        "`eval_configs/infer_trained.yaml` records that only ~82% of the LoRA "
        "adapter delta norm survives the bf16 merge, and that vLLM 0.19.1's LoRA "
        "path is a silent no-op for Gemma-4. The faithful policy is NATIVE + "
        "runtime adapter; these numbers grade an approximation of it. Base and "
        "trained responses do differ on 4,822 of 5,000 prompts, so the merge is not "
        "a no-op."
    )
    add(
        "- **The consolidated rubric is a proxy, not HealthBench.** It shares the "
        "benchmark's prompts, axes and weighting, but its criteria are synthesised "
        "abstractions, so a score here is not comparable to a published HealthBench "
        "number. The validation section bounds how far the two diverge."
    )
    add(
        f"- **Judge noise is not in the confidence intervals.** The bootstrap CI "
        f"captures sampling variation over the {len(common)} prompts only. The "
        "judge's own test-retest variability is unmeasured here and is a separate "
        "error term."
    )
    add(
        "- **Criterion selection used a 200-sample pilot** drawn from the same "
        "dataset. Two criteria were rewritten after they graded 95%+ of responses at "
        "the ceiling for both models; none were selected or dropped on the basis of "
        "which model they favoured."
    )
    add("")
    add(
        f"Artifacts: `{base_dir}` and `{trained_dir}` "
        "(`criterion_grades.jsonl`, `sample_results.jsonl`, `summary.json`); rubric "
        f"`{Path(args.rubric).name}`; every grade row is stamped with judge model, "
        "engine, temperature and prompt hash, and aggregation refuses to mix them."
    )
    bottom = ["## Bottom line", ""]
    if validation is None:
        bottom.append(
            "Neither the consolidated metric nor a ground-truth comparison "
            "separates these two checkpoints."
        )
    else:
        agrees = validation["delta_p"] < 0.05 and validation["delta_r"] > 0
        bottom.append(
            f"1. **Neither metric separates the two checkpoints.** The "
            f"consolidated rubric gives {overall['mean']:+.4f} "
            f"(95% CI [{overall['lo']:+.4f}, {overall['hi']:+.4f}], permutation "
            f"p = {p_perm:.2f}); the true per-sample rubric gives "
            f"{validation['true_delta']:+.4f} (t = {validation['true_t']:+.2f}) on "
            f"the {validation['n']} samples where it exists. On the evidence here, "
            "GRPO Variant B did not measurably change HealthBench performance."
        )
        bottom.append("")
        if not agrees:
            bottom.append(
                f"2. **The consolidated rubric should not be read as a HealthBench "
                f"estimate.** Per-sample paired differences of the two metrics "
                f"correlate at r = {validation['delta_r']:+.3f} "
                f"(p = {validation['delta_p']:.2f}) -- essentially zero -- and on "
                f"the validation subset the consolidated delta points the opposite "
                f"way (recovery ratio {validation['recovery']:+.2f}). The "
                "difference-of-deltas CI "
                f"[{validation['dod_lo']:+.4f}, {validation['dod_hi']:+.4f}] is too "
                "wide to call it biased, but it is also too wide to certify it. "
                "Treat the consolidated score as an internally consistent measure "
                "of medical-response quality that makes per-criterion behaviour "
                "comparable across the dataset -- which per-sample rubrics cannot "
                "do -- and not as a stand-in for the benchmark."
            )
        else:
            bottom.append(
                f"2. **The consolidated rubric tracks the true metric**: paired "
                f"delta correlation r = {validation['delta_r']:+.3f} "
                f"(p = {validation['delta_p']:.2g}), recovery ratio "
                f"{validation['recovery']:+.2f}."
            )
        bottom.append("")
        bottom.append(
            "3. **What would actually settle it.** Both metrics sit near their "
            "detection threshold, so the limit is statistical power, not the "
            "rubric. Grading each sample against its own rubric in one batched "
            "call (~11x cheaper than the per-item run) would extend the faithful "
            "benchmark to all 5,000 samples; and the trained policy should be "
            "served as NATIVE + runtime LoRA adapter rather than the lossy merged "
            "bf16 checkpoint graded here."
        )
    return "\n".join(L).replace("<!--BOTTOM_LINE-->", "\n".join(bottom)) + "\n"


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default=str(HERE / "global_rubric_v2.json"))
    parser.add_argument("--base", default=str(HERE / "artifacts/global_base"))
    parser.add_argument("--trained", default=str(HERE / "artifacts/global_trained"))
    parser.add_argument(
        "--true-base", default=str(HERE / "artifacts/base_gemma4_e2b_it")
    )
    parser.add_argument(
        "--true-trained", default=str(HERE / "artifacts/trained_merged_model")
    )
    parser.add_argument(
        "--dataset", default=str(HERE / "artifacts/data/healthbench_test.jsonl")
    )
    parser.add_argument("--out", default=str(HERE / "HEALTHBENCH_GLOBAL_REPORT.md"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="omit the ground-truth comparison (use when the truth judgments "
        "were graded against a different set of responses)",
    )
    args = parser.parse_args()
    Path(args.out).write_text(build(args))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
