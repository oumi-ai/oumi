"""Compare judge runs on one policy's rollouts and write a report.

Metrics per judge run (variant x model):

Label-free checks (use the two control completions in every group):
  ref_top        P(reference answer scores >= every policy sibling)
  mismatch_bottom P(mismatched reference scores <= every policy sibling)
  ref_gap        mean(score(reference) - score(mismatch)); larger = clearer
  mismatch_mean  mean score given to fluent-but-wrong answers (lower = better)

GRPO signal shape (policy samples only, groups of n):
  zero_adv_rate  fraction of groups where all siblings tie (no gradient)
  group_std      mean within-group std of scores
  len_bias       mean within-group Spearman(score, response length)

Agreement with a designated gold run (policy samples only):
  spearman_gold  overall Spearman
  tau_gold       mean within-group Kendall tau-b (groups where both vary)
  top1_gold      P(judge's argmax is among gold's argmax set)
  pair_gold      pairwise ordering agreement over sibling pairs gold orders strictly
                 (judge ties count 0.5, so 0.5 = uninformative, 1.0 = identical order)

Self-consistency (needs a --rep 1 run of the same variant x model):
  self_spearman, self_exact, and self_tau / self_pair (within-group, the noise
  ceiling for tau_gold / pair_gold)

Item-level (explicit-rubric runs only): per rubric category, how often each
judge marks an item met, and how often it agrees with the gold judge's verdict.

Cost: tokens and an approximate USD cost from the price table below.

Usage::

    python analyze.py --policy gemma-4-E2B-it --gold rubric_explicit__gpt-4.1
"""

from __future__ import annotations

import argparse
import itertools
import re

import numpy as np
import pandas as pd
from common import JUDGE_OUTPUTS_DIR, RESULTS_DIR, ROLLOUTS_DIR, read_jsonl
from scipy import stats

# USD per 1M tokens (input, output). Update as needed; only used for the cost column.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}

_RUN_RE = re.compile(r"^(?P<variant>.+?)__(?P<model>.+?)(?:__rep(?P<rep>\d+))?$")


def load_runs(policy: str) -> dict[tuple[str, str, int], list[dict]]:
    runs = {}
    for path in sorted((JUDGE_OUTPUTS_DIR / policy).glob("*.jsonl")):
        if "__limit" in path.stem:
            continue
        m = _RUN_RE.match(path.stem)
        if not m:
            continue
        runs[(m["variant"], m["model"], int(m["rep"] or 0))] = read_jsonl(path)
    return runs


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return stats.spearmanr(a, b).statistic


def group_metrics(df: pd.DataFrame, col: str) -> dict:
    """Control checks + signal shape for one score column."""
    ref_top, mm_bottom, gaps, mm_scores, ref_scores = [], [], [], [], []
    zero_adv, stds, len_rhos = [], [], []
    for _, g in df.groupby("prompt_idx"):
        pol = g[g.kind == "policy"][col].dropna()
        ref = g[g.kind == "reference"][col].dropna()
        mm = g[g.kind == "mismatch"][col].dropna()
        if len(pol) >= 2:
            zero_adv.append((pol == pol.iloc[0]).all())
            stds.append(pol.std(ddof=0))
            lens = g[g.kind == "policy"].loc[pol.index, "response"].str.len()
            len_rhos.append(_spearman(pol, lens))
        if len(ref) and len(pol):
            ref_top.append(ref.iloc[0] >= pol.max())
            ref_scores.append(ref.iloc[0])
        if len(mm) and len(pol):
            mm_bottom.append(mm.iloc[0] <= pol.min())
            mm_scores.append(mm.iloc[0])
        if len(ref) and len(mm):
            gaps.append(ref.iloc[0] - mm.iloc[0])
    return {
        "ref_top": np.mean(ref_top) if ref_top else np.nan,
        "mismatch_bottom": np.mean(mm_bottom) if mm_bottom else np.nan,
        "ref_gap": np.mean(gaps) if gaps else np.nan,
        "ref_mean": np.mean(ref_scores) if ref_scores else np.nan,
        "mismatch_mean": np.mean(mm_scores) if mm_scores else np.nan,
        "policy_mean": df[df.kind == "policy"][col].mean(),
        "zero_adv_rate": np.mean(zero_adv) if zero_adv else np.nan,
        "group_std": np.mean(stds) if stds else np.nan,
        "len_bias": np.nanmean(len_rhos) if len_rhos else np.nan,
    }


def agreement(df: pd.DataFrame, col: str, gold: str) -> dict:
    pol = df[df.kind == "policy"].dropna(subset=[col, gold])
    taus, top1, pair_ok, pair_n = [], [], 0, 0
    for _, g in pol.groupby("prompt_idx"):
        if len(g) < 2:
            continue
        a, b = g[col].to_numpy(float), g[gold].to_numpy(float)
        if np.std(a) > 0 and np.std(b) > 0:
            taus.append(stats.kendalltau(a, b).statistic)
        if np.std(b) > 0:
            gold_best = set(np.flatnonzero(b == b.max()))
            top1.append(int(np.argmax(a)) in gold_best)
        for i, j in itertools.combinations(range(len(g)), 2):
            if b[i] == b[j]:
                continue
            pair_n += 1
            sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            pair_ok += 1.0 if sa == sb else (0.5 if sa == 0 else 0.0)
    return {
        "spearman_gold": _spearman(pol[col], pol[gold]),
        "tau_gold": np.mean(taus) if taus else np.nan,
        "top1_gold": np.mean(top1) if top1 else np.nan,
        "pair_gold": pair_ok / pair_n if pair_n else np.nan,
    }


def cost_usd(model: str, recs: list[dict]) -> float:
    price = PRICES.get(model)
    if not price:
        return np.nan
    pin = sum(r.get("prompt_tokens") or 0 for r in recs)
    pout = sum(r.get("completion_tokens") or 0 for r in recs)
    return (pin * price[0] + pout * price[1]) / 1e6


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument(
        "--gold", default=None, help="'<variant>__<model>' run used as reference."
    )
    args = p.parse_args()

    rollouts = read_jsonl(ROLLOUTS_DIR / f"{args.policy}.jsonl")
    df = pd.DataFrame(rollouts)
    runs = load_runs(args.policy)
    if not runs:
        raise SystemExit("No judge outputs found.")

    cols = {}
    for (variant, model, rep), recs in runs.items():
        assert len(recs) == len(df), (
            f"{variant}/{model}/rep{rep}: {len(recs)} != {len(df)}"
        )
        col = f"{variant}__{model}" + (f"__rep{rep}" if rep else "")
        df[col] = [r["score"] for r in recs]
        cols[(variant, model, rep)] = col

    gold = args.gold
    if gold and gold not in df.columns:
        raise SystemExit(
            f"gold run {gold!r} not found; available: {sorted(cols.values())}"
        )

    rows = []
    for (variant, model, rep), col in cols.items():
        if rep:
            continue
        recs = runs[(variant, model, rep)]
        row = {"variant": variant, "model": model, "fail_rate": df[col].isna().mean()}
        row.update(group_metrics(df, col))
        if gold and col != gold:
            row.update(agreement(df, col, gold))
        rep_col = cols.get((variant, model, 1))
        if rep_col:
            both = df.dropna(subset=[col, rep_col])
            row["self_spearman"] = _spearman(both[col], both[rep_col])
            row["self_exact"] = (both[col] == both[rep_col]).mean()
            selfagree = agreement(df, col, rep_col)
            row["self_tau"] = selfagree["tau_gold"]
            row["self_pair"] = selfagree["pair_gold"]
        row["tokens_in"] = sum(r.get("prompt_tokens") or 0 for r in recs)
        row["tokens_out"] = sum(r.get("completion_tokens") or 0 for r in recs)
        row["cost_usd"] = cost_usd(model, recs)
        row["wall_s"] = recs[0].get("run_wall_s")
        rows.append(row)
    summary = (
        pd.DataFrame(rows).sort_values(["variant", "model"]).reset_index(drop=True)
    )

    # Inter-judge Spearman matrix on policy samples.
    base_cols = [c for (_, _, rep), c in cols.items() if not rep]
    pol = df[df.kind == "policy"]
    corr = pol[base_cols].corr(method="spearman")

    out_dir = RESULTS_DIR / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    corr.to_csv(out_dir / "judge_spearman_matrix.csv")
    df.to_parquet(out_dir / "scores.parquet", index=False)

    # Disagreement digest vs gold: biggest per-sample gaps, for reading.
    if gold:
        digest = []
        for col in base_cols:
            if col == gold:
                continue
            sub = pol.dropna(subset=[col, gold]).copy()
            sub["diff"] = sub[col] - sub[gold]
            worst = sub.reindex(
                sub["diff"].abs().sort_values(ascending=False).index
            ).head(5)
            for _, r in worst.iterrows():
                digest.append(
                    {
                        "judge": col,
                        "prompt_idx": int(r.prompt_idx),
                        "gen_idx": int(r.gen_idx),
                        "judge_score": r[col],
                        "gold_score": r[gold],
                        "question": r.question[:160],
                        "response_head": r.response[:300].replace("\n", " "),
                    }
                )
        pd.DataFrame(digest).to_csv(out_dir / "disagreements_vs_gold.csv", index=False)

    metric_cols = [
        "ref_top",
        "mismatch_bottom",
        "mismatch_mean",
        "zero_adv_rate",
        "group_std",
        "len_bias",
        "spearman_gold",
        "tau_gold",
        "top1_gold",
        "pair_gold",
    ]
    base = summary[~summary.variant.str.endswith(("_noexpl", "_expl"))]
    by_variant = base.groupby("variant")[metric_cols].mean().round(3)
    by_model = base.groupby("model")[metric_cols].mean().round(3)
    by_variant.to_csv(out_dir / "by_variant.csv")
    by_model.to_csv(out_dir / "by_model.csv")

    # Item-level analysis for explicit-rubric runs.
    item_rows = []
    explicit_runs = {
        k: v
        for k, v in runs.items()
        if k[0].startswith("rubric_explicit") and k[2] == 0
    }
    gold_key = None
    if gold:
        gm = _RUN_RE.match(gold)
        if gm and gm["variant"].startswith("rubric_explicit"):
            gold_key = (gm["variant"], gm["model"], 0)
    variant_rubrics: dict[str, list | None] = {}
    for variant, _, _ in explicit_runs:
        if variant not in variant_rubrics:
            try:
                from run_judges import load_variant

                variant_rubrics[variant] = load_variant(variant).get("fixed_rubric")
            except Exception:
                variant_rubrics[variant] = None

    for (variant, model, rep), recs in explicit_runs.items():
        gold_recs = runs.get(gold_key) if gold_key else None
        fixed_rubric = variant_rubrics.get(variant)
        per_cat: dict[str, dict[str, list]] = {}
        for i, (rec, rr) in enumerate(zip(rollouts, recs)):
            rubric_items = fixed_rubric or rec["rubric"]
            met = rr.get("raw_value")
            if not isinstance(met, list) or len(met) != len(rubric_items):
                continue
            same_rubric = (
                gold_key is not None
                and variant_rubrics.get(gold_key[0]) == fixed_rubric
            )
            gmet = (
                gold_recs[i].get("raw_value") if (gold_recs and same_rubric) else None
            )
            if not isinstance(gmet, list) or len(gmet) != len(met):
                gmet = None
            for j, item in enumerate(rubric_items):
                cat = item["description"].split(" Criteria:")[0].strip()
                d = per_cat.setdefault(cat, {"met": [], "agree": [], "kind": []})
                d["met"].append(bool(met[j]))
                d["kind"].append(rec["kind"])
                if gmet is not None:
                    d["agree"].append(bool(met[j]) == bool(gmet[j]))
        for cat, d in per_cat.items():
            kinds = np.array(d["kind"])
            met = np.array(d["met"])
            item_rows.append(
                {
                    "variant": variant,
                    "model": model,
                    "category": cat,
                    "n_items": len(met),
                    "met_rate_policy": met[kinds == "policy"].mean()
                    if (kinds == "policy").any()
                    else np.nan,
                    "met_rate_reference": met[kinds == "reference"].mean()
                    if (kinds == "reference").any()
                    else np.nan,
                    "met_rate_mismatch": met[kinds == "mismatch"].mean()
                    if (kinds == "mismatch").any()
                    else np.nan,
                    "agree_gold": np.mean(d["agree"]) if d["agree"] else np.nan,
                }
            )
    item_df = pd.DataFrame(item_rows)
    if len(item_df):
        item_df.to_csv(out_dir / "explicit_item_level.csv", index=False)

    fmt = summary.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    md = [
        f"# Judge comparison on `{args.policy}` rollouts",
        "",
        f"{len(df)} scored items: {int((df.kind == 'policy').sum())} policy samples in "
        f"{df.prompt_idx.nunique()} groups, plus reference and mismatch controls.",
        f"Gold run: `{gold}`" if gold else "No gold run specified.",
        "",
        "## Summary",
        "",
        fmt.to_markdown(index=False),
        "",
        "## Mean over judge models, per prompt variant",
        "",
        by_variant.to_markdown(),
        "",
        "## Mean over prompt variants, per judge model",
        "",
        by_model.to_markdown(),
        "",
        "## Explicit-rubric item level (met rate by category; agreement with gold)",
        "",
        item_df.round(3).to_markdown(index=False)
        if len(item_df)
        else "(no explicit runs)",
        "",
        "## Inter-judge Spearman (policy samples)",
        "",
        corr.round(2).to_markdown(),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(
        f"\nWrote {out_dir}/summary.{{csv,md}}, judge_spearman_matrix.csv, scores.parquet"
    )


if __name__ == "__main__":
    main()
