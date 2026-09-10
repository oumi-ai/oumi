"""Print the headline tables used in results/REPORT.md (run after analyze.py)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from analyze import agreement, load_runs
from common import ROLLOUTS_DIR, read_jsonl

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

policy = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-E2B-it"
SHORT = {
    "Qwen_Qwen3-4B-Instruct-2507": "qwen3-4b (local)",
    "meta-llama_Llama-3.1-8B-Instruct": "llama3.1-8b (local)",
    "google_gemma-4-E4B-it": "gemma4-e4b (local)",
}


def short(s):
    return s.replace(SHORT)


s = pd.read_csv(f"results/{policy}/summary.csv")
s["model"] = short(s["model"])

print(
    "\n## A. Judge model ranking on the explicit rubric (v1), gold = gpt-4.1 explicit"
)
a = s[s.variant == "rubric_explicit"].sort_values("pair_gold", ascending=False)
print(
    a[
        [
            "model",
            "spearman_gold",
            "tau_gold",
            "top1_gold",
            "pair_gold",
            "self_pair",
            "self_exact",
            "ref_top",
            "mismatch_bottom",
            "zero_adv_rate",
            "len_bias",
            "fail_rate",
            "cost_usd",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

print("\n## B. Prompt variants, mean over judge models")
print(pd.read_csv(f"results/{policy}/by_variant.csv").round(2).to_string(index=False))

print("\n## C. Judge models, mean over prompt variants")
bm = pd.read_csv(f"results/{policy}/by_model.csv")
bm["model"] = short(bm["model"])
print(bm.sort_values("pair_gold", ascending=False).round(2).to_string(index=False))

print("\n## D. Self-consistency (noise ceiling), rep0 vs rep1")
d = s.dropna(subset=["self_pair"]).sort_values(["variant", "model"])
print(
    d[["variant", "model", "self_spearman", "self_exact", "self_tau", "self_pair"]]
    .round(2)
    .to_string(index=False)
)

print("\n## E. Explanation on/off (int_0_10 variants)")
e = s[
    s.variant.str.contains("noexpl")
    | s.variant.isin(["rubric_implicit", "reference_only"])
]
e = e[e.model.isin(["gpt-4o-mini", "gpt-4.1-mini"])].sort_values(["model", "variant"])
print(
    e[
        [
            "variant",
            "model",
            "spearman_gold",
            "tau_gold",
            "pair_gold",
            "zero_adv_rate",
            "ref_top",
            "tokens_out",
            "cost_usd",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

print("\n## F. Explicit v1 vs v2 (evidence quotes)")
f = s[s.variant.str.startswith("rubric_explicit")].sort_values(["model", "variant"])
print(
    f[
        [
            "variant",
            "model",
            "spearman_gold",
            "pair_gold",
            "self_exact",
            "self_pair",
            "ref_top",
            "policy_mean",
            "fail_rate",
            "tokens_out",
            "cost_usd",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

# G. Does averaging two judge samples help? mean(rep0, rep1) vs gold.
runs = load_runs(policy)
df = pd.DataFrame(read_jsonl(ROLLOUTS_DIR / f"{policy}.jsonl"))
gold = "rubric_explicit__gpt-4.1"
df[gold] = [r["score"] for r in runs[("rubric_explicit", "gpt-4.1", 0)]]
rows = []
for (variant, model, rep), recs in runs.items():
    if rep != 1 or (variant, model) == ("rubric_explicit", "gpt-4.1"):
        continue
    r0 = runs.get((variant, model, 0))
    if not r0:
        continue
    a0 = np.array([r["score"] if r["score"] is not None else np.nan for r in r0], float)
    a1 = np.array(
        [r["score"] if r["score"] is not None else np.nan for r in recs], float
    )
    df["_single"] = a0
    df["_avg"] = np.nanmean(np.vstack([a0, a1]), axis=0)
    one = agreement(df, "_single", gold)
    two = agreement(df, "_avg", gold)
    rows.append(
        {
            "variant": variant,
            "model": SHORT.get(model, model),
            "pair_gold_1sample": one["pair_gold"],
            "pair_gold_2sample_avg": two["pair_gold"],
            "tau_1": one["tau_gold"],
            "tau_2": two["tau_gold"],
        }
    )
print("\n## G. Averaging two judge samples (vs gold)")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# H. Alternative gold: gpt-5-mini explicit. Does the judge ranking hold?
alt = ("rubric_explicit", "gpt-5-mini", 0)
if alt in runs:
    df["_alt"] = [r["score"] for r in runs[alt]]
    rows = []
    for (variant, model, rep), recs in runs.items():
        if rep or variant != "rubric_explicit" or model == "gpt-5-mini":
            continue
        df["_j"] = [r["score"] for r in recs]
        g1 = (
            agreement(df, "_j", gold)
            if model != "gpt-4.1"
            else {"pair_gold": np.nan, "spearman_gold": np.nan}
        )
        g2 = agreement(df, "_j", "_alt")
        rows.append(
            {
                "model": SHORT.get(model, model),
                "pair_vs_gpt4.1": g1["pair_gold"],
                "pair_vs_gpt5mini": g2["pair_gold"],
                "spearman_vs_gpt4.1": g1["spearman_gold"],
                "spearman_vs_gpt5mini": g2["spearman_gold"],
            }
        )
    print("\n## H. Robustness of the ranking to the choice of gold (explicit v1)")
    print(
        pd.DataFrame(rows)
        .sort_values("pair_vs_gpt5mini", ascending=False)
        .round(2)
        .to_string(index=False)
    )
