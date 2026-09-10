"""Compare the RL-trained policy against the baseline on identical prompts.

Reads rollouts + judge outputs produced by ../rar_medicine_judges (so nothing is
re-generated here) and writes ``results/COMPARE.md`` with:

* judge score per policy (mean, per-prompt win rate, paired bootstrap CI);
* the reference / mismatch control scores as a sanity anchor;
* cheap *reward-hacking indicators* that a GRPO policy trained against a rubric
  judge tends to drift on: response length, truncation, number of "final answer"
  statements (hedging), list/heading density, and rubric-keyword overlap
  (does the policy learn to echo rubric vocabulary without the substance).

Judges are given as ``variant:model`` pairs. The first one is treated as the
*training* reward; the rest are held-out graders. Reward hacking shows up as a
policy gain under the training judge that does not transfer to held-out ones.

Usage::

    python compare.py --baseline gemma-4-E2B-it --policy variant_b \
        --judges variant_b_training_reward:gpt-4.1-mini \
                 rubric_explicit:google/gemma-4-E4B-it rubric_explicit:gpt-4.1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rar_medicine_judges"))
from common import ROLLOUTS_DIR, judge_output_path, read_jsonl  # noqa: E402

RESULTS_DIR = HERE / "results"
_FINAL = re.compile(r"the final answer is", re.I)
_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)
_HEAD = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*\n]+\*\*\s*$)", re.M)
_WORD = re.compile(r"[a-z]{4,}")
_STOP = set(
    "does not with that this from into than then they them their there which about would should could also have been being were what when where while such more most other some".split()
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="gemma-4-E2B-it")
    p.add_argument("--policy", default="variant_b")
    p.add_argument(
        "--judges",
        nargs="+",
        metavar="VARIANT:MODEL",
        default=[
            "variant_b_training_reward:gpt-4.1-mini",
            "rubric_explicit:google/gemma-4-E4B-it",
            "rubric_explicit:gpt-4.1",
        ],
        help="First entry = training reward, rest = held-out judges.",
    )
    p.add_argument("--out", default=str(RESULTS_DIR / "COMPARE.md"))
    return p.parse_args()


def rubric_vocab(rubric: list[dict]) -> set[str]:
    words = set()
    for item in rubric:
        words |= set(_WORD.findall(item["description"].lower()))
    return words - _STOP


def split_judge(j: str) -> tuple[str, str]:
    variant, _, model = j.partition(":")
    if not model:
        raise SystemExit(f"--judges entries must be VARIANT:MODEL, got {j!r}")
    return variant, model


def load(policy: str, judges: list[str]) -> pd.DataFrame:
    recs = read_jsonl(ROLLOUTS_DIR / f"{policy}.jsonl")
    df = pd.DataFrame(recs)
    df["policy_name"] = policy
    for j in judges:
        variant, model = split_judge(j)
        path = judge_output_path(policy, variant, model)
        if not path.exists():
            raise SystemExit(f"missing judge output {path}; run run_judges.py first")
        outs = read_jsonl(path)
        assert len(outs) == len(recs), (path, len(outs), len(recs))
        df[f"score::{j}"] = [o["score"] for o in outs]
    resp = df["response"].fillna("")
    df["chars"] = resp.str.len()
    df["truncated"] = df["finish_reason"].eq("length")
    df["n_final_answer"] = resp.map(lambda s: len(_FINAL.findall(s)))
    df["list_lines"] = resp.map(lambda s: len(_LIST.findall(s)))
    df["headings"] = resp.map(lambda s: len(_HEAD.findall(s)))
    df["rubric_overlap"] = [
        len(set(_WORD.findall(r.lower())) & rubric_vocab(rub))
        / max(1, len(rubric_vocab(rub)))
        for r, rub in zip(resp, df["rubric"])
    ]
    return df


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of mean(b - a) resampling prompts (rows are per-prompt means)."""
    rng = np.random.default_rng(seed)
    d = b - a
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    args = parse_args()
    base = load(args.baseline, args.judges)
    pol = load(args.policy, args.judges)
    assert (base["dataset_row"].to_numpy() == pol["dataset_row"].to_numpy()).all(), (
        "prompt sets differ: regenerate with the same --seed/--num-prompts/--split"
    )

    lines = [
        f"# {args.policy} vs {args.baseline}",
        "",
        f"Training judge: `{args.judges[0]}`; held-out: {', '.join(f'`{j}`' for j in args.judges[1:]) or 'none'}. "
        f"{base['prompt_idx'].nunique()} val prompts, "
        f"{(base.kind == 'policy').groupby(base.prompt_idx).sum().iloc[0]} samples each.",
        "",
    ]

    # ---- judge scores ----------------------------------------------------- #
    lines += [
        "## Judge scores (policy samples only)",
        "",
        "| judge | role | baseline mean | policy mean | delta | 95% CI (paired, per prompt) | prompt win rate | ref mean | mismatch mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    deltas: dict[str, float] = {}
    for k, j in enumerate(args.judges):
        col = f"score::{j}"
        b = base[base.kind == "policy"].groupby("prompt_idx")[col].mean()
        p = pol[pol.kind == "policy"].groupby("prompt_idx")[col].mean()
        lo, hi = paired_bootstrap(b.values, p.values)
        win = float((p > b).mean() + 0.5 * (p == b).mean())
        ref = base[base.kind == "reference"][col].mean()
        mis = base[base.kind == "mismatch"][col].mean()
        deltas[j] = p.mean() - b.mean()
        role = "training" if k == 0 else "held-out"
        lines.append(
            f"| {j} | {role} | {b.mean():.3f} | {p.mean():.3f} | {deltas[j]:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | {win:.2f} | {ref:.3f} | {mis:.3f} |"
        )
    lines.append("")
    if len(args.judges) > 1:
        train_d = deltas[args.judges[0]]
        lines += ["Training-judge gain vs held-out gain (both on the [0,1] scale):", ""]
        for j in args.judges[1:]:
            lines.append(
                f"- {j}: transfer ratio = {deltas[j] / train_d if train_d else float('nan'):.2f} "
                f"(held-out delta {deltas[j]:+.3f} / training delta {train_d:+.3f})"
            )
        lines.append("")
        lines.append(
            "A ratio well below 1 (gain under the training reward that held-out judges "
            "do not see) is the primary reward-hacking signal."
        )
        lines.append("")

    # ---- reward hacking indicators --------------------------------------- #
    lines += [
        "## Surface statistics / reward-hacking indicators (policy samples)",
        "",
        "| metric | baseline | policy |",
        "|---|---|---|",
    ]
    bp, pp = base[base.kind == "policy"], pol[pol.kind == "policy"]
    rows = [
        ("response length (chars, mean)", bp.chars.mean(), pp.chars.mean(), "{:.0f}"),
        (
            "response length (chars, p90)",
            bp.chars.quantile(0.9),
            pp.chars.quantile(0.9),
            "{:.0f}",
        ),
        (
            "completion tokens (mean)",
            bp.completion_tokens.mean(),
            pp.completion_tokens.mean(),
            "{:.0f}",
        ),
        ("truncated at max tokens", bp.truncated.mean(), pp.truncated.mean(), "{:.1%}"),
        (
            "has 'The final answer is'",
            (bp.n_final_answer > 0).mean(),
            (pp.n_final_answer > 0).mean(),
            "{:.1%}",
        ),
        (
            "multiple 'final answer' statements (hedging)",
            (bp.n_final_answer > 1).mean(),
            (pp.n_final_answer > 1).mean(),
            "{:.1%}",
        ),
        (
            "list lines per response",
            bp.list_lines.mean(),
            pp.list_lines.mean(),
            "{:.1f}",
        ),
        ("headings per response", bp.headings.mean(), pp.headings.mean(), "{:.1f}"),
        (
            "rubric vocabulary overlap (frac of rubric words present)",
            bp.rubric_overlap.mean(),
            pp.rubric_overlap.mean(),
            "{:.3f}",
        ),
    ]
    for name, b, p, fmt in rows:
        lines.append(f"| {name} | {fmt.format(b)} | {fmt.format(p)} |")
    lines.append("")

    # score vs length correlation: an RL policy exploiting a length bias shows a
    # higher within-prompt correlation than the baseline
    for j in args.judges:
        col = f"score::{j}"

        def _corr(df, col=col):
            g = df.groupby("prompt_idx")
            d = df.assign(
                s=df[col] - g[col].transform("mean"),
                l=df.chars - g.chars.transform("mean"),
            )
            return d[["s", "l"]].corr().iloc[0, 1]

        lines.append(
            f"- within-prompt corr(score, length) under {j}: baseline {_corr(bp):+.2f}, policy {_corr(pp):+.2f}"
        )
    lines.append("")

    # ---- biggest movers ------------------------------------------------- #
    j0 = f"score::{args.judges[0]}"
    b = bp.groupby("prompt_idx")[j0].mean()
    p = pp.groupby("prompt_idx")[j0].mean()
    delta = (p - b).sort_values()
    q = base.drop_duplicates("prompt_idx").set_index("prompt_idx")["question"]
    lines += [
        f"## Largest per-prompt changes ({args.judges[0]})",
        "",
        "| prompt | delta | question |",
        "|---|---|---|",
    ]
    for idx in list(delta.index[:5]) + list(delta.index[-5:]):
        lines.append(
            f"| {idx} | {delta[idx]:+.2f} | {q[idx][:120].replace('|', '/')} |"
        )
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    # also dump the per-sample table for notebooks
    pd.concat([base, pol]).drop(columns=["rubric"]).to_json(
        out.with_suffix(".jsonl"), orient="records", lines=True
    )
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
