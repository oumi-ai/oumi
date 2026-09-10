"""Training curve on the held-out test set: score every available checkpoint
with the training judge and gold judges, side by side.

    python checkpoint_curve.py                 # default checkpoints + judges below
    python checkpoint_curve.py --judges variant_b_training_reward:gpt-4.1-mini fixed_rubric_implicit:claude-sonnet-5

Reads results/test1k/responses__<model>.jsonl and judge__<model>__*.jsonl (run
`eval_test1k.py generate/judge` first); writes results/test1k/CURVE.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "test1k"
CKPTS = [
    ("base", 0),
    ("fullft_step200", 200),
    ("fullft", 250),
]  # final save == step 250 weights (verified bitwise)
JUDGES = [
    "variant_b_training_reward:gpt-4.1-mini",
    "fixed_rubric_implicit:gpt-5.6-terra",
    "fixed_rubric_implicit:claude-sonnet-5",
]
_FINAL = re.compile(r"the final answer is", re.I)
_DUP = re.compile(r"the final answer is[^\n]*\n+\s*the final answer is", re.I)


def scores(model, judge):
    v, _, m = judge.partition(":")
    p = OUT / f"judge__{model}__{v}__{m.replace('/', '_')}.jsonl"
    if not p.exists():
        return None
    return np.array(
        [
            (
                json.loads(line)["score"]
                if json.loads(line)["score"] is not None
                else np.nan
            )
            for line in open(p)
        ]
    )


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--judges", nargs="+", default=JUDGES)
    a.add_argument(
        "--ckpts",
        nargs="+",
        default=[f"{m}:{s}" for m, s in CKPTS],
        metavar="MODEL:STEP",
    )
    args = a.parse_args()
    ckpts = [(c.split(":")[0], int(c.split(":")[1])) for c in args.ckpts]
    ckpts = [(m, s) for m, s in ckpts if (OUT / f"responses__{m}.jsonl").exists()]
    resp = {
        m: [json.loads(line) for line in open(OUT / f"responses__{m}.jsonl")]
        for m, _ in ckpts
    }
    S = {(m, j): scores(m, j) for m, _ in ckpts for j in args.judges}
    base = ckpts[0][0]

    L = [
        "# Checkpoint curve on medqa_test_1k (n=1000, greedy)",
        "",
        f"Training judge: `{args.judges[0]}`. Gold judges: {', '.join(f'`{j}`' for j in args.judges[1:])}.",
        "",
    ]

    L += [
        "## Mean score (0-1) per checkpoint",
        "",
        "| step | " + " | ".join(args.judges) + " |",
        "|---|" + "---|" * len(args.judges),
    ]
    for m, s in ckpts:
        L.append(
            f"| {s} ({m}) | "
            + " | ".join(
                "–" if S[(m, j)] is None else f"{np.nanmean(S[(m, j)]):.3f}"
                for j in args.judges
            )
            + " |"
        )
    L.append("")

    L += [
        "## Gain over base and transfer ratio (gold gain / training gain)",
        "",
        "| step | "
        + " | ".join(f"Δ {j.split(':')[1]}" for j in args.judges)
        + " | "
        + " | ".join(f"transfer {j.split(':')[1]}" for j in args.judges[1:])
        + " |",
        "|---|" + "---|" * (2 * len(args.judges) - 1),
    ]
    for m, s in ckpts[1:]:
        d = {}
        for j in args.judges:
            b, f = S[(base, j)], S[(m, j)]
            d[j] = (
                np.nan
                if b is None or f is None
                else float(np.nanmean(f) - np.nanmean(b))
            )
        t = d[args.judges[0]]
        L.append(
            f"| {s} | "
            + " | ".join(f"{d[j]:+.3f}" for j in args.judges)
            + " | "
            + " | ".join(
                f"{d[j] / t:.2f}" if t and not np.isnan(d[j]) else "–"
                for j in args.judges[1:]
            )
            + " |"
        )
    L.append("")

    L += [
        "## Wrong-answer rate (score <= 3) and 9-10 rate",
        "",
        "| step | "
        + " | ".join(f"wrong {j.split(':')[1]}" for j in args.judges)
        + " | "
        + " | ".join(f"9-10 {j.split(':')[1]}" for j in args.judges)
        + " |",
        "|---|" + "---|" * (2 * len(args.judges)),
    ]
    for m, s in ckpts:
        L.append(
            f"| {s} | "
            + " | ".join(
                "–" if S[(m, j)] is None else f"{np.nanmean(S[(m, j)] <= 0.3):.1%}"
                for j in args.judges
            )
            + " | "
            + " | ".join(
                "–" if S[(m, j)] is None else f"{np.nanmean(S[(m, j)] >= 0.9):.1%}"
                for j in args.judges
            )
            + " |"
        )
    L.append("")

    L += [
        "## Surface statistics",
        "",
        "| step | tokens mean | truncated | has final answer | final answer repeated back-to-back |",
        "|---|---|---|---|---|",
    ]
    for m, s in ckpts:
        R = resp[m]
        toks = np.mean([r["completion_tokens"] or 0 for r in R])
        L.append(
            f"| {s} | {toks:.0f} | {np.mean([r['finish_reason'] == 'length' for r in R]):.1%} | "
            f"{np.mean([bool(_FINAL.search(r['response'])) for r in R]):.1%} | {np.mean([bool(_DUP.search(r['response'])) for r in R]):.1%} |"
        )
    L.append("")

    # training-judge over-credit: 9-10 by training judge but <=3 by the gold judge
    L += [
        "## Training-judge over-credit: responses scored 9-10 by the training judge but <= 3 by a gold judge",
        "",
        "| step | training 9-10s | "
        + " | ".join(f"gold-wrong per {j.split(':')[1]}" for j in args.judges[1:])
        + " |",
        "|---|---|" + "---|" * (len(args.judges) - 1),
    ]
    for m, s in ckpts:
        t = S[(m, args.judges[0])]
        if t is None:
            continue
        cells = []
        for j in args.judges[1:]:
            g = S[(m, j)]
            cells.append(
                "–" if g is None else str(int(np.nansum((t >= 0.9) & (g <= 0.3))))
            )
        L.append(f"| {s} | {int(np.nansum(t >= 0.9))} | " + " | ".join(cells) + " |")
    L.append("")

    (OUT / "CURVE.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"-> {OUT / 'CURVE.md'}")


if __name__ == "__main__":
    main()
