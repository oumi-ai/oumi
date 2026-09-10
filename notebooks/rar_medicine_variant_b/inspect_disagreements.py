"""List responses where the training judge (gpt-4.1-mini) scores high but
strong held-out judges score low, with every judge's explanation side by side.

    python inspect_disagreements.py                       # fullft, training>=0.9, strict median<=0.7
    python inspect_disagreements.py --model base
    python inspect_disagreements.py --hi 0.9 --lo 0.6 --top 30

Writes results/test1k/DISAGREEMENTS__<model>.md and .jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "test1k"
TRAIN = "variant_b_training_reward__gpt-4.1-mini"
STRICT = {
    "gpt-5.5": "fixed_rubric_implicit__gpt-5.5",
    "gpt-5.6-terra": "fixed_rubric_implicit__gpt-5.6-terra",
    "claude-sonnet-5": "fixed_rubric_implicit__claude-sonnet-5",
    "claude-opus-5": "fixed_rubric_implicit__claude-opus-5",
}
OTHERS = {
    "gpt-4.1": "fixed_rubric_implicit__gpt-4.1",
    "gemini-2.5-pro": "fixed_rubric_implicit__gemini-2.5-pro",
    "gemma-4-E4B": "fixed_rubric_implicit__google_gemma-4-E4B-it",
    "Qwen3-4B": "fixed_rubric_implicit__Qwen_Qwen3-4B-Instruct-2507",
}


def rj(path):
    return [json.loads(line) for line in open(path)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fullft")
    p.add_argument("--hi", type=float, default=0.9, help="training judge score >= hi")
    p.add_argument(
        "--lo", type=float, default=0.7, help="median strict-judge score <= lo"
    )
    p.add_argument(
        "--top", type=int, default=40, help="cases to write in full to the .md"
    )
    a = p.parse_args()

    resp = rj(OUT / f"responses__{a.model}.jsonl")
    train = rj(OUT / f"judge__{a.model}__{TRAIN}.jsonl")
    judges = {
        n: rj(OUT / f"judge__{a.model}__{j}.jsonl")
        for n, j in {**STRICT, **OTHERS}.items()
    }

    rows = []
    for i, (r, t) in enumerate(zip(resp, train)):
        strict = [
            judges[n][i]["score"] for n in STRICT if judges[n][i]["score"] is not None
        ]
        if t["score"] is None or not strict:
            continue
        med = float(np.median(strict))
        if t["score"] >= a.hi and med <= a.lo:
            rows.append(
                {
                    "i": i,
                    "idx": r["idx"],
                    "question": r["question"],
                    "reference_answer": r["reference_answer"],
                    "response": r["response"],
                    "finish_reason": r["finish_reason"],
                    "chars": len(r["response"]),
                    "training_score": t["score"],
                    "strict_median": med,
                    "gap": t["score"] - med,
                    "judges": {
                        "gpt-4.1-mini (training)": {
                            "score": t["score"],
                            "explanation": t["explanation"],
                        },
                        **{
                            n: {
                                "score": judges[n][i]["score"],
                                "explanation": judges[n][i]["explanation"],
                            }
                            for n in {**STRICT, **OTHERS}
                        },
                    },
                }
            )
    rows.sort(key=lambda x: -x["gap"])
    print(
        f"{a.model}: {len(rows)} responses with training >= {a.hi} and strict-median <= {a.lo} "
        f"(of {sum(t['score'] is not None and t['score'] >= a.hi for t in train)} scored >= {a.hi} by the training judge)"
    )

    with open(OUT / f"DISAGREEMENTS__{a.model}.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    L = [
        f"# {a.model}: training judge >= {a.hi}, median of strict judges (gpt-5.5, gpt-5.6-terra, sonnet-5, opus-5) <= {a.lo}",
        "",
        f"{len(rows)} cases, sorted by gap. Top {min(a.top, len(rows))} shown in full.",
        "",
    ]
    L += [
        "| # | idx | training | strict median | gap | chars | finish |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, x in enumerate(rows[: a.top], 1):
        L.append(
            f"| {k} | {x['idx']} | {x['training_score']:.1f} | {x['strict_median']:.2f} | {x['gap']:+.2f} | {x['chars']} | {x['finish_reason']} |"
        )
    L.append("")
    for k, x in enumerate(rows[: a.top], 1):
        L += [
            "---",
            f"## {k}. idx {x['idx']}  (training {x['training_score']:.1f} vs strict median {x['strict_median']:.2f})",
            "",
            "**Question**",
            "",
            x["question"].strip(),
            "",
            "**Reference answer**",
            "",
            x["reference_answer"].strip(),
            "",
            f"**Response** ({x['chars']} chars, finish={x['finish_reason']})",
            "",
            "```",
            x["response"].strip(),
            "```",
            "",
            "**Judges**",
            "",
            "| judge | score | explanation |",
            "|---|---|---|",
        ]
        for n, v in x["judges"].items():
            s = "—" if v["score"] is None else f"{v['score'] * 10:.0f}/10"
            e = (v["explanation"] or "").replace("\n", " ").replace("|", "/")
            L.append(f"| {n} | {s} | {e} |")
        L.append("")
    (OUT / f"DISAGREEMENTS__{a.model}.md").write_text("\n".join(L))
    print(f"-> {OUT / f'DISAGREEMENTS__{a.model}.md'}")


if __name__ == "__main__":
    main()
