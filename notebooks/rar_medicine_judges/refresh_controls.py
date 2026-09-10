"""Rewrite the mismatch (gen_idx=-2) control rows of an existing rollout file
using nearest-neighbour references from the full split. Policy samples and the
reference control are untouched.

    python refresh_controls.py --policy gemma-4-E2B-it --split val
"""

from __future__ import annotations

import argparse

from common import DATASET_ID, ROLLOUTS_DIR, read_jsonl, write_jsonl
from common_controls import nearest_neighbour_map


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--split", default="val")
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=args.split)
    pool = [
        {"dataset_row": i, "question": q, "reference_answer": a}
        for i, (q, a) in enumerate(zip(ds["question"], ds["reference_answer"]))
    ]

    path = ROLLOUTS_DIR / f"{args.policy}.jsonl"
    recs = read_jsonl(path)
    prompts = {r["prompt_idx"]: r for r in recs if r["kind"] == "reference"}
    order = sorted(prompts)
    nn = nearest_neighbour_map([prompts[i] for i in order], pool)
    nn_by_prompt = dict(zip(order, nn))

    n = 0
    for r in recs:
        if r["kind"] != "mismatch":
            continue
        j = nn_by_prompt[r["prompt_idx"]]
        r["response"] = pool[j]["reference_answer"]
        r["mismatch_source_dataset_row"] = pool[j]["dataset_row"]
        r["mismatch_source_question"] = pool[j]["question"]
        r.pop("mismatch_source_prompt_idx", None)
        n += 1
    write_jsonl(path, recs)
    print(f"Rewrote {n} mismatch controls in {path}")
    ex = next(r for r in recs if r["kind"] == "mismatch")
    print(
        "\nExample\n  Q:",
        ex["question"][:200],
        "\n  NN Q:",
        ex["mismatch_source_question"][:200],
    )


if __name__ == "__main__":
    main()
