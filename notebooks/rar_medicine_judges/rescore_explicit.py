"""Re-score an explicit-rubric run from its stored per-item verdicts.

The judge's verdicts are the expensive part; the aggregation is arithmetic. This
rewrites the `score` field of an existing run under a different aggregation (for
example with the `cap_if_unmet` rule disabled, or with a weight class dropped)
and saves it as a new run, so aggregation choices can be compared without
re-calling any judge.

    python rescore_explicit.py --policy gemma-4-E2B-it \
        --variant fixed_rubric_explicit --suffix nocap --no-cap
"""

from __future__ import annotations

import argparse

from common import (
    JUDGE_OUTPUTS_DIR,
    ROLLOUTS_DIR,
    explicit_rubric_score,
    read_jsonl,
    write_jsonl,
)
from run_judges import apply_caps, load_variant, rubric_for


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument(
        "--suffix",
        required=True,
        help="Name of the rescored variant: <variant>_<suffix>.",
    )
    p.add_argument(
        "--no-cap", action="store_true", help="Ignore the variant's cap_if_unmet rules."
    )
    p.add_argument(
        "--drop-weights",
        type=int,
        nargs="*",
        default=[],
        help="Weights to exclude from scoring, e.g. 1 2 to drop Optional items.",
    )
    args = p.parse_args()

    variant = load_variant(args.variant)
    if args.no_cap:
        variant = {**variant, "cap_if_unmet": {}}
    rollouts = read_jsonl(ROLLOUTS_DIR / f"{args.policy}.jsonl")

    for path in sorted(
        (JUDGE_OUTPUTS_DIR / args.policy).glob(f"{args.variant}__*.jsonl")
    ):
        if "__limit" in path.stem:
            continue
        recs = read_jsonl(path)
        out = []
        for rec, rr in zip(rollouts, recs):
            rubric = rubric_for(variant, rec)
            met = rr.get("raw_value")
            new = dict(rr)
            if isinstance(met, list) and len(met) == len(rubric):
                keep = [
                    i
                    for i, it in enumerate(rubric)
                    if int(it["weight"]) not in args.drop_weights
                ]
                sub = [rubric[i] for i in keep]
                sub_met = [bool(met[i]) for i in keep]
                new["score"] = apply_caps(
                    variant, sub, sub_met, explicit_rubric_score(sub, sub_met)
                )
            out.append(new)
        new_stem = path.stem.replace(args.variant, f"{args.variant}_{args.suffix}", 1)
        dest = path.with_name(new_stem + ".jsonl")
        write_jsonl(dest, out)
        print(f"{path.name} -> {dest.name}")


if __name__ == "__main__":
    main()
