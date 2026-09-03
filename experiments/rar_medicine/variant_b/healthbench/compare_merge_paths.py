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

"""Compares the merged-bf16 and runtime-adapter serving paths.

The merged checkpoint loses ~18% of the LoRA delta norm to bf16 rounding, so the
two paths serve different policies. This reports whether that difference is
visible in the HealthBench consolidated score, and -- first -- whether the
runtime adapter is doing anything at all, which is the failure vLLM 0.19.1
exhibits silently for Gemma-4.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _rows(path: Path, key: str = "prompt_id") -> dict[str, dict]:
    with path.open() as handle:
        return {str(json.loads(line)[key]): json.loads(line) for line in handle}


def _responses(path: Path) -> dict[str, str]:
    out = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[str(row["metadata"]["prompt_id"])] = row["messages"][-1]["content"]
    return out


def _paired(
    delta: np.ndarray, boots: int, seed: int
) -> tuple[float, float, float, float]:
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (boots, len(delta)))].mean(axis=1)
    return (
        float(delta.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        float(delta.mean() / se) if se else 0.0,
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-base", default=str(HERE / "artifacts/native_base"))
    parser.add_argument(
        "--native-trained", default=str(HERE / "artifacts/native_trained")
    )
    parser.add_argument("--merged-base", default=str(HERE / "artifacts/global_base"))
    parser.add_argument(
        "--merged-trained", default=str(HERE / "artifacts/global_trained")
    )
    parser.add_argument("--boots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nb, nt = Path(args.native_base), Path(args.native_trained)

    print("=" * 72)
    print("1. IS THE RUNTIME ADAPTER ACTUALLY DOING ANYTHING?")
    print("=" * 72)
    rb = _responses(nb / "model_responses.jsonl")
    rt = _responses(nt / "model_responses.jsonl")
    ids = sorted(set(rb) & set(rt))
    identical = sum(rb[i] == rt[i] for i in ids)
    rng = random.Random(args.seed)
    sample = rng.sample(ids, min(250, len(ids)))
    ratio = [difflib.SequenceMatcher(None, rb[i], rt[i]).ratio() for i in sample]
    print(f"paired prompts                : {len(ids)}")
    print(
        f"identical responses           : {identical}/{len(ids)} = {identical / len(ids):.1%}"
    )
    print(f"median char similarity        : {np.median(ratio):.3f}")
    if identical == len(ids):
        print("\n  *** ADAPTER IS A NO-OP -- results below are meaningless. ***")
        return
    print("  -> adapter is active (a no-op path would be 100% identical)")

    print()
    print("=" * 72)
    print("2. CONSOLIDATED SCORE, RUNTIME ADAPTER (unmerged) vs MERGED bf16")
    print("=" * 72)
    arms = {
        "NATIVE + adapter (unmerged)": (nb, nt),
        "vLLM + merged bf16": (Path(args.merged_base), Path(args.merged_trained)),
    }
    results = {}
    for label, (bdir, tdir) in arms.items():
        try:
            b = _rows(bdir / "sample_results.jsonl")
            t = _rows(tdir / "sample_results.jsonl")
        except FileNotFoundError:
            print(f"{label}: results missing, skipped")
            continue
        common = sorted(set(b) & set(t))
        bs = np.array([b[p]["score"] for p in common])
        ts = np.array([t[p]["score"] for p in common])
        mean, lo, hi, tv = _paired(ts - bs, args.boots, args.seed)
        results[label] = (common, bs, ts)
        print(
            f"{label:30s} n={len(common):4d}  base {bs.mean():.4f}  "
            f"trained {ts.mean():.4f}  delta {mean:+.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  t {tv:+.2f}"
        )

    if len(results) == 2:
        (ca, ba, ta), (cb, bb, tb) = results.values()
        shared = sorted(set(ca) & set(cb))
        if shared:
            ia = {p: i for i, p in enumerate(ca)}
            ib = {p: i for i, p in enumerate(cb)}
            da = np.array([ta[ia[p]] - ba[ia[p]] for p in shared])
            db = np.array([tb[ib[p]] - bb[ib[p]] for p in shared])
            mean, lo, hi, tv = _paired(da - db, args.boots, args.seed)
            print()
            print(
                f"difference of the two deltas on {len(shared)} shared prompts: "
                f"{mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
            )
            print(
                "  -> "
                + (
                    "the serving path changes the measured effect."
                    if not (lo <= 0 <= hi)
                    else "the serving path does not measurably change the effect."
                )
            )

    print()
    print("=" * 72)
    print("3. PER-AXIS, RUNTIME ADAPTER")
    print("=" * 72)
    import make_global_report as M

    rubric = json.loads((HERE / "global_rubric_v2.json").read_text())
    b = _rows(nb / "sample_results.jsonl")
    t = _rows(nt / "sample_results.jsonl")
    common = sorted(set(b) & set(t))
    for axis in M.AXIS_ORDER:
        ab = np.array([M._axis_score(b[p]["grades"], rubric, axis) for p in common])
        at = np.array([M._axis_score(t[p]["grades"], rubric, axis) for p in common])
        mean, lo, hi, tv = _paired(at - ab, args.boots, args.seed)
        flag = "  *" if not (lo <= 0 <= hi) else ""
        print(
            f"{axis:24s} base {ab.mean():.4f}  trained {at.mean():.4f}  "
            f"delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  t {tv:+.2f}{flag}"
        )


if __name__ == "__main__":
    main()
