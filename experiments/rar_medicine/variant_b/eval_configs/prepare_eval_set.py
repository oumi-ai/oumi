"""Download the RaR-Medicine test split and build the fixed 1000-sample eval set.

Writes (deterministic, so re-runnable):
  experiments/rar_medicine/data/test-00000-of-00001.parquet
      full hub test split (2,242 rows), as downloaded
  output/rar_medicine_grpo_verl_variant_b/eval/test_1000.parquet
      the 1000 sampled rows + `idx` (row position in the hub test split) — keep
      this: it carries the reference answers and per-sample rubrics for judging
  output/rar_medicine_grpo_verl_variant_b/eval/test_1000.jsonl
      the same rows as oumi `Conversation` JSONL, the `input_path` for `oumi infer`

Each conversation uses the exact system prompt the policy was trained with
(imported from ../rar_medicine_grpo.py so the two cannot drift) and carries
`idx`, `question_source` and `reference_answer` in `metadata`, which `oumi infer`
copies into its output rows.

Sampling: `DataFrame.sample(n=1000, random_state=42)` over the hub test split,
then sorted by original position. The test split has no id column and repeated
questions exist, so `idx` is the join key.

Usage:
    python prepare_eval_set.py [--n 1000] [--seed 42]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
_DATA_DIR = _REPO_ROOT / "experiments/rar_medicine/data"  # raw hub parquets
_EVAL_DIR = (
    _REPO_ROOT / "output/rar_medicine_grpo_verl_variant_b/eval"
)  # eval set + infer outputs
_HUB_REPO = "anisha2102/RaR-Medicine"
_HUB_TEST_FILE = "data/test-00000-of-00001.parquet"

# The training system prompt lives next to the dataset/reward registrations.
sys.path.insert(0, str(_HERE.parent))
from rar_medicine_grpo import _SYSTEM_PROMPT  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    eval_dir = _EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    cached = hf_hub_download(_HUB_REPO, _HUB_TEST_FILE, repo_type="dataset")
    local = _DATA_DIR / Path(_HUB_TEST_FILE).name
    shutil.copyfile(cached, local)
    full = pd.read_parquet(local)
    print(f"test split: {len(full)} rows, columns={full.columns.tolist()} -> {local}")

    sample = full.sample(n=args.n, random_state=args.seed).sort_index()
    sample.insert(0, "idx", sample.index.astype(int))
    sample = sample.reset_index(drop=True)
    parquet_path = eval_dir / f"test_{args.n}.parquet"
    sample.to_parquet(parquet_path, index=False)

    jsonl_path = eval_dir / f"test_{args.n}.jsonl"
    with jsonl_path.open("w") as f:
        for row in sample.to_dict(orient="records"):
            idx = int(row["idx"])
            conv = {
                "conversation_id": f"rar_medicine_test_{idx:04d}",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["question"]).strip()},
                ],
                "metadata": {
                    "idx": idx,
                    "question_source": str(row["question_source"]),
                    "reference_answer": str(row["reference_answer"]).strip(),
                },
            }
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(
        f"sampled {len(sample)} rows (seed={args.seed}); sources:\n"
        f"{sample.question_source.value_counts().to_string()}\n"
        f"question chars: median {int(sample.question.str.len().median())}, "
        f"max {int(sample.question.str.len().max())}\n"
        f"-> {parquet_path}\n-> {jsonl_path}"
    )


if __name__ == "__main__":
    main()
