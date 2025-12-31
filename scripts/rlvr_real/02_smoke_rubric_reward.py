#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oumi.datasets.grpo.rewards import rubric_reward  # noqa: E402


def _read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _load_completions(path: Path, limit: int) -> list[str]:
    rows = _read_rows(path, limit)
    completions = []
    for row in rows:
        completion = row.get("completion")
        if completion is None:
            raise ValueError(f"Missing completion in {path}")
        completions.append(str(completion))
    return completions


def _run(
    *,
    mode: str,
    completions: list[str],
    prompts: list[str],
    rubrics: list[list],
    system_prompts: list[str],
    judge_model: str,
) -> None:
    group_rubrics = mode == "group"
    rewards = rubric_reward(
        completions=[[{"content": c}] for c in completions],
        prompts=prompts,
        rubrics=rubrics,
        system_prompt=system_prompts,
        judge_model=judge_model,
        group_rubrics=group_rubrics,
    )

    label = "group_rubrics=True" if group_rubrics else "group_rubrics=False"
    print(f"\nResults ({label})")
    for idx, reward in enumerate(rewards):
        rubric_count = len(rubrics[idx]) if idx < len(rubrics) else 0
        print(f"{idx}: reward={reward:.4f} rubrics={rubric_count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test rubric_reward with real judge calls."
    )
    parser.add_argument(
        "--data",
        default="configs/examples/grpo_rlvr/sample_data_weighted.jsonl",
        help="Path to weighted rubric jsonl.",
    )
    parser.add_argument("--rows", type=int, default=1, help="Rows to evaluate.")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Judge model name.",
    )
    parser.add_argument(
        "--completion",
        default="This is a short placeholder answer for the rubric judge smoke test.",
        help="Fallback completion text (ignored if --completions-file is set).",
    )
    parser.add_argument(
        "--completions-file",
        help="Optional jsonl file with a 'completion' field per row.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "per", "group"),
        default="both",
        help="Run per-rubric, grouped, or both.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    rows = _read_rows(data_path, args.rows)
    prompts = [row.get("prompt", "") for row in rows]
    rubrics = [row.get("rubrics", []) for row in rows]
    system_prompts = [row.get("system_prompt", "") for row in rows]

    if args.completions_file:
        completions = _load_completions(Path(args.completions_file), args.rows)
    else:
        completions = [args.completion for _ in rows]

    if args.mode in ("both", "per"):
        _run(
            mode="per",
            completions=completions,
            prompts=prompts,
            rubrics=rubrics,
            system_prompts=system_prompts,
            judge_model=args.judge_model,
        )
    if args.mode in ("both", "group"):
        _run(
            mode="group",
            completions=completions,
            prompts=prompts,
            rubrics=rubrics,
            system_prompts=system_prompts,
            judge_model=args.judge_model,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
