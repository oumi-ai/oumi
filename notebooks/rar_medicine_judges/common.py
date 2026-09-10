"""Shared helpers for the RaR-Medicine LLM-judge experiments.

Data model
----------
A *rollout record* (one line of ``rollouts/<policy>.jsonl``) is::

    {
      "prompt_idx": int,        # index into the sampled prompt set
      "gen_idx": int,           # 0..n-1 = policy samples; -1 = reference answer
                                #   control; -2 = mismatched-reference control
      "kind": "policy" | "reference" | "mismatch",
      "question": str, "reference_answer": str,
      "rubric": [{"title","description","weight"}], "question_source": str,
      "response": str, "finish_reason": str | None,
      "completion_tokens": int | None,
    }

A *judge record* (one line of ``judge_outputs/<policy>/<variant>__<model>.jsonl``),
aligned line-by-line with the rollout file, is::

    {"score": float | None,      # normalised to [0, 1]; None = judge failed
     "raw_value": ...,           # the un-normalised judgment (int, enum, verdict list)
     "explanation": str | None, "raw_output": str,
     "latency_s": float, "prompt_tokens": int | None, "completion_tokens": int | None}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROLLOUTS_DIR = HERE / "rollouts"
JUDGE_OUTPUTS_DIR = HERE / "judge_outputs"
JUDGE_PROMPTS_DIR = HERE / "judge_prompts"
RESULTS_DIR = HERE / "results"

DATASET_ID = "anisha2102/RaR-Medicine"

# System prompt for the *policy* (the model being judged). The rubrics
# frequently require an explicit "The final answer is ..." statement, so we
# ask for one, mirroring the RaR setup.
POLICY_SYSTEM_PROMPT = (
    "You are a medical expert. Answer the question accurately and concisely, "
    "explaining the key reasoning. Finish with one sentence of the form "
    '"The final answer is ...".'
)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path | str, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def format_rubric(rubric: list[dict[str, Any]], numbered: bool = True) -> str:
    """Render rubric items as a numbered list with their weights.

    Item descriptions already start with their category ("Essential Criteria:",
    "Important Criteria:", "Optional Criteria:", "Pitfall Criteria:").
    """
    lines = []
    for i, item in enumerate(rubric, start=1):
        prefix = f"{i}. " if numbered else "- "
        lines.append(f"{prefix}[weight {item['weight']:+d}] {item['description']}")
    return "\n".join(lines)


def positive_weight_total(rubric: list[dict[str, Any]]) -> float:
    return float(sum(max(0, int(item["weight"])) for item in rubric))


def explicit_rubric_score(rubric: list[dict[str, Any]], met: list[bool]) -> float:
    """RaR-style explicit aggregation, normalised to [0, 1].

    ``met[i]`` is whether the response *satisfies criterion i as written*.
    Positive-weight items add their weight when met. Pitfall items (negative
    weight, phrased as "Does not ...") subtract |weight| when *not* met, i.e.
    when the pitfall is committed. The result is divided by the sum of positive
    weights and clipped to [0, 1].
    """
    total = 0.0
    for item, ok in zip(rubric, met):
        w = int(item["weight"])
        if w >= 0:
            total += w if ok else 0.0
        else:
            total += 0.0 if ok else w  # committed pitfall -> negative
    denom = positive_weight_total(rubric) or 1.0
    return min(1.0, max(0.0, total / denom))


def judge_output_path(policy_name: str, variant: str, model: str, rep: int = 0) -> Path:
    model_slug = model.replace("/", "_")
    suffix = f"__rep{rep}" if rep else ""
    return JUDGE_OUTPUTS_DIR / policy_name / f"{variant}__{model_slug}{suffix}.jsonl"
