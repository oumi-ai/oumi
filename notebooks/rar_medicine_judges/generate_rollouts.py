"""Sample GRPO-style rollout groups from a Gemma policy on RaR-Medicine.

For each of ``--num-prompts`` questions (seeded sample from ``--split``) we draw
``--num-generations`` completions at temperature 1.0, exactly like a GRPO
rollout group. Two *control* completions are appended to every group so judges
can be checked without human labels:

* ``gen_idx=-1`` (kind=reference): the dataset's reference answer. A good judge
  should score it near the top of its group.
* ``gen_idx=-2`` (kind=mismatch): the reference answer of the most similar
  *other* question in the split. Fluent, on-topic, confidently wrong. A good judge should score it at
  the bottom of its group.

Usage::

    python generate_rollouts.py --policy google/gemma-4-E2B-it --gpu 0
    python generate_rollouts.py --policy google/gemma-4-E4B-it --gpu 1
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from common import (
    DATASET_ID,
    POLICY_SYSTEM_PROMPT,
    ROLLOUTS_DIR,
    write_jsonl,
)
from common_controls import nearest_neighbour_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="google/gemma-4-E2B-it")
    p.add_argument("--policy-name", default=None, help="Output file stem.")
    p.add_argument("--split", default="val")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    policy_name = args.policy_name or Path(args.policy).name
    out_path = ROLLOUTS_DIR / f"{policy_name}.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to regenerate.")

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=args.split)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.num_prompts)
    prompts = [dict(ds[i], dataset_row=i) for i in indices]
    print(f"Sampled {len(prompts)} prompts from {DATASET_ID}[{args.split}]")

    from oumi.core.configs import GenerationParams, ModelParams
    from oumi.core.types.conversation import Conversation, Message, Role
    from oumi.inference import VLLMInferenceEngine

    conversations = []
    for prompt_idx, p in enumerate(prompts):
        for gen_idx in range(args.num_generations):
            conversations.append(
                Conversation(
                    messages=[
                        Message(role=Role.SYSTEM, content=POLICY_SYSTEM_PROMPT),
                        Message(role=Role.USER, content=p["question"]),
                    ],
                    metadata={"prompt_idx": prompt_idx, "gen_idx": gen_idx},
                )
            )

    engine = VLLMInferenceEngine(
        ModelParams(model_name=args.policy, torch_dtype_str="bfloat16"),
        generation_params=GenerationParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=1.0,
            seed=args.seed,
        ),
    )
    completed = engine.infer(conversations)
    print(f"Generated {len(completed)} completions")

    by_key = {(c.metadata["prompt_idx"], c.metadata["gen_idx"]): c for c in completed}

    # Mismatch control: the reference answer of the most similar *other*
    # question in the split (on-topic but wrong; see common_controls).
    pool = [
        {"dataset_row": i, "question": q, "reference_answer": a}
        for i, (q, a) in enumerate(zip(ds["question"], ds["reference_answer"]))
    ]
    nn = nearest_neighbour_map(prompts, pool)
    records = []
    for prompt_idx, p in enumerate(prompts):
        base = {
            "prompt_idx": prompt_idx,
            "dataset_row": p["dataset_row"],
            "question": p["question"],
            "reference_answer": p["reference_answer"],
            "rubric": p["rubric"],
            "question_source": p["question_source"],
        }
        for gen_idx in range(args.num_generations):
            conv = by_key[(prompt_idx, gen_idx)]
            msg = conv.last_message(Role.ASSISTANT)
            usage = (conv.metadata or {}).get("usage") or {}
            records.append(
                {
                    **base,
                    "gen_idx": gen_idx,
                    "kind": "policy",
                    "response": msg.content if msg else "",
                    "finish_reason": (conv.metadata or {}).get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )
        records.append(
            {
                **base,
                "gen_idx": -1,
                "kind": "reference",
                "response": p["reference_answer"],
                "finish_reason": None,
                "completion_tokens": None,
            }
        )
        other = pool[nn[prompt_idx]]
        records.append(
            {
                **base,
                "gen_idx": -2,
                "kind": "mismatch",
                "response": other["reference_answer"],
                "mismatch_source_dataset_row": other["dataset_row"],
                "mismatch_source_question": other["question"],
                "finish_reason": None,
                "completion_tokens": None,
            }
        )

    write_jsonl(out_path, records)
    n_trunc = sum(
        r["finish_reason"] == "length" for r in records if r["kind"] == "policy"
    )
    print(
        f"Wrote {len(records)} records -> {out_path}  (truncated policy samples: {n_trunc})"
    )


if __name__ == "__main__":
    main()
