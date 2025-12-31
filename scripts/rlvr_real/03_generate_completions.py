#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oumi.core.configs import (  # noqa: E402
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)
from oumi.infer import infer  # noqa: E402


def _read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "content", None)
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate completions for RLVR rubric smoke testing."
    )
    parser.add_argument(
        "--data",
        default="configs/examples/grpo_rlvr/sample_data_weighted.jsonl",
        help="Path to weighted rubric jsonl.",
    )
    parser.add_argument(
        "--output",
        default="output/rlvr_real/completions.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument("--rows", type=int, default=1, help="Rows to generate.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    rows = _read_rows(Path(args.data), args.rows)
    prompts = []
    for row in rows:
        prompt = row.get("prompt", "")
        system_prompt = row.get("system_prompt", "")
        if system_prompt:
            prompt = f"[System: {system_prompt}]\n\n{prompt}"
        prompts.append(prompt)

    config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=ModelParams(model_name=args.model),
        generation=GenerationParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
    )

    generations = infer(config=config, inputs=prompts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for row, convo in zip(rows, generations):
            completion = _extract_text(convo.messages[-1].content)
            payload = dict(row)
            payload["completion"] = completion
            payload["completion_model"] = args.model
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} completions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
