"""Score every rollout with one or more LLM judges.

Each (prompt variant, judge model) pair produces one JSONL under
``judge_outputs/<policy>/`` aligned line-by-line with the rollout file. Runs are
cached: existing output files are skipped unless ``--overwrite``.

The judge is driven directly through Oumi's inference engines with JSON-schema
guided decoding (the same mechanism ``SimpleJudge`` uses), so one engine can be
shared across all prompt variants. ``export_simple_judge_config`` converts a
single-score variant into a ``SimpleJudge`` YAML for use as a GRPO reward.

Examples::

    # OpenAI judges, all variants
    python run_judges.py --policy gemma-4-E2B-it \
        --models gpt-4o-mini gpt-4.1-mini gpt-4.1 --engine OPENAI

    # Local judge on GPU 2
    CUDA_VISIBLE_DEVICES=2 python run_judges.py --policy gemma-4-E2B-it \
        --models Qwen/Qwen3-4B-Instruct-2507 --engine VLLM

    # Repeat a run to measure self-consistency
    python run_judges.py ... --rep 1
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml
from common import (
    JUDGE_PROMPTS_DIR,
    ROLLOUTS_DIR,
    explicit_rubric_score,
    format_rubric,
    judge_output_path,
    read_jsonl,
    write_jsonl,
)

# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #


def load_variant(name: str) -> dict[str, Any]:
    path = JUDGE_PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        v = yaml.safe_load(f)
    v.setdefault("name", name)
    v.setdefault("include_explanation", False)
    return v


def list_variants() -> list[str]:
    return sorted(p.stem for p in JUDGE_PROMPTS_DIR.glob("*.yaml"))


def rubric_for(variant: dict[str, Any], rec: dict[str, Any]) -> list[dict[str, Any]]:
    """The rubric this variant grades against.

    Most variants use the dataset's per-question rubric. A variant that declares
    `fixed_rubric` uses that same list for every record, which is what makes it
    reusable on datasets with no rubrics of their own.
    """
    return variant.get("fixed_rubric") or rec["rubric"]


def apply_caps(
    variant: dict[str, Any], rubric: list[dict[str, Any]], met: list[bool], score: float
) -> float:
    """Enforce `cap_if_unmet`: {item_id: max_score} when that item is not met.

    Weighted sums are linear, so a rule like "a wrong final answer cannot score
    above 3/10" cannot live in the weights; it is applied here instead.
    """
    caps = variant.get("cap_if_unmet") or {}
    if not caps:
        return score
    by_id = {
        str(item.get("id", i + 1)): ok for i, (item, ok) in enumerate(zip(rubric, met))
    }
    for item_id, cap in caps.items():
        if by_id.get(str(item_id)) is False:
            score = min(score, float(cap))
    return score


def build_schema(variant: dict[str, Any], rubric_len: int, explanation: bool) -> dict:
    """JSON schema for the judge's reply. Explanation (if any) comes first so the
    model reasons before committing to a verdict.
    """
    props: dict[str, Any] = {}
    if explanation:
        props["explanation"] = {"type": "string"}

    out = variant["output"]
    if out == "int_0_10":
        props["score"] = {"type": "integer"}  # 0-10 enforced by prompt + parser
    elif out == "enum":
        props["label"] = {"type": "string", "enum": list(variant["scores"].keys())}
    elif out in ("explicit_rubric", "explicit_rubric_evidence"):
        fixed = variant.get("fixed_rubric")
        if fixed:
            rubric_len = len(fixed)
            item_schema: dict[str, Any] = {
                "type": "string",
                "enum": [str(item["id"]) for item in fixed],
            }
        else:
            item_schema = {"type": "integer"}
        item_props: dict[str, Any] = {"item": item_schema}
        if out == "explicit_rubric_evidence":
            item_props["evidence"] = {"type": "string"}
        item_props["met"] = {"type": "boolean"}
        props["verdicts"] = {
            "type": "array",
            "minItems": rubric_len,
            "maxItems": rubric_len,
            "items": {
                "type": "object",
                "properties": item_props,
                "required": list(item_props.keys()),
                "additionalProperties": False,
            },
        }
    else:
        raise ValueError(f"Unknown output type {out!r}")

    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


def format_suffix(variant: dict[str, Any], explanation: bool) -> str:
    """Plain-language description of the JSON reply, appended to the system
    prompt (guided decoding enforces it, but the model should also be told).
    """
    out = variant["output"]
    if out == "int_0_10":
        body = '"score": an integer from 0 to 10'
    elif out == "enum":
        opts = ", ".join(f'"{k}"' for k in variant["scores"])
        body = f'"label": one of {opts}'
    elif out == "explicit_rubric_evidence":
        body = (
            '"verdicts": a list with one object per rubric item, in order, each '
            'of the form {"item": <item number>, "evidence": "<verbatim quote or none>", '
            '"met": true|false}'
        )
    elif variant.get("fixed_rubric"):
        ids = ", ".join(str(i["id"]) for i in variant["fixed_rubric"])
        body = (
            '"verdicts": a list with one object per criterion, in the order '
            f"{ids}, each of the form "
            '{"item": "<criterion id>", "met": true|false}'
        )
    else:
        body = (
            '"verdicts": a list with one object per rubric item, in order, each '
            'of the form {"item": <item number>, "met": true|false}'
        )
    if explanation:
        return (
            "\n\nReply with a JSON object with two keys, in this order: "
            '"explanation": a brief justification (2-4 sentences) referencing the '
            f"relevant criteria; then {body}."
        )
    return f"\n\nReply with a JSON object with exactly one key: {body}."


def build_user_prompt(variant: dict[str, Any], rec: dict[str, Any]) -> str:
    """Fill the variant's template.

    A fixed-rubric variant carries its criteria in the system instruction, so its
    template has no {rubric} placeholder and none is rendered.
    """
    fields = {
        "question": rec["question"],
        "response": rec["response"],
        "reference_answer": rec["reference_answer"],
        "rubric": format_rubric(rubric_for(variant, rec)),
    }
    return variant["prompt_template"].format(**fields)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_output(
    variant: dict[str, Any], raw: str, rubric: list[dict[str, Any]]
) -> tuple[float | None, Any, str | None]:
    """Returns (normalised score in [0,1] or None, raw value, explanation)."""
    text = _strip_thinking(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None, None, None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, None, None
    if not isinstance(data, dict):
        return None, None, None

    explanation = data.get("explanation")
    out = variant["output"]
    try:
        if out == "int_0_10":
            v = int(data["score"])
            if not 0 <= v <= 10:
                return None, v, explanation
            return v / 10.0, v, explanation
        if out == "enum":
            label = str(data["label"])
            if label not in variant["scores"]:
                return None, label, explanation
            return float(variant["scores"][label]), label, explanation
        if out in ("explicit_rubric", "explicit_rubric_evidence"):
            verdicts = data["verdicts"]
            if len(verdicts) != len(rubric):
                return None, verdicts, explanation
            # Trust order, but honour the item label when it is valid: an
            # integer position for dataset rubrics, an id string for fixed ones.
            id_to_pos = {
                str(item["id"]): i for i, item in enumerate(rubric) if "id" in item
            }
            met = [False] * len(rubric)
            for pos, vd in enumerate(verdicts):
                label = vd.get("item", pos + 1)
                if str(label) in id_to_pos:
                    idx = id_to_pos[str(label)]
                else:
                    try:
                        idx = int(label) - 1
                    except (TypeError, ValueError):
                        idx = pos
                    if not 0 <= idx < len(rubric):
                        idx = pos
                met[idx] = bool(vd["met"])
            score = apply_caps(variant, rubric, met, explicit_rubric_score(rubric, met))
            return score, met, explanation
    except (KeyError, TypeError, ValueError):
        return None, data, explanation
    return None, data, explanation


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #


def build_engine(engine_name: str, model: str, args: argparse.Namespace):
    from oumi.builders.inference_engines import build_inference_engine
    from oumi.core.configs import InferenceEngineType, ModelParams, RemoteParams

    engine_type = InferenceEngineType(engine_name)
    if engine_type == InferenceEngineType.VLLM:
        from oumi.inference import VLLMInferenceEngine

        return VLLMInferenceEngine(
            ModelParams(
                model_name=model,
                torch_dtype_str="bfloat16",
                model_max_length=args.vllm_max_model_len,
            ),
            gpu_memory_utilization=args.vllm_gpu_mem,
            enforce_eager=False,
        )
    remote = RemoteParams(
        num_workers=args.num_workers,
        politeness_policy=0.0,
        max_retries=6,
        use_adaptive_concurrency=False,
    )
    if args.api_url:
        remote.api_url = args.api_url
    return build_inference_engine(engine_type, ModelParams(model_name=model), remote)


def run_variant(
    engine,
    engine_name: str,
    model: str,
    variant: dict[str, Any],
    records: list[dict[str, Any]],
    explanation: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from oumi.core.configs import (
        GenerationParams,
        GuidedDecodingParams,
        InferenceConfig,
        InferenceEngineType,
        ModelParams,
    )
    from oumi.core.types.conversation import Conversation, Message, Role

    system = variant["system_instruction"].rstrip() + format_suffix(
        variant, explanation
    )
    convs = []
    for i, rec in enumerate(records):
        schema = build_schema(variant, len(rubric_for(variant, rec)), explanation)
        convs.append(
            Conversation(
                messages=[
                    Message(role=Role.SYSTEM, content=system),
                    Message(role=Role.USER, content=build_user_prompt(variant, rec)),
                ],
                metadata={"i": i, "schema": schema},
            )
        )

    # Guided decoding is part of GenerationParams, and rubric lengths differ per
    # record, so group conversations by schema and run one batch per schema.
    by_schema: dict[str, list[Conversation]] = {}
    for c in convs:
        by_schema.setdefault(
            json.dumps(c.metadata["schema"], sort_keys=True), []
        ).append(c)

    strict = engine_name != "BEDROCK"
    results: dict[int, Conversation] = {}
    t0 = time.time()
    for schema_key, group in by_schema.items():
        gen = GenerationParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            guided_decoding=GuidedDecodingParams(
                json=json.loads(schema_key), strict=strict
            ),
        )
        cfg = InferenceConfig(
            model=ModelParams(model_name=model),
            generation=gen,
            engine=InferenceEngineType(engine_name),
        )
        for out in engine.infer(group, inference_config=cfg):
            results[out.metadata["i"]] = out
    elapsed = time.time() - t0

    judge_records = []
    for i, rec in enumerate(records):
        conv = results.get(i)
        msg = conv.last_message(Role.ASSISTANT) if conv else None
        raw = (msg.content if msg and isinstance(msg.content, str) else "") or ""
        score, raw_value, expl = parse_output(variant, raw, rubric_for(variant, rec))
        usage = (conv.metadata or {}).get("usage") or {} if conv else {}
        judge_records.append(
            {
                "score": score,
                "raw_value": raw_value,
                "explanation": expl,
                "raw_output": raw,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "run_wall_s": elapsed,
            }
        )
    return judge_records


def export_simple_judge_config(
    variant: dict[str, Any], model: str, engine_name: str, explanation: bool, path: Path
) -> None:
    """Write a SimpleJudge-compatible JudgeConfig YAML for a single-score variant,
    ready to plug into a GRPO judge reward.
    """
    if variant["output"].startswith("explicit_rubric"):
        raise ValueError(
            "explicit_rubric needs per-item output; SimpleJudge is single-field."
        )
    judge_params: dict[str, Any] = {
        "system_instruction": variant["system_instruction"],
        "prompt_template": variant["prompt_template"],
        "response_format": "JSON",
        "include_explanation": explanation,
    }
    if variant["output"] == "int_0_10":
        judge_params["judgment_type"] = "INT"
    else:
        judge_params["judgment_type"] = "ENUM"
        judge_params["judgment_scores"] = variant["scores"]
    cfg = {
        "judge_params": judge_params,
        "inference_config": {
            "model": {"model_name": model},
            "engine": engine_name,
            "generation": {"max_new_tokens": 1024, "temperature": 0.0},
        },
    }

    def _str(dumper, value):
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
        )

    yaml.SafeDumper.add_representer(str, _str)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy", required=True, help="Rollout file stem, e.g. gemma-4-E2B-it"
    )
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument(
        "--engine",
        default="OPENAI",
        help="OPENAI | VLLM | REMOTE_VLLM | ANTHROPIC | GEMINI ...",
    )
    p.add_argument(
        "--variants", nargs="+", default=None, help="Default: all in judge_prompts/"
    )
    p.add_argument(
        "--explanation",
        choices=["default", "on", "off"],
        default="default",
        help="Override each variant's include_explanation.",
    )
    p.add_argument(
        "--rep", type=int, default=0, help="Repetition id (for self-consistency)."
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--api-url", default=None)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N records (smoke test).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.engine == "VLLM":
        import os

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    records = read_jsonl(ROLLOUTS_DIR / f"{args.policy}.jsonl")
    if args.limit:
        records = records[: args.limit]
    variants = [load_variant(v) for v in (args.variants or list_variants())]

    for model in args.models:
        engine = None
        for variant in variants:
            explanation = {
                "default": variant["include_explanation"],
                "on": True,
                "off": False,
            }[args.explanation]
            vname = variant["name"]
            if args.explanation != "default":
                vname += "_expl" if explanation else "_noexpl"
            out_path = judge_output_path(args.policy, vname, model, args.rep)
            if args.limit:
                out_path = out_path.with_name(
                    out_path.stem + f"__limit{args.limit}.jsonl"
                )
            if out_path.exists() and not args.overwrite:
                print(f"skip (cached): {out_path}")
                continue
            if engine is None:
                engine = build_engine(args.engine, model, args)
            print(f"judging {len(records)} records: variant={vname} model={model}")
            t0 = time.time()
            outs = run_variant(
                engine, args.engine, model, variant, records, explanation, args
            )
            n_fail = sum(o["score"] is None for o in outs)
            write_jsonl(out_path, outs)
            print(f"  -> {out_path}  ({time.time() - t0:.0f}s, {n_fail} unparseable)")


if __name__ == "__main__":
    main()
