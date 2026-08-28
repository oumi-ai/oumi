"""Summarize / compare the `oumi infer` output files in output/rar_medicine_grpo_verl_variant_b/eval/outputs/.

Per file: row count, response-token stats, how many responses hit the 1024-token
cap, how many contain "final answer". Across files: how many trained responses are
byte-identical to the base response for the same conversation_id.

Truncation is recomputed from the response's token count (>= --cap tokens) rather
than read from `metadata.finish_reason`: oumi's NATIVE engine labels every
sequence in a batch `length` when any member of the batch runs to the cap (the
2026-08-27 trained run had 640 `length` labels for 12 real truncations). The
vLLM-produced base file's labels are correct.

Usage:
    python summarize_outputs.py [--fix-finish-reason]

--fix-finish-reason rewrites each file's `metadata.finish_reason` from the token
count, keeping the engine's original value as `metadata.finish_reason_engine`
(no-op where they already agree) and fills `metadata.usage.completion_tokens`
when missing.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUT_DIR = _REPO_ROOT / "output/rar_medicine_grpo_verl_variant_b/eval/outputs"
_BASE_FILE = "base_gemma4_e2b_it.jsonl"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _response(row: dict) -> str:
    last = row["messages"][-1]
    return (last.get("content") or "") if last.get("role") == "assistant" else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--cap", type=int, default=1024, help="max_new_tokens used for the runs"
    )
    ap.add_argument("--fix-finish-reason", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    files = sorted(p for p in _OUT_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no outputs in {_OUT_DIR}")

    per_file: dict[str, dict[str, str]] = {}
    for path in files:
        rows = _load(path)
        texts = [_response(r) for r in rows]
        ntok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        # Retokenizing decoded text can be off by a token or two from the count
        # the engine generated; anything within 4 tokens of the cap is a truncation.
        truncated = [n >= args.cap - 4 for n in ntok]
        engine_labels = Counter(r["metadata"].get("finish_reason") for r in rows)
        print(
            f"{path.name}: {len(rows)} rows | response tokens mean {statistics.mean(ntok):.0f}, "
            f"median {statistics.median(ntok):.0f}, max {max(ntok)} | hit {args.cap}-token cap: "
            f"{sum(truncated)} | engine finish_reason labels: {dict(engine_labels)} | "
            f"'final answer' in response: {sum('final answer' in t.lower() for t in texts)} | "
            f"empty: {sum(not t.strip() for t in texts)}"
        )
        per_file[path.name] = {r["conversation_id"]: t for r, t in zip(rows, texts)}

        if args.fix_finish_reason:
            changed = 0
            for r, n, trunc in zip(rows, ntok, truncated):
                md = r.setdefault("metadata", {})
                want = "length" if trunc else "stop"
                if md.get("finish_reason") != want:
                    md.setdefault("finish_reason_engine", md.get("finish_reason"))
                    md["finish_reason"] = want
                    changed += 1
                usage = md.get("usage") or {}
                if "completion_tokens" not in usage:
                    usage["completion_tokens"] = n
                    md["usage"] = usage
            if changed:
                with path.open("w") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  finish_reason relabelled on {changed} rows")

    base = per_file.get(_BASE_FILE)
    if base:
        for name, texts in per_file.items():
            if name == _BASE_FILE:
                continue
            common = [cid for cid in texts if cid in base]
            identical = sum(texts[cid] == base[cid] for cid in common)
            print(
                f"{name} vs {_BASE_FILE}: {identical}/{len(common)} responses byte-identical"
            )


if __name__ == "__main__":
    main()
