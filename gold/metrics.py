#!/usr/bin/env python3
"""
Countdown accuracy checker for data shaped like:
  - target: int
  - nums: list[int]
  - messages: list[{"role": "...", "content": "..."}]

Each assistant message is treated as an independent completion.

A completion is correct iff:
1) The assistant message contains exactly one <answer>...</answer>.
2) LHS uses only provided nums, each at most once.
3) Expression evaluates exactly to target.
4) If an '=' is present inside <answer>, we also require:
     LHS == RHS == target
   and we do NOT count RHS literals toward number usage.

We compute pass@n over the last n assistant messages:
  pass@n = any of the last n completions is correct.

Examples:
  python3 metrics.py output/qwen2.5_1.5b_baseline.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_baseline_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_baseline_summary.json

  python3 metrics.py output/qwen2.5_1.5b_ckpt1700.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_ckpt1700_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt1700_summary.json
  python3 metrics.py output/qwen2.5_1.5b_ckpt1700.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_ckpt1700_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt1700_summary.json

  python3 metrics.py output/qwen2.5_1.5b_ckpt1500.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_ckpt1500_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt1500_summary.json
  python3 metrics.py output/qwen2.5_1.5b_ckpt1500.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_ckpt1500_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt1500_summary.json

  python3 metrics.py output/qwen2.5_1.5b_ckpt500.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_ckpt500_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt500_summary.json
  python3 metrics.py output/qwen2.5_1.5b_ckpt500.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_ckpt500_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt500_summary.json

  python3 metrics.py output/qwen2.5_1.5b_ckpt1000.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_ckpt1000_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt1000_summary.json
  python3 metrics.py output/qwen2.5_1.5b_ckpt1000.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_ckpt1000_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt1000_summary.json

  python3 metrics.py output/qwen2.5_1.5b_ckpt2000.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_ckpt2000_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt2000_summary.json
  python3 metrics.py output/qwen2.5_1.5b_ckpt2000.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_ckpt2000_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt2000_summary.json

  python3 metrics.py output/qwen2.5_1.5b_ckpt3000.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_ckpt3000_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt3000_summary.json
  python3 metrics.py output/qwen2.5_1.5b_ckpt3000.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_ckpt3000_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_ckpt3000_summary.json

  python3 metrics.py output/qwen3_4b_baseline.jsonl \
    --n 1 \
    --out output/qwen3_4b_baseline_judged.jsonl \
    --summary-out output/qwen3_4b_baseline_summary.json
  python3 metrics.py output/qwen3_4b_baseline.jsonl \
    --n 4 \
    --out output/qwen3_4b_baseline_judged.jsonl \
    --summary-out output/qwen3_4b_baseline_summary.json

  python3 metrics.py output/qwen2.5_1.5b_baseline.jsonl \
    --n 1 \
    --out output/qwen2.5_1.5b_baseline_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_baseline_summary.json
  python3 metrics.py output/qwen2.5_1.5b_baseline.jsonl \
    --n 4 \
    --out output/qwen2.5_1.5b_baseline_judged.jsonl \
    --summary-out output/qwen2.5_1.5b_baseline_summary.json

  python3 metrics.py file.jsonl --n 5
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable
from dataclasses import asdict


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class CheckResult:
    ok: bool
    reason: str = ""


def normalize_ops(s: str) -> str:
    """Normalize common unicode operator variants to ASCII."""
    return (
        s.replace("×", "*")
         .replace("÷", "/")
         .replace("−", "-")
         .strip()
    )


def extract_answer_equation_last(text: str) -> str | None:
    """
    Return the INNER STRING of the FINAL <answer>...</answer> block in text.
    If none exist, return None.
    """
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    eq = normalize_ops(matches[-1])
    return eq if eq else None



def last_n_assistant_contents(messages: Any, n: int) -> list[str]:
    """
    Extract the last n assistant message contents from a messages list.
    Returns them in chronological order (oldest -> newest among the last n).
    """
    if n <= 0 or not isinstance(messages, list):
        return []

    assistant_texts: list[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            assistant_texts.append(str(m.get("content", "") or ""))

    if not assistant_texts:
        return []

    return assistant_texts[-n:]


class SafeCountdownEvaluator:
    """
    Safe evaluator for arithmetic expressions using only +, -, *, / and parentheses.
    - Uses Fractions for exact arithmetic.
    - Tracks integer literals encountered (for number-usage checking).
    """

    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def __init__(self) -> None:
        self.used_numbers: list[int] = []

    def parse(self, expr: str) -> ast.AST:
        return ast.parse(expr, mode="eval")

    def eval(self, node: ast.AST) -> Fraction:
        return self._eval_node(node)

    def _eval_node(self, node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or node.value is None:
                raise ValueError("Invalid literal")
            if isinstance(node.value, int):
                self.used_numbers.append(int(node.value))
                return Fraction(int(node.value), 1)
            if isinstance(node.value, float):
                raise ValueError("Float literals not allowed; use integers only")
            raise ValueError("Unsupported literal type")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.ALLOWED_UNARYOPS):
            val = self._eval_node(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val

        if isinstance(node, ast.BinOp) and isinstance(node.op, self.ALLOWED_BINOPS):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right

        raise ValueError(f"Disallowed syntax: {type(node).__name__}")


def split_lhs_rhs(answer_text: str) -> tuple[str, str | None]:
    """
    If the answer contains '=', treat it as an equation: LHS = RHS
    We split on the LAST '=' to be conservative.
    """
    if "=" not in answer_text:
        return answer_text.strip(), None
    lhs, rhs = answer_text.rsplit("=", 1)
    return lhs.strip(), rhs.strip()


def eval_expr(expr: str) -> tuple[Fraction, list[int]]:
    ev = SafeCountdownEvaluator()
    tree = ev.parse(expr)
    val = ev.eval(tree)
    return val, ev.used_numbers


def check_completion_detailed(target: int, nums: list[int], assistant_text: str) -> dict[str, Any]:
    """
    Like check_completion, but returns a JSON-serializable dict with debug info.
    """
    out: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "answer_text": None,
        "lhs": None,
        "rhs": None,
        "lhs_val": None,
        "rhs_val": None,
        "used_nums": None,
        "assistant_content": assistant_text,
    }

    answer = extract_answer_equation_last(assistant_text)
    out["answer_text"] = answer
    if answer is None:
        out["reason"] = "missing_answer_tags"
        return out

    lhs, rhs = split_lhs_rhs(answer)
    out["lhs"] = lhs
    out["rhs"] = rhs

    try:
        lhs_val, used_nums = eval_expr(lhs)
        out["lhs_val"] = str(lhs_val)
        out["used_nums"] = used_nums
    except Exception as e:
        out["reason"] = f"lhs_parse_or_eval_error: {e}"
        return out

    provided = Counter(nums)
    used = Counter(used_nums)
    for k, cnt in used.items():
        if provided[k] < cnt:
            out["reason"] = f"invalid_number_usage: used {k} x{cnt}, available x{provided[k]}"
            return out

    target_frac = Fraction(int(target), 1)

    if rhs is not None and rhs != "":
        try:
            rhs_val, _ = eval_expr(rhs)  # ignore rhs usage
            out["rhs_val"] = str(rhs_val)
        except Exception as e:
            out["reason"] = f"rhs_parse_or_eval_error: {e}"
            return out

        if lhs_val != rhs_val:
            out["reason"] = f"equation_not_balanced: lhs {lhs_val} != rhs {rhs_val}"
            return out

        if rhs_val != target_frac:
            out["reason"] = f"wrong_value: got {rhs_val}, target {target}"
            return out

        out["ok"] = True
        out["reason"] = "ok"
        return out

    # no RHS
    if lhs_val != target_frac:
        out["reason"] = f"wrong_value: got {lhs_val}, target {target}"
        return out

    out["ok"] = True
    out["reason"] = "ok"
    return out


def check_row_pass_n_detailed(
    target: int, nums: list[int], messages: list[dict[str, Any]], n: int
) -> tuple[bool, list[dict[str, Any]]]:
    texts = last_n_assistant_contents(messages, n)
    details = [check_completion_detailed(target, nums, t) for t in texts]
    pass_ok = any(d["ok"] for d in details)
    return pass_ok, details



def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="JSONL with keys: target, nums, messages")
    ap.add_argument("--n", type=int, default=1, help="Compute pass@n using the last n assistant messages")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-show", type=int, default=50)
    ap.add_argument("--out", type=str, default=None, help="Write augmented JSONL to this path")
    ap.add_argument("--summary-out", type=str, default=None, help="Write dataset-level metrics JSON to this path")
    args = ap.parse_args()

    total = 0
    valid = 0
    passed = 0
    failures = Counter()
    shown = 0

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for obj in iter_jsonl(args.jsonl):
            total += 1
            target = obj.get("target")
            nums = obj.get("nums")
            messages = obj.get("messages")

            if not (
                isinstance(target, int)
                and isinstance(nums, list)
                and all(isinstance(x, int) for x in nums)
                and isinstance(messages, list)
            ):
                failures["bad_input_row"] += 1
                if out_f:
                    obj.setdefault("metrics", {})
                    obj["metrics"]["error"] = "bad_input_row"
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            valid += 1

            pass_ok, details = check_row_pass_n_detailed(target, nums, messages, args.n)

            if pass_ok:
                passed += 1
            else:
                if details:
                    reason_key = str(details[-1]["reason"]).split(":")[0]
                    failures[reason_key] += 1
                else:
                    failures["no_assistant_messages"] += 1

                if args.verbose and shown < args.max_show:
                    print(f"[{total}] FAIL pass@{args.n}")
                    print(f"  target={target}, nums={nums}")
                    if not details:
                        print("  (no assistant messages found)")
                    else:
                        for i, d in enumerate(details, start=1):
                            status = "OK" if d["ok"] else f"FAIL {d['reason']}"
                            print(f"  completion[{i}/{len(details)}]: {status}")
                            print(f"    text={d['assistant_content']!r}")
                    shown += 1

            if out_f:
                obj.setdefault("metrics", {})
                obj["metrics"]["n"] = args.n
                obj["metrics"][f"pass@{args.n}"] = pass_ok
                obj["metrics"]["completions"] = details
                # handy for dataset slicing without re-parsing completions
                obj["metrics"]["newest_reason_key"] = (
                    "ok" if pass_ok else (str(details[-1]["reason"]).split(":")[0] if details else "no_assistant_messages")
                )
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    finally:
        if out_f:
            out_f.close()

    acc = passed / valid if valid else 0.0
    print(f"Total rows (including bad rows): {total}")
    print(f"Valid rows: {valid}")
    print(f"Passed (pass@{args.n}): {passed}")
    print(f"Pass@{args.n}: {acc:.6f}")

    if failures:
        print("\nFailure breakdown (valid rows that failed, using newest completion reason key; plus bad_input_row):")
        for k, v in failures.most_common():
            print(f"  {k}: {v}")

    summary = {
        "input_jsonl": args.jsonl,
        "output_jsonl": args.out,
        "n": args.n,
        "total_rows": total,
        "valid_rows": valid,
        "passed_rows": passed,
        "pass@n": acc,
        "failure_breakdown": dict(failures),
    }

    if args.summary_out:
        with open(f'{args.summary_out.split(".json")[0]}_pass@{args.n}.json', "w", encoding="utf-8") as sf:
            sf.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")



if __name__ == "__main__":
    main()
