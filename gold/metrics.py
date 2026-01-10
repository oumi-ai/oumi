#!/usr/bin/env python3
"""
Countdown accuracy checker for data shaped like:
  - target: int
  - nums: list[int]
  - messages: list[{"role": "...", "content": "..."}]

A row is correct iff:
1) The LAST assistant message contains exactly one <answer>...</answer>.
2) LHS uses only provided nums, each at most once.
3) Expression evaluates exactly to target.
4) If an '=' is present inside <answer>, we also require:
     LHS == RHS == target
   and we do NOT count RHS literals toward number usage.

python3 metrics.py /data/shanghong/oumi/gold/data/train_90.jsonl --verbose --max-show 20
python3 metrics.py /data/shanghong/oumi/gold/output/qwen2.5_7b_baseline.jsonl --verbose --max-show 20
python3 metrics.py /data/shanghong/oumi/gold/output/qwen2.5_1.5b_baseline.jsonl --verbose --max-show 20
python3 metrics.py /data/shanghong/oumi/gold/output/qwen2.5_1.5b_ckpt1000.jsonl --verbose --max-show 20
python3 metrics.py /data/shanghong/oumi/gold/output/qwen3_4b_baseline.jsonl --verbose --max-show 20
python3 metrics.py /data/shanghong/oumi/gold/output/qwen2.5_1.5b_ckpt500.jsonl --verbose --max-show 20

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


def extract_answer_equation(text: str) -> str | None:
    """
    Return the INNER STRING of the FINAL <answer>...</answer> block.
    If none exist, return None.
    """
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None

    # take the final <answer>...</answer>
    eq = normalize_ops(matches[-1])
    return eq if eq else None


def last_assistant_content(messages: Any) -> str:
    """Extract the last assistant message content from a messages list."""
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", "") or "")
    # fallback: concatenate if roles are missing
    parts = []
    for m in messages:
        if isinstance(m, dict) and "content" in m:
            parts.append(str(m["content"]))
    return "\n".join(parts)


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
        # parse as a single expression
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
    If the answer contains '=', treat it as an equation:
      LHS = RHS
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


def check_row(target: int, nums: list[int], messages: list[dict[str, Any]]) -> CheckResult:
    text = last_assistant_content(messages)
    answer = extract_answer_equation(text)
    if answer is None:
        return CheckResult(False, "missing_answer_tags")

    lhs, rhs = split_lhs_rhs(answer)

    # Evaluate LHS (and track used numbers)
    try:
        lhs_val, used_nums = eval_expr(lhs)
    except Exception as e:
        return CheckResult(False, f"lhs_parse_or_eval_error: {e}")

    # Number usage check (LHS only)
    provided = Counter(nums)
    used = Counter(used_nums)
    for k, cnt in used.items():
        if provided[k] < cnt:
            return CheckResult(False, f"invalid_number_usage: used {k} x{cnt}, available x{provided[k]}")

    target_frac = Fraction(int(target), 1)

    # If RHS exists: evaluate it WITHOUT counting numbers toward usage
    if rhs is not None and rhs != "":
        try:
            rhs_val, _ = eval_expr(rhs)  # ignore rhs used numbers on purpose
        except Exception as e:
            return CheckResult(False, f"rhs_parse_or_eval_error: {e}")

        if lhs_val != rhs_val:
            return CheckResult(False, f"equation_not_balanced: lhs {lhs_val} != rhs {rhs_val}")

        if rhs_val != target_frac:
            return CheckResult(False, f"wrong_value: got {rhs_val}, target {target}")

        return CheckResult(True, "ok")

    # No RHS: just require LHS == target
    if lhs_val != target_frac:
        return CheckResult(False, f"wrong_value: got {lhs_val}, target {target}")

    return CheckResult(True, "ok")


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
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-show", type=int, default=50)
    args = ap.parse_args()

    total = 0
    correct = 0
    failures = Counter()
    shown = 0

    for obj in iter_jsonl(args.jsonl):
        total += 1
        target = obj.get("target")
        nums = obj.get("nums")
        messages = obj.get("messages")
        # print(obj)
        # print(messages)
        # print(target, nums)
        # print("--------------------------------")

        if not isinstance(target, int) or not isinstance(nums, list) or not all(isinstance(x, int) for x in nums) or not isinstance(messages, list):
            failures["bad_input_row"] += 1
            continue

        res = check_row(target, nums, messages)
        if res.ok:
            correct += 1
        else:
            failures[res.reason.split(":")[0]] += 1
            if args.verbose and shown < args.max_show:
                text = last_assistant_content(messages)
                ans = extract_answer_equation(text)
                print(f"[{total}] FAIL {res.reason}")
                print(f"  target={target}, nums={nums}")
                print(f"  extracted_answer={ans!r}, text={text}")
                shown += 1

    acc = correct / total if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.6f}")

    if failures:
        print("\nFailure breakdown:")
        for k, v in failures.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
