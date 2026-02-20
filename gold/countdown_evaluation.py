#!/usr/bin/env python3
"""
Countdown Game Evaluation Script

Evaluates model performance on the Countdown math game where models must:
- Reach a target value using given numbers and arithmetic operations (+, -, *, /)
- Use each number exactly once
- Provide answers in <answer>...</answer> tags with correct format

Input JSONL format:
  {
    "target": int,           # Target value to reach
    "nums": [int, ...],      # List of numbers to use
    "messages": [            # Conversation history
      {"role": "...", "content": "..."}
    ]
  }

Correctness Criteria (ALL must be satisfied):
  1. Answer enclosed in <answer>...</answer> tags
  2. Each number from 'nums' used exactly once in the expression
  3. Expression evaluates exactly to the target value
  4. If '=' present: both LHS and RHS must equal target (RHS nums don't count toward usage)

Metrics:
  pass@n: Whether ANY of the last n assistant messages is correct

Usage:
  python3 countdown_evaluation.py input.jsonl --n 4 \
    --out judged_output.jsonl \
    --summary-out summary.json \
    --verbose
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


# Regex to extract answer from <answer>...</answer> tags (case-insensitive, multiline)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class CheckResult:
    """Result of validation check."""
    ok: bool
    reason: str = ""


def normalize_ops(s: str) -> str:
    """
    Normalize common unicode operator variants to ASCII equivalents.

    Converts: × → *, ÷ → /, − → -
    """
    return (
        s.replace("×", "*")
         .replace("÷", "/")
         .replace("−", "-")
         .strip()
    )


def extract_answer_equation_last(text: str) -> str | None:
    """
    Extract the LAST <answer>...</answer> block from text.

    Returns:
        The inner string (normalized) of the final answer block, or None if not found.
    """
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    eq = normalize_ops(matches[-1])
    return eq if eq else None



def last_n_assistant_contents(messages: Any, n: int) -> list[str]:
    """
    Extract the last n assistant message contents from a messages list.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        n: Number of most recent assistant messages to extract

    Returns:
        List of assistant message contents in chronological order (oldest to newest)
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
    Safe evaluator for arithmetic expressions with strict validation.

    Features:
    - Only allows arithmetic operations: +, -, *, /
    - Uses Fraction arithmetic for exact computation (no floating point errors)
    - Tracks all integer literals used in the expression
    - Prevents execution of arbitrary Python code

    Attributes:
        used_numbers: List of all integer literals encountered during evaluation
    """

    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def __init__(self) -> None:
        self.used_numbers: list[int] = []

    def parse(self, expr: str) -> ast.AST:
        """Parse expression string into AST."""
        return ast.parse(expr, mode="eval")

    def eval(self, node: ast.AST) -> Fraction:
        """Evaluate AST node and return result as Fraction."""
        return self._eval_node(node)

    def _eval_node(self, node: ast.AST) -> Fraction:
        """
        Recursively evaluate an AST node.

        Handles:
        - Constants (integers only)
        - Unary operations (+, -)
        - Binary operations (+, -, *, /)

        Raises:
            ValueError: If node type is not allowed
            ZeroDivisionError: If division by zero
        """
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        # Handle integer constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or node.value is None:
                raise ValueError("Invalid literal")
            if isinstance(node.value, int):
                self.used_numbers.append(int(node.value))
                return Fraction(int(node.value), 1)
            if isinstance(node.value, float):
                raise ValueError("Float literals not allowed; use integers only")
            raise ValueError("Unsupported literal type")

        # Handle unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.ALLOWED_UNARYOPS):
            val = self._eval_node(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val

        # Handle binary operations (x + y, x - y, x * y, x / y)
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
    Split answer text into left-hand side and right-hand side of equation.

    If '=' is present, splits on the LAST '=' to handle potential nested expressions.
    If '=' is not present, treats entire text as LHS.

    Args:
        answer_text: The text extracted from <answer> tags

    Returns:
        Tuple of (lhs, rhs) where rhs is None if no '=' present
    """
    if "=" not in answer_text:
        return answer_text.strip(), None
    lhs, rhs = answer_text.rsplit("=", 1)
    return lhs.strip(), rhs.strip()


def eval_expr(expr: str) -> tuple[Fraction, list[int]]:
    """
    Safely evaluate arithmetic expression and track number usage.

    Args:
        expr: Arithmetic expression string (e.g., "3 + 5 * 2")

    Returns:
        Tuple of (result as Fraction, list of integer literals used)

    Raises:
        ValueError: If expression contains disallowed operations
        ZeroDivisionError: If division by zero occurs
    """
    ev = SafeCountdownEvaluator()
    tree = ev.parse(expr)
    val = ev.eval(tree)
    return val, ev.used_numbers


def check_completion_detailed(target: int, nums: list[int], assistant_text: str) -> dict[str, Any]:
    """
    Validates a single completion against Countdown rules.

    Returns a detailed dict with validation status and debug information:
      - ok: bool - whether the completion is correct
      - reason: str - 'ok' if correct, otherwise failure reason
      - answer_text: str | None - extracted answer from tags
      - lhs/rhs/lhs_val/rhs_val: parsed equation components
      - used_nums: list of numbers used in LHS expression
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

    # Rule 1: Answer must be enclosed in <answer>...</answer> tags
    answer = extract_answer_equation_last(assistant_text)
    out["answer_text"] = answer
    if answer is None:
        out["reason"] = "missing_answer_tags"
        return out

    # Split into LHS and RHS if '=' is present
    lhs, rhs = split_lhs_rhs(answer)
    out["lhs"] = lhs
    out["rhs"] = rhs

    # Parse and evaluate LHS expression
    try:
        lhs_val, used_nums = eval_expr(lhs)
        out["lhs_val"] = str(lhs_val)
        out["used_nums"] = used_nums
    except Exception as e:
        out["reason"] = f"lhs_parse_or_eval_error: {e}"
        return out

    # Rule 2: Each number must be used exactly once
    if Counter(used_nums) != Counter(nums):
        out["reason"] = f"invalid_number_usage: used {sorted(used_nums)}, expected {sorted(nums)}"
        return out

    target_frac = Fraction(int(target), 1)

    # If RHS is present (equation format: LHS = RHS)
    if rhs is not None and rhs != "":
        try:
            rhs_val, _ = eval_expr(rhs)  # Note: RHS numbers don't count toward usage
            out["rhs_val"] = str(rhs_val)
        except Exception as e:
            out["reason"] = f"rhs_parse_or_eval_error: {e}"
            return out

        # Verify equation is balanced
        if lhs_val != rhs_val:
            out["reason"] = f"equation_not_balanced: lhs {lhs_val} != rhs {rhs_val}"
            return out

        # Rule 3: Expression must evaluate to target
        if rhs_val != target_frac:
            out["reason"] = f"wrong_value: got {rhs_val}, target {target}"
            return out

        out["ok"] = True
        out["reason"] = "ok"
        return out

    # No RHS present - verify LHS equals target directly
    if lhs_val != target_frac:
        out["reason"] = f"wrong_value: got {lhs_val}, target {target}"
        return out

    out["ok"] = True
    out["reason"] = "ok"
    return out


def check_row_pass_n_detailed(
    target: int, nums: list[int], messages: list[dict[str, Any]], n: int
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Compute pass@n metric for a single example.

    Args:
        target: Target value to reach
        nums: List of numbers to use
        messages: Conversation history
        n: Number of most recent assistant messages to consider

    Returns:
        Tuple of (pass_ok, details) where:
        - pass_ok: True if ANY of the last n completions is correct
        - details: List of validation results for each completion
    """
    texts = last_n_assistant_contents(messages, n)
    details = [check_completion_detailed(target, nums, t) for t in texts]
    pass_ok = any(d["ok"] for d in details)
    return pass_ok, details


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    """
    Iterate over lines in a JSONL file, parsing each as JSON.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON objects (one per non-empty line)

    Raises:
        ValueError: If any line contains invalid JSON
    """
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
    ap = argparse.ArgumentParser(
        description="Evaluate Countdown game predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("jsonl", help="Input JSONL file with keys: target, nums, messages")
    ap.add_argument(
        "--n",
        type=int,
        default=1,
        help="Compute pass@n using the last n assistant messages (default: 1)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed failure information",
    )
    ap.add_argument(
        "--max-show",
        type=int,
        default=50,
        help="Maximum number of failures to show in verbose mode (default: 50)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write augmented JSONL with per-example metrics to this path",
    )
    ap.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Write dataset-level summary metrics JSON to this path",
    )
    args = ap.parse_args()

    # Counters for tracking results
    total = 0
    valid = 0
    passed = 0
    failures = Counter()
    shown = 0

    print(f"Evaluating: {args.jsonl}")
    print(f"Computing pass@{args.n} metric")
    print("-" * 60)

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for obj in iter_jsonl(args.jsonl):
            total += 1
            metadata = obj.get("metadata", {})
            target = metadata.get("target")
            nums = metadata.get("nums")
            messages = obj.get("messages")

            # Validate input format
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

            # Evaluate pass@n for this example
            pass_ok, details = check_row_pass_n_detailed(target, nums, messages, args.n)

            if pass_ok:
                passed += 1
            else:
                # Track failure reasons
                if details:
                    reason_key = str(details[-1]["reason"]).split(":")[0]
                    failures[reason_key] += 1
                else:
                    failures["no_assistant_messages"] += 1

                # Show verbose failure details
                if args.verbose and shown < args.max_show:
                    print(f"\n[Example {total}] FAIL pass@{args.n}")
                    print(f"  Target: {target}")
                    print(f"  Numbers: {nums}")
                    if not details:
                        print("  (no assistant messages found)")
                    else:
                        for i, d in enumerate(details, start=1):
                            status = "✓ CORRECT" if d["ok"] else f"✗ FAIL: {d['reason']}"
                            print(f"  Completion {i}/{len(details)}: {status}")
                            if not d["ok"]:
                                print(f"    Answer: {d['answer_text']}")
                                if d['used_nums']:
                                    print(f"    Used numbers: {d['used_nums']}")
                    shown += 1

            # Write per-example metrics
            if out_f:
                obj.setdefault("metrics", {})
                obj["metrics"]["n"] = args.n
                obj["metrics"][f"pass@{args.n}"] = pass_ok
                obj["metrics"]["completions"] = details
                obj["metrics"]["newest_reason_key"] = (
                    "ok" if pass_ok else (str(details[-1]["reason"]).split(":")[0] if details else "no_assistant_messages")
                )
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    finally:
        if out_f:
            out_f.close()
            print(f"\nWrote per-example results to: {args.out}")

    # Calculate and display summary metrics
    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)

    acc = passed / valid if valid else 0.0
    print(f"Total examples: {total}")
    print(f"Valid examples: {valid}")
    if total != valid:
        print(f"Invalid examples: {total - valid}")
    print(f"\nPassed examples: {passed}")
    print(f"Failed examples: {valid - passed}")
    print(f"\nPass@{args.n}: {acc:.4f} ({acc*100:.2f}%)")

    if failures:
        print(f"\n{'-' * 60}")
        print("FAILURE BREAKDOWN")
        print(f"{'-' * 60}")
        for k, v in failures.most_common():
            pct = (v / valid * 100) if valid else 0
            print(f"  {k:30s}: {v:5d} ({pct:5.2f}%)")

    # Save summary metrics
    summary = {
        "input_jsonl": args.jsonl,
        "output_jsonl": args.out,
        "n": args.n,
        "total_rows": total,
        "valid_rows": valid,
        "passed_rows": passed,
        "failed_rows": valid - passed,
        f"pass@{args.n}": acc,
        f"pass@{args.n}_percent": round(acc * 100, 2),
        "failure_breakdown": dict(failures),
    }

    if args.summary_out:
        summary_path = f'{args.summary_out.split(".json")[0]}_pass@{args.n}.json'
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"\nWrote summary to: {summary_path}")



if __name__ == "__main__":
    main()
