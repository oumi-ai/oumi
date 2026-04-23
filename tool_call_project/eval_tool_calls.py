"""Evaluate tool call correctness on the Hermes Reasoning Tool Use dataset.

For each example in the filtered dataset, compares the model's predicted tool
call against the gold tool call stored in metadata. Measures:
  - Tool name accuracy: did the model pick the right function?
  - Argument accuracy: did the model pass the right arguments?
  - Exact match: both name and arguments correct?

The filtered dataset (produced by filter_dataset.py) has:
  - messages: input context (truncated before the gold assistant tool call turn)
  - metadata.gold_tool_call: {"name": str, "arguments": dict}

Usage:
    # Sanity check with gold answers (should be 100%):
    python eval_tool_calls.py --dataset_path /path/to/test_tool_calls_only.jsonl

    # Evaluate model predictions:
    python eval_tool_calls.py --dataset_path /path/to/test.jsonl \
        --predictions_path /path/to/preds.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_last_tool_call(content: str) -> dict | None:
    """Extract the last <tool_call> JSON block from a string.

    Finds the last </tool_call> and works backwards to the nearest <tool_call>
    before it, to avoid matching nested tags inside <think> blocks.

    Returns {"name": str, "arguments": dict} or None if no tool call found.
    """
    close_tag = "</tool_call>"
    open_tag = "<tool_call>"

    # Find the last </tool_call>
    close_pos = content.rfind(close_tag)
    if close_pos == -1:
        return None

    # Find the nearest <tool_call> before it
    open_pos = content.rfind(open_tag, 0, close_pos)
    if open_pos == -1:
        return None

    raw = content[open_pos + len(open_tag) : close_pos].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fallback: find outermost {} in case of extra text
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def normalize_value(v):
    """Normalize a value for comparison: convert numeric strings to numbers."""
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v.strip()
    if isinstance(v, dict):
        return normalize_args(v)
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    return v


def normalize_args(args) -> dict:
    """Normalize arguments for comparison: sort keys, normalize types."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args.strip()}
    if not isinstance(args, dict):
        return {"_raw": str(args)}
    return {k: normalize_value(v) for k, v in sorted(args.items())}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compare_tool_calls(pred: dict, gold: dict) -> dict:
    """Compare a predicted tool call against ground truth.

    Returns dict with keys: name_correct, args_correct, exact_match.
    """
    pred_name = pred.get("name")
    gold_name = gold.get("name")
    name_correct = pred_name is not None and pred_name == gold_name
    args_correct = normalize_args(pred.get("arguments", {})) == normalize_args(
        gold.get("arguments", {})
    )
    return {
        "name_correct": name_correct,
        "args_correct": args_correct,
        "exact_match": name_correct and args_correct,
    }


def evaluate(
    predictions: list[dict | None], golds: list[dict], categories: list[str]
) -> dict:
    """Compute metrics over all examples.

    Args:
        predictions: list of parsed tool call dicts (or None if parsing failed)
        golds: list of gold tool call dicts
        categories: list of scenario_category strings

    Returns:
        dict with overall and per-category metrics.
    """
    overall = defaultdict(int)
    per_cat = defaultdict(lambda: defaultdict(int))

    for pred, gold, cat in zip(predictions, golds, categories):
        overall["total"] += 1
        per_cat[cat]["total"] += 1

        if pred is None:
            overall["parse_failures"] += 1
            per_cat[cat]["parse_failures"] += 1
            continue

        result = compare_tool_calls(pred, gold)
        for key in ["name_correct", "args_correct", "exact_match"]:
            if result[key]:
                overall[key] += 1
                per_cat[cat][key] += 1

    def compute_rates(counts: dict) -> dict:
        t = counts["total"]
        return {
            "total": t,
            "parse_failures": counts.get("parse_failures", 0),
            "name_accuracy": counts.get("name_correct", 0) / t if t else 0,
            "args_accuracy": counts.get("args_correct", 0) / t if t else 0,
            "exact_match": counts.get("exact_match", 0) / t if t else 0,
        }

    return {
        "overall": compute_rates(dict(overall)),
        "per_category": {
            cat: compute_rates(dict(counts)) for cat, counts in sorted(per_cat.items())
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool call correctness")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to filtered JSONL dataset (from filter_dataset.py)",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Path to predictions JSONL (one line per example, with 'content' field). "
        "If not provided, uses gold answers for sanity check.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only evaluate the first N examples.",
    )
    args = parser.parse_args()

    # Load dataset — gold tool calls are in metadata
    golds = []
    categories = []
    with open(args.dataset_path) as f:
        for i, line in enumerate(f):
            if args.num_samples and i >= args.num_samples:
                break
            record = json.loads(line)
            golds.append(record["metadata"]["gold_tool_call"])
            categories.append(record["metadata"]["scenario_category"])
    print(f"Loaded {len(golds)} examples")

    # Get predictions
    if args.predictions_path:
        predictions = []
        with open(args.predictions_path) as f:
            for line in f:
                pred_record = json.loads(line)
                content = pred_record.get("content", "")
                predictions.append(parse_last_tool_call(content))
        assert len(predictions) == len(golds), (
            f"Mismatch: {len(predictions)} predictions vs {len(golds)} examples"
        )
    else:
        print("No predictions file — running sanity check with gold answers\n")
        predictions = golds  # gold is already parsed, no need to re-parse

    # Evaluate
    results = evaluate(predictions, golds, categories)

    # Print results
    print("=" * 50)
    print("OVERALL")
    print("=" * 50)
    r = results["overall"]
    print(f"  Total:          {r['total']}")
    print(f"  Parse failures: {r['parse_failures']}")
    print(f"  Name accuracy:  {r['name_accuracy']:.4f}")
    print(f"  Args accuracy:  {r['args_accuracy']:.4f}")
    print(f"  Exact match:    {r['exact_match']:.4f}")

    print()
    print("PER CATEGORY")
    print("=" * 50)
    for cat, r in results["per_category"].items():
        print(f"  {cat}:")
        print(f"    Total:          {r['total']}")
        print(f"    Parse failures: {r['parse_failures']}")
        print(f"    Name accuracy:  {r['name_accuracy']:.4f}")
        print(f"    Args accuracy:  {r['args_accuracy']:.4f}")
        print(f"    Exact match:    {r['exact_match']:.4f}")

    # Save as JSON
    output_path = Path(args.dataset_path).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
