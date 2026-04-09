"""Filter Hermes Reasoning Tool Use dataset for training or evaluation.

Applies quality filters and optionally prepares data for evaluation.

Filters (applied in both modes):
1. Remove 'relevance' examples (no tool calls expected).
2. Remove mislabeled examples (assistant turns with <tool_response> but no <tool_call>).

Additional processing in eval mode (--mode eval):
3. Truncate messages to the context before the final assistant tool call.
4. Store gold tool call in metadata["gold_tool_call"].
5. Store gold assistant content in metadata["gold_assistant_content"].
6. Skip records with no parseable tool call.

Usage:
    # Training data (full conversations, quality filtered):
    python filter_dataset.py --mode train

    # Eval data (truncated, with gold tool call in metadata):
    python filter_dataset.py --mode eval

    # Both:
    python filter_dataset.py --mode both
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path("/data/shanghong/oumi/gold/data")
OUTPUT_DIR = Path("/data/shanghong/oumi/tool_call_project/data")

SPLITS = {
    "train": "hermes_reasoning_tool_use_train_split.jsonl",
    "val": "hermes_reasoning_tool_use_val_split.jsonl",
    "test": "hermes_reasoning_tool_use_test_split.jsonl",
}


# ---------------------------------------------------------------------------
# Shared filters
# ---------------------------------------------------------------------------


def has_mislabeled_assistant(messages: list[dict]) -> bool:
    """Check if any assistant turn contains <tool_response> without <tool_call>."""
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            if "<tool_response>" in content and "<tool_call>" not in content:
                return True
    return False


# ---------------------------------------------------------------------------
# Eval-only processing
# ---------------------------------------------------------------------------


def parse_last_tool_call(content: str) -> dict | None:
    """Extract the last <tool_call> JSON block from a message.

    Finds the last </tool_call> and works backwards to the nearest <tool_call>
    before it, to avoid matching nested tags inside <think> blocks.
    """
    close_tag = "</tool_call>"
    open_tag = "<tool_call>"

    close_pos = content.rfind(close_tag)
    if close_pos == -1:
        return None

    open_pos = content.rfind(open_tag, 0, close_pos)
    if open_pos == -1:
        return None

    raw = content[open_pos + len(open_tag) : close_pos].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
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


def truncate_for_eval(record: dict) -> dict | None:
    """Truncate to input messages and extract gold tool call.

    Returns the processed record, or None if no parseable tool call found.
    """
    messages = record["messages"]

    # Find the last assistant message that contains a tool call
    last_tool_call_idx = None
    last_tool_call = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            tc = parse_last_tool_call(messages[i]["content"])
            if tc is not None:
                last_tool_call_idx = i
                last_tool_call = tc
                break

    if last_tool_call is None:
        return None

    input_messages = messages[:last_tool_call_idx]
    if not input_messages:
        return None

    record["messages"] = input_messages
    record["metadata"]["gold_tool_call"] = last_tool_call
    record["metadata"]["gold_assistant_content"] = messages[last_tool_call_idx]["content"]
    return record


# ---------------------------------------------------------------------------
# Main filter logic
# ---------------------------------------------------------------------------


def filter_split(input_path: Path, output_path: Path, eval_mode: bool) -> dict:
    """Filter a single split. Returns counts dict."""
    counts = {"kept": 0, "relevance": 0, "mislabeled": 0, "no_tool_call": 0}
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)

            # Filter 1: remove relevance examples
            if record["metadata"]["scenario_category"] == "relevance":
                counts["relevance"] += 1
                continue

            # Filter 2: remove mislabeled assistant turns
            if has_mislabeled_assistant(record["messages"]):
                counts["mislabeled"] += 1
                continue

            # Eval mode: truncate and extract gold tool call
            if eval_mode:
                record = truncate_for_eval(record)
                if record is None:
                    counts["no_tool_call"] += 1
                    continue
                fout.write(json.dumps(record) + "\n")
            else:
                fout.write(line)

            counts["kept"] += 1

    return counts


def run(mode: str):
    """Run filtering for the given mode."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_mode = mode == "eval"
    suffix = "_tool_calls_only.jsonl" if eval_mode else "_clean.jsonl"
    label = "eval" if eval_mode else "train"

    print(f"Mode: {label}")
    for split_name, filename in SPLITS.items():
        input_path = DATA_DIR / filename
        output_path = OUTPUT_DIR / filename.replace(".jsonl", suffix)

        if not input_path.exists():
            print(f"  Skipping {split_name}: {input_path} not found")
            continue

        counts = filter_split(input_path, output_path, eval_mode=eval_mode)
        total_removed = counts["relevance"] + counts["mislabeled"] + counts["no_tool_call"]
        parts = [f"relevance={counts['relevance']}", f"mislabeled={counts['mislabeled']}"]
        if eval_mode:
            parts.append(f"no_tool_call={counts['no_tool_call']}")
        print(
            f"  {split_name}: kept {counts['kept']}, "
            f"removed {total_removed} ({', '.join(parts)}) "
            f"-> {output_path.name}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Filter Hermes tool use dataset for training or evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="'train' = quality filter only, 'eval' = filter + truncate for eval, "
        "'both' = run both (default: both)",
    )
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        run("train")
    if args.mode in ("eval", "both"):
        if args.mode == "both":
            print()
        run("eval")


if __name__ == "__main__":
    main()
