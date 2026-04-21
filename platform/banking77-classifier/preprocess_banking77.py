"""Pre-process Banking-77 dataset into Oumi-compatible JSONL format."""

import json
import random
from pathlib import Path

import datasets
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
LABELS_FILE = SCRIPT_DIR / "banking77_labels.txt"
IN_CONTEXT_FILE = SCRIPT_DIR / "banking77_in_context_examples.txt"


def load_labels() -> list[str]:
    labels = []
    with open(LABELS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def load_in_context_examples() -> list[str]:
    examples = []
    with open(IN_CONTEXT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(line.replace("\\n", "\n"))
    return examples


def build_classifier_instruction(
    labels: list[str], in_context_examples: list[str] | None = None
) -> str:
    id_list = "\n".join(f"{i}: {label}" for i, label in enumerate(labels))
    sep = "\n\n"
    examples_block = (
        "EXAMPLES TO HELP DISTINGUISH SIMILAR INTENTS:\n\n"
        + sep.join(in_context_examples)
        if in_context_examples is not None and len(in_context_examples) > 0
        else ""
    )
    return f"""You are a banking intent classifier. Classify the user's query into one of 77 banking intents (output is a single integer ID).


IDS:
{id_list}


CRITICAL INSTRUCTIONS:
1. Choose exactly one integer ID (0-76).
2. Reply with ONLY that number. No words, no reasoning, no punctuation.
Examples: 0, 1, 42


{examples_block}
Remember: Respond with ONLY the numeric ID, nothing else."""


def transform_row(
    row: dict, labels: list[str], in_context_examples: list[str] | None = None
) -> dict:
    label_id = row["label"]
    label_name = labels[label_id]
    query = row["text"]
    classifier_instruction = build_classifier_instruction(labels, in_context_examples)
    return {
        "messages": [
            {"role": "system", "content": classifier_instruction},
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(label_id)},
        ],
        "metadata": {
            "label": str(label_id),
            "label_name": label_name,
        },
    }


def write_jsonl(samples: list[dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Wrote {len(samples)} samples to {output_path.name}")


def confirm(step: str) -> None:
    response = (
        input(f"\n[{step}] Press Enter to continue or 'q' to quit: ").strip().lower()
    )
    if response == "q":
        print("Exiting.")
        raise SystemExit(0)


def main() -> None:
    labels = load_labels()
    assert len(labels) == 77, f"Expected 77 labels, got {len(labels)}"
    in_context_examples = load_in_context_examples()
    assert len(in_context_examples) == 8, (
        f"Expected 8 in-context examples, got {len(in_context_examples)}"
    )

    # ── Step 1: Download and read dataset ────────────────────────────────────
    print("Step 1: Downloading Banking-77 dataset from HuggingFace Hub...")
    dataset = datasets.load_dataset("PolyAI/banking77")
    df: pd.DataFrame = dataset[datasets.Split.TRAIN].to_pandas()
    df_test: pd.DataFrame = dataset[datasets.Split.TEST].to_pandas()
    print(f"  Train: {len(df)} rows  |  Test: {len(df_test)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample train row: {df.iloc[0].to_dict()}")

    confirm("Step 1 complete")

    # ── Step 2: Verify transform function ────────────────────────────────────
    print("\nStep 2: Verifying transform function...")
    sample_no_shot = transform_row(df.iloc[0].to_dict(), labels, in_context_examples=[])
    print("  No-shot sample:")
    print(
        f"    system (first 120 chars): {sample_no_shot['messages'][0]['content'][:120]!r}"
    )
    print(f"    user: {sample_no_shot['messages'][1]}")
    print(f"    assistant: {sample_no_shot['messages'][2]}")
    print(f"    metadata: {sample_no_shot['metadata']}")

    sample_2shot = transform_row(
        df.iloc[0].to_dict(), labels, in_context_examples=in_context_examples[:2]
    )
    print("\n  2-shot sample (system prompt, last 200 chars):")
    print(f"    {sample_2shot['messages'][0]['content'][-200:]!r}")

    confirm("Step 2 complete")

    # ── Step 3: Create JSONL datasets ─────────────────────────────────────────
    print("\nStep 3: Creating JSONL datasets...")

    # print("  banking77-test.jsonl")
    # write_jsonl(
    #     [transform_row(row, labels, in_context_examples=[]) for row in df_test.to_dict("records")],
    #     SCRIPT_DIR / "banking77-test.jsonl",
    # )

    # print("  banking77-test-1-shot.jsonl")
    # write_jsonl(
    #     [transform_row(row, labels, in_context_examples=random.sample(in_context_examples, 1)) for row in df_test.to_dict("records")],
    #     SCRIPT_DIR / "banking77-test-1-shot.jsonl",
    # )

    # print("  banking77-test-3-shot.jsonl")
    # write_jsonl(
    #     [transform_row(row, labels, in_context_examples=random.sample(in_context_examples, 3)) for row in df_test.to_dict("records")],
    #     SCRIPT_DIR / "banking77-test-3-shot.jsonl",
    # )

    # print("  banking77-test-5-shot.jsonl")
    # write_jsonl(
    #     [transform_row(row, labels, in_context_examples=random.sample(in_context_examples, 5)) for row in df_test.to_dict("records")],
    #     SCRIPT_DIR / "banking77-test-5-shot.jsonl",
    # )

    # print("  banking77-train.jsonl")
    # write_jsonl(
    #     [transform_row(row, labels, in_context_examples=[]) for row in df.to_dict("records")],
    #     SCRIPT_DIR / "banking77-train.jsonl",
    # )

    print("  banking77-test-tiny.jsonl")
    write_jsonl(
        [
            transform_row(row, labels, in_context_examples=[])
            for row in random.sample(df_test.to_dict("records"), 100)
        ],
        SCRIPT_DIR / "banking77-test-tiny.jsonl",
    )

    confirm("Step 3 complete — all datasets written")
    print("Done.")


if __name__ == "__main__":
    main()
