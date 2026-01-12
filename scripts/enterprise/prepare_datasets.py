# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare enterprise task datasets in Oumi conversation format.

This script downloads datasets from HuggingFace and converts them to
Oumi JSONL format for SFT training.

Supported tasks:
- banking77: 77-class customer query classification
- pubmedqa: 3-class medical QA (yes/no/maybe)
- tatqa: Tabular question answering
- nl2sql: Natural language to SQL conversion

Usage:
    # Prepare all datasets
    python scripts/enterprise/prepare_datasets.py --all

    # Prepare specific dataset
    python scripts/enterprise/prepare_datasets.py --task banking77

    # Specify output directory
    python scripts/enterprise/prepare_datasets.py --all --output-dir data/enterprise
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from oumi.utils.logging import logger, update_logger_level


def save_jsonl(data: list[dict], output_path: Path) -> None:
    """Save data as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(data)} conversations to {output_path}")


def format_conversation(user_content: str, assistant_content: str) -> dict:
    """Format a single conversation in Oumi format."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def prepare_banking77(output_dir: Path) -> None:
    """Prepare Banking77 dataset for classification.

    Banking77 is a 77-class classification dataset of customer service queries.
    Source: https://huggingface.co/datasets/legacy-datasets/banking77
    """
    logger.info("Preparing Banking77 dataset...")

    # Use legacy-datasets version which is in parquet format
    dataset = load_dataset("legacy-datasets/banking77")

    # Get label names
    label_names = dataset["train"].features["label"].names  # type: ignore[attr-defined]

    # Build instruction with all intent IDs
    intent_list = "\n".join(f"{i}: {name}" for i, name in enumerate(label_names))
    instruction = f"""You are a banking intent classifier. Classify the user's query into one of 77 banking intents (output is a single integer ID).

IDs:

{intent_list}

CRITICAL INSTRUCTIONS:
1. Choose exactly one integer ID (0-76).
2. Reply with ONLY that number. No words, no reasoning, no punctuation.
Examples: 0, 1, 42

Remember: Respond with ONLY the numeric ID, nothing else."""

    def convert_example(example: dict) -> dict:
        text = example["text"]
        label_id = example["label"]
        label_name = label_names[label_id]

        user_content = f"{instruction}\n\n<query>\n{text}\n</query>"

        # Create conversation with metadata
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(label_id)},
            ],
            "metadata": {
                "label": label_id,
                "label_name": label_name,
            },
        }

    # Convert train and test splits
    train_data = [convert_example(ex) for ex in tqdm(dataset["train"], desc="Train")]  # type: ignore[attr-defined]
    test_data = [convert_example(ex) for ex in tqdm(dataset["test"], desc="Test")]  # type: ignore[attr-defined]

    # Carve out validation set from train
    val_size = 100
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]

    # Save
    save_jsonl(train_data, output_dir / "banking77" / "train.jsonl")
    save_jsonl(val_data, output_dir / "banking77" / "val.jsonl")
    save_jsonl(test_data, output_dir / "banking77" / "test.jsonl")

    # Save label mapping for evaluation
    label_mapping = {i: name for i, name in enumerate(label_names)}
    with open(output_dir / "banking77" / "labels.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    logger.info(
        f"Banking77: {len(train_data)} train, {len(val_data)} val, "
        f"{len(test_data)} test, {len(label_names)} classes"
    )


def prepare_pubmedqa(output_dir: Path) -> None:
    """Prepare PubMedQA dataset for classification.

    PubMedQA is a 3-class (yes/no/maybe) QA dataset based on PubMed abstracts.
    Source: https://huggingface.co/datasets/bigbio/pubmed_qa
    """
    logger.info("Preparing PubMedQA dataset...")

    # Load the labeled subset (has train/test splits with answers)
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    def convert_example(example: dict) -> dict:
        context = " ".join(example["context"]["contexts"])
        question = example["question"]
        # Answer is yes/no/maybe
        answer = example["final_decision"]

        user_content = (
            "Based on the following medical context, answer the question "
            "with 'yes', 'no', or 'maybe'.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return format_conversation(user_content, answer)

    # PubMedQA labeled only has train split, we'll create our own splits
    full_data = [
        convert_example(ex)
        for ex in tqdm(dataset["train"], desc="Converting")  # type: ignore[attr-defined]
    ]

    # Split: train / val (100) / test (100)
    val_size = 100
    test_size = 100
    test_data = full_data[-test_size:]
    val_data = full_data[-(test_size + val_size) : -test_size]
    train_data = full_data[: -(test_size + val_size)]

    # Save
    save_jsonl(train_data, output_dir / "pubmedqa" / "train.jsonl")
    save_jsonl(val_data, output_dir / "pubmedqa" / "val.jsonl")
    save_jsonl(test_data, output_dir / "pubmedqa" / "test.jsonl")

    logger.info(
        f"PubMedQA: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )


def prepare_tatqa(output_dir: Path) -> None:
    """Prepare TAT-QA dataset for tabular question answering.

    TAT-QA is a QA dataset requiring reasoning over tabular and textual data.
    Source: https://huggingface.co/datasets/next-tat/TAT-QA

    Note: The HuggingFace dataset has format issues, so we load raw JSON directly.
    """
    logger.info("Preparing TAT-QA dataset...")

    import urllib.request

    # Download raw JSON files directly from HuggingFace
    base_url = "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/"
    train_url = base_url + "tatqa_dataset_train.json"
    dev_url = base_url + "tatqa_dataset_dev.json"

    logger.info("Downloading TAT-QA train data...")
    with urllib.request.urlopen(train_url) as response:
        train_raw = json.loads(response.read().decode("utf-8"))

    logger.info("Downloading TAT-QA dev data...")
    with urllib.request.urlopen(dev_url) as response:
        dev_raw = json.loads(response.read().decode("utf-8"))

    def format_table(table: dict) -> str:
        """Format table as text."""
        if not table:
            return ""
        # Table has structure: {"table": [[row1], [row2], ...]}
        rows = table if isinstance(table, list) else table.get("table", [])
        if not rows:
            return ""
        lines = []
        for row in rows:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            elif isinstance(row, dict):
                lines.append(" | ".join(str(v) for v in row.values()))
        return "\n".join(lines)

    def convert_document(doc: dict) -> list[dict]:
        """Convert a TAT-QA document to conversations."""
        conversations = []

        table = doc.get("table", {})
        table_text = format_table(
            table.get("table", []) if isinstance(table, dict) else table
        )
        paragraphs = doc.get("paragraphs", [])
        para_text = " ".join(
            p.get("text", p) if isinstance(p, dict) else str(p) for p in paragraphs
        )

        # Build context
        context_parts = []
        if table_text:
            context_parts.append(f"Table:\n{table_text}")
        if para_text:
            context_parts.append(f"Text: {para_text}")
        context = "\n\n".join(context_parts)

        # Process each question in the document
        for qa in doc.get("questions", []):
            question = qa.get("question", "")
            answer = qa.get("answer", "")

            # Handle different answer formats
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            elif isinstance(answer, dict):
                answer = str(answer.get("value", answer))

            if not question or not answer:
                continue

            user_content = (
                "Answer the question based on the following table and text.\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Put your final answer in \\boxed{}.\n\n"
                "Answer:"
            )
            conversations.append(format_conversation(user_content, f"\\boxed{{{str(answer)}}}"))

        return conversations

    # Convert all documents
    train_data = []
    for doc in tqdm(train_raw, desc="Train"):
        train_data.extend(convert_document(doc))

    test_data = []
    for doc in tqdm(dev_raw, desc="Dev"):
        test_data.extend(convert_document(doc))

    # Carve out validation set from train
    val_size = 100
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]

    # Save
    save_jsonl(train_data, output_dir / "tatqa" / "train.jsonl")
    save_jsonl(val_data, output_dir / "tatqa" / "val.jsonl")
    save_jsonl(test_data, output_dir / "tatqa" / "test.jsonl")

    logger.info(
        f"TAT-QA: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )


def prepare_nl2sql(output_dir: Path) -> None:
    """Prepare NL2SQL dataset for text-to-SQL conversion.

    Uses NovaSky-AI's SQL dataset for natural language to SQL translation.
    Source: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data
    """
    logger.info("Preparing NL2SQL dataset...")

    dataset = load_dataset("NovaSky-AI/SkyRL-SQL-653-data")

    def convert_example(example: dict) -> dict:
        # Extract content from prompt messages and SQL from reward_model.ground_truth
        prompt_messages = example.get("prompt", [])
        reward_model = example.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if reward_model else ""

        # The dataset has prompt as a list of messages (system + user)
        # System message has task instructions, user message has schema and question
        system_content = ""
        user_content_raw = ""
        for msg in prompt_messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content_raw = msg.get("content", "")

        # Combine system instructions with user content (schema + question)
        # Simplify the system prompt for our use case
        user_content = (
            "You are a SQL expert. Generate a valid SQL query to answer the question "
            "based on the provided database schema.\n\n"
            f"{user_content_raw}\n\n"
            "Provide your solution as a SQL query within a ```sql markdown block.\n\n"
        )

        # Clean up SQL
        sql = ground_truth.strip() if ground_truth else ""

        return format_conversation(user_content, sql)

    # Convert all data
    train_split = dataset.get("train", dataset.get("all", None))  # type: ignore[attr-defined]
    if train_split is None:
        # Try loading without split
        dataset = load_dataset("NovaSky-AI/SkyRL-SQL-653-data", split="train")
        full_data = [convert_example(ex) for ex in tqdm(dataset, desc="Converting")]
    else:
        full_data = [convert_example(ex) for ex in tqdm(train_split, desc="Converting")]

    # Split: train / val (50) / test (100)
    val_size = 50
    test_size = 100
    test_data = full_data[-test_size:]
    val_data = full_data[-(test_size + val_size) : -test_size]
    train_data = full_data[: -(test_size + val_size)]

    # Save
    save_jsonl(train_data, output_dir / "nl2sql" / "train.jsonl")
    save_jsonl(val_data, output_dir / "nl2sql" / "val.jsonl")
    save_jsonl(test_data, output_dir / "nl2sql" / "test.jsonl")

    logger.info(
        f"NL2SQL: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )


TASK_PREPARERS = {
    "banking77": prepare_banking77,
    "pubmedqa": prepare_pubmedqa,
    "tatqa": prepare_tatqa,
    "nl2sql": prepare_nl2sql,
}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare enterprise task datasets in Oumi format"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASK_PREPARERS.keys()),
        help="Specific task to prepare",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all tasks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/enterprise",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logging level",
    )
    args = parser.parse_args()

    update_logger_level("oumi", level=args.log_level)

    output_dir = Path(args.output_dir)

    if args.all:
        tasks = list(TASK_PREPARERS.keys())
    elif args.task:
        tasks = [args.task]
    else:
        parser.error("Must specify --task or --all")
        return

    logger.info(f"Preparing {len(tasks)} task(s): {tasks}")
    logger.info(f"Output directory: {output_dir}")

    for task in tasks:
        try:
            TASK_PREPARERS[task](output_dir)
        except Exception as e:
            logger.error(f"Failed to prepare {task}: {e}")
            raise

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
