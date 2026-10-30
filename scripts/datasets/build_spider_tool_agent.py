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

"""Build Spider NL2SQL tool-agent train/val rows in Oumi Conversation format."""

import argparse
import functools
import json
import random
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from oumi.utils.logging import logger

_SYSTEM_PROMPT = (
    "You are a SQL assistant for a SQLite database with this schema:\n{schema}\n"
    "You may call the tool run_sql(query) to execute a SQL query and inspect "
    "results. When you are done, give your final answer as a single SQL query "
    "inside a ```sql code block."
)


def _read_only(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # Decode without raising, as the reward does, so we keep rows it could score;
    # backslashreplace also keeps the DDL we splice into the prompt JSON-encodable.
    connection.text_factory = lambda b: b.decode("utf-8", "backslashreplace")
    return connection


@functools.cache
def _schema_ddl(db_path: Path) -> str:
    with closing(_read_only(db_path)) as connection:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
    return "\n\n".join(row[0].strip() for row in rows)


def _gold_executes(db_path: Path, sql: str) -> bool:
    """Whether the gold query runs against its own DB (some Spider rows don't)."""
    try:
        with closing(_read_only(db_path)) as connection:
            connection.execute(sql).fetchall()
        return True
    except sqlite3.Error:
        return False


def _build_rows(
    examples: list[dict[str, Any]],
    db_root: Path,
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if 0 < limit < len(examples):
        examples = random.Random(seed).sample(examples, limit)

    rows = []
    dropped = 0
    for example in examples:
        db_id = example["db_id"]
        db_path = db_root / db_id / f"{db_id}.sqlite"
        schema = _schema_ddl(db_path)
        # An unscoreable gold makes the row untrainable and aborts the reward mid-run.
        if not _gold_executes(db_path, example["query"]):
            dropped += 1
            logger.warning(f"{split}: dropped {db_id} gold {example['query']!r}")
            continue
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT.format(schema=schema),
                    },
                    {"role": "user", "content": example["question"]},
                ],
                "metadata": {
                    "agent_name": "tool_agent",
                    "ground_truth": example["query"],
                    "tools_kwargs": {
                        "run_sql": {
                            "create_kwargs": {"db_path": str(db_path.absolute())}
                        }
                    },
                },
            }
        )
    logger.info(f"{split}: kept {len(rows)} row(s), dropped {dropped} unscoreable")
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def main() -> None:
    """Build train and validation JSONL files from a Spider release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spider-root",
        type=Path,
        required=True,
        help="Spider release root holding train_spider.json and dev.json.",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        help="Directory of <db_id>/<db_id>.sqlite files. Defaults to "
        "<spider-root>/database.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/grpo_verl_nl2sql"),
        help="Where train.jsonl and val.jsonl are written.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Sample this many train examples. Non-positive means all of them.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Sample this many dev examples. Non-positive means all of them.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the two limit samples."
    )
    args = parser.parse_args()

    db_root = args.db_root or args.spider_root / "database"
    train_examples = json.loads(
        (args.spider_root / "train_spider.json").read_text(encoding="utf-8")
    )
    val_examples = json.loads(
        (args.spider_root / "dev.json").read_text(encoding="utf-8")
    )
    _write_jsonl(
        _build_rows(
            train_examples,
            db_root,
            split="train",
            limit=args.train_limit,
            seed=args.seed,
        ),
        args.out_dir / "train.jsonl",
    )
    _write_jsonl(
        _build_rows(
            val_examples, db_root, split="val", limit=args.val_limit, seed=args.seed
        ),
        args.out_dir / "val.jsonl",
    )


if __name__ == "__main__":
    main()
