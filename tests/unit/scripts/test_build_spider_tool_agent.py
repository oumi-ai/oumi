import json
import sqlite3
import subprocess
import sys
from pathlib import Path

_TRAIN_EXAMPLE = {
    "db_id": "company",
    "question": "How many employees are there?",
    "query": "SELECT count(*) FROM employees",
}
_DEV_EXAMPLE = {
    "db_id": "company",
    "question": "Who works here?",
    "query": "SELECT name FROM employees",
}
_UNSCOREABLE_EXAMPLE = {
    "db_id": "company",
    "question": "How many contractors are there?",
    "query": "SELECT count(*) FROM contractors",
}


def _spider_release(tmp_path, *, train, dev):
    """Lay out a minimal Spider release and return (spider_root, db_path)."""
    spider_root = tmp_path / "spider"
    database_dir = spider_root / "database" / "company"
    database_dir.mkdir(parents=True)
    db_path = database_dir / "company.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            "CREATE TABLE employees(name TEXT);"
            "INSERT INTO employees VALUES ('Ada'), ('Bo');"
        )
    (spider_root / "train_spider.json").write_text(json.dumps(train))
    (spider_root / "dev.json").write_text(json.dumps(dev))
    return spider_root, db_path


def _build(tmp_path):
    """Run the builder with relative args from a different cwd."""
    script = (
        Path(__file__).parents[3]
        / "scripts"
        / "datasets"
        / "build_spider_tool_agent.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--spider-root",
            "spider",
            "--db-root",
            "spider/database",
            "--out-dir",
            "prepared",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return tmp_path / "prepared"


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_builds_spider_tool_agent_rows(tmp_path):
    _, db_path = _spider_release(tmp_path, train=[_TRAIN_EXAMPLE], dev=[_DEV_EXAMPLE])
    output_dir = _build(tmp_path)

    (row,) = _rows(output_dir / "train.jsonl")
    assert row["messages"][-1] == {
        "role": "user",
        "content": "How many employees are there?",
    }
    assert "CREATE TABLE employees" in row["messages"][0]["content"]
    # Relative args + a different cwd: the emitted db_path must still be absolute.
    assert row["metadata"] == {
        "agent_name": "tool_agent",
        "ground_truth": "SELECT count(*) FROM employees",
        "tools_kwargs": {"run_sql": {"create_kwargs": {"db_path": str(db_path)}}},
    }


def test_val_rows_come_from_dev_not_train(tmp_path):
    _spider_release(tmp_path, train=[_TRAIN_EXAMPLE], dev=[_DEV_EXAMPLE])
    output_dir = _build(tmp_path)

    (row,) = _rows(output_dir / "val.jsonl")
    assert row["messages"][-1]["content"] == "Who works here?"
    assert row["metadata"]["ground_truth"] == "SELECT name FROM employees"


def test_rows_whose_gold_does_not_execute_are_dropped(tmp_path):
    _spider_release(
        tmp_path,
        train=[_TRAIN_EXAMPLE, _UNSCOREABLE_EXAMPLE],
        dev=[_UNSCOREABLE_EXAMPLE],
    )
    output_dir = _build(tmp_path)

    kept = _rows(output_dir / "train.jsonl")
    assert [row["metadata"]["ground_truth"] for row in kept] == [
        "SELECT count(*) FROM employees"
    ]
    assert (output_dir / "val.jsonl").read_text() == ""


def test_gold_selecting_non_utf8_text_is_kept(tmp_path):
    example = {
        "db_id": "company",
        "question": "Which names are on file?",
        "query": "SELECT name FROM people",
    }
    _, db_path = _spider_release(tmp_path, train=[example], dev=[_TRAIN_EXAMPLE])
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE people(name TEXT)")
        connection.execute("INSERT INTO people VALUES (CAST(x'636166e9' AS TEXT))")
    output_dir = _build(tmp_path)

    (row,) = _rows(output_dir / "train.jsonl")
    assert row["metadata"]["ground_truth"] == "SELECT name FROM people"
