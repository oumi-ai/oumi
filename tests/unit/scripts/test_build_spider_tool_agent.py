import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_builds_spider_tool_agent_rows(tmp_path):
    repo_root = Path(__file__).parents[3]
    script = repo_root / "scripts" / "datasets" / "build_spider_tool_agent.py"
    spider_root = tmp_path / "spider"
    db_root = spider_root / "database"
    database_dir = db_root / "company"
    database_dir.mkdir(parents=True)
    db_path = database_dir / "company.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            "CREATE TABLE employees(name TEXT);"
            "INSERT INTO employees VALUES ('Ada'), ('Bo');"
        )

    example = {
        "db_id": "company",
        "question": "How many employees are there?",
        "query": "SELECT count(*) FROM employees",
    }
    spider_root.mkdir(exist_ok=True)
    (spider_root / "train_spider.json").write_text(json.dumps([example]))
    (spider_root / "dev.json").write_text(json.dumps([example]))
    output_dir = tmp_path / "prepared"

    # Relative args + a different cwd: emitted db_path must still be absolute.
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
    row = json.loads((output_dir / "train.jsonl").read_text())
    assert row["messages"][-1] == {
        "role": "user",
        "content": "How many employees are there?",
    }
    assert "CREATE TABLE employees" in row["messages"][0]["content"]
    assert row["metadata"] == {
        "agent_name": "tool_agent",
        "ground_truth": "SELECT count(*) FROM employees",
        "tools_kwargs": {"run_sql": {"create_kwargs": {"db_path": str(db_path)}}},
    }
    assert (output_dir / "val.jsonl").is_file()
