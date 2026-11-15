import sqlite3
import tempfile
from contextlib import closing

import pytest

from oumi.datasets.grpo.rewards.sql_execution_match import (
    _has_top_level_order_by,
    sql_execution_match,
)

SCHEMA = "CREATE TABLE t(x INTEGER); CREATE TABLE u(y INTEGER); CREATE TABLE p(n TEXT);"
# p holds latin-1 bytes (invalid UTF-8), as Spider DBs do.
SEED = (
    "INSERT INTO t VALUES (1),(2),(3); INSERT INTO u VALUES (9);"
    "INSERT INTO p VALUES (CAST(x'636166e9' AS TEXT)), (CAST(x'636166ff' AS TEXT));"
)
INLINE_DB = {"schema_sql": SCHEMA, "seed_sql": SEED}
COUNT = "SELECT count(*) FROM t"
UNION = "SELECT x FROM t UNION SELECT y FROM u"


def _score(solution, gold, db=INLINE_DB):
    """Call the reward with the DB spec where verl puts it: run_sql's create_kwargs."""
    return sql_execution_match(
        data_source="nl2sql",
        solution_str=solution,
        ground_truth=gold,
        extra_info={"tools_kwargs": {"run_sql": {"create_kwargs": db}}},
    )


@pytest.mark.parametrize(
    ("solution", "gold", "expected"),
    [
        # Extraction
        pytest.param(f"Here you go:\n```sql\n{COUNT}\n```", COUNT, 1.0, id="fenced"),
        pytest.param(f"I think the query is:\n{COUNT}", COUNT, 1.0, id="last-line"),
        pytest.param(
            "Query:\nSELECT count(*)\nFROM t\nWHERE x > 0",
            "SELECT count(*) FROM t WHERE x > 0",
            1.0,
            id="multiline",
        ),
        pytest.param("", COUNT, 0.0, id="empty"),
        pytest.param(
            "A:\nSELECT x FROM t\nUNION\nSELECT y FROM u", UNION, 1.0, id="union"
        ),
        pytest.param(
            "A:\nSELECT x FROM t\nUNION ALL\nSELECT y FROM u",
            UNION,
            1.0,
            id="union-all",
        ),
        pytest.param(
            "A:\nWITH b AS (SELECT x FROM t UNION SELECT y FROM u)\nSELECT * FROM b",
            UNION,
            1.0,
            id="cte",
        ),
        pytest.param(
            "A:\nSELECT x FROM t\nUNION\n"
            "SELECT y FROM u WHERE y IN (\nSELECT y FROM u\n)",
            UNION,
            1.0,
            id="nested-subquery",
        ),
        # Taking only the last UNION arm would score this 1.0.
        pytest.param(
            "SELECT x FROM t\nUNION\nSELECT y FROM u", "SELECT y FROM u", 0.0, id="arm"
        ),
        pytest.param(
            "SELECT y FROM u\n<tool_response>{'rows': [[9]]}</tool_response>\n"
            "SELECT x FROM t",
            "SELECT x FROM t",
            1.0,
            id="final-turn-only",
        ),
        # Result comparison
        pytest.param("SELECT x FROM t WHERE x = 1", COUNT, 0.0, id="wrong"),
        pytest.param("SELECT this is not valid sql", COUNT, 0.0, id="invalid-pred"),
        pytest.param(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
            "SELECT count(*) FROM n",
            COUNT,
            0.0,
            id="runaway-pred",
        ),
        pytest.param(
            "SELECT x FROM t WHERE x = 1",
            "SELECT 1 UNION ALL SELECT 1",
            0.0,
            id="multiplicity",
        ),
        # avg() returns 2.0, the gold 2; both the ordered and unordered paths agree.
        pytest.param("SELECT avg(x) FROM t", "SELECT 2", 1.0, id="int-vs-float"),
        pytest.param(
            "SELECT avg(x) FROM t",
            "SELECT 2 ORDER BY 1",
            1.0,
            id="int-vs-float-ordered",
        ),
        pytest.param(
            "SELECT x FROM t",
            "WITH RECURSIVE n(i) AS "
            "(SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT i FROM n",
            1.0,
            id="recursive-gold",
        ),
        pytest.param("SELECT n FROM p", "SELECT n FROM p", 1.0, id="non-utf8"),
        # Lossy decoding would map both rows to U+FFFD and score this 1.0.
        pytest.param(
            "SELECT n FROM p WHERE rowid = 1",
            "SELECT n FROM p WHERE rowid = 2",
            0.0,
            id="non-utf8-differ",
        ),
        # Row order
        pytest.param(
            "SELECT x FROM t ORDER BY x DESC",
            "SELECT x FROM t ORDER BY x ASC",
            0.0,
            id="wrong-order",
        ),
        pytest.param(
            "SELECT x FROM t ORDER BY x ASC",
            "SELECT x FROM t ORDER BY x ASC",
            1.0,
            id="same-order",
        ),
        pytest.param(
            "SELECT x FROM t WHERE x < 3 ORDER BY x DESC",
            "SELECT x FROM t WHERE x IN (SELECT x FROM t ORDER BY x LIMIT 2)",
            1.0,
            id="subquery-order-by-ignored",
        ),
    ],
)
def test_score(solution, gold, expected):
    assert _score(solution, gold) == expected


@pytest.mark.parametrize("solution", [COUNT, "SELECT this is also invalid"])
def test_invalid_gold_sql_raises(solution):
    with pytest.raises(sqlite3.Error):
        _score(solution, "SELECT this is not valid sql")


# The full set of invalid specs is covered by test_database_executable_environment.
@pytest.mark.parametrize("db", [{}, "not a mapping"])
def test_invalid_db_spec_raises(db):
    with pytest.raises(ValueError):
        _score(COUNT, COUNT, db=db)


@pytest.fixture
def db_file(tmp_path):
    """A pre-staged SQLite file (Spider-style) referenced by db_path, not inlined."""
    db = tmp_path / "shared.sqlite"
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript(SCHEMA + SEED)
        conn.commit()
    return db


def test_db_path_scores_against_the_file_and_keeps_it(db_file):
    assert _score(COUNT, COUNT, db={"db_path": str(db_file)}) == 1.0
    assert db_file.exists()


@pytest.mark.parametrize("write", ["DELETE FROM t", "VACUUM INTO '{dest}'"])
def test_write_scores_zero_and_leaves_no_trace(db_file, tmp_path, write):
    dest = tmp_path / "escaped.sqlite"
    assert _score(write.format(dest=dest), COUNT, db={"db_path": str(db_file)}) == 0.0
    assert not dest.exists()
    with closing(sqlite3.connect(db_file)) as conn:
        assert conn.execute(COUNT).fetchone()[0] == 3


def test_inline_snapshot_is_deleted_after_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _score(COUNT, COUNT)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT x FROM t ORDER BY x", True),
        ("select x from t order by x", True),
        ("SELECT x FROM t ORDER \n BY x", True),
        ("SELECT x FROM t ORDER /* gap */ BY x", True),
        ("SELECT x FROM t ORDER BY x -- trailing note", True),
        ("SELECT x FROM t ORDERBY x", False),
        ("SELECT x FROM t ORDER x", False),
        # Quoted text and comments are not clauses.
        ("SELECT 'ORDER BY (x)' FROM t", False),
        ('SELECT "ORDER BY" FROM t', False),
        ("SELECT x FROM t -- ORDER BY (x)\n", False),
        ("SELECT x FROM t /* ORDER BY (x) */", False),
        # A comment marker inside a string must not swallow a real ORDER BY.
        ("SELECT x FROM t WHERE n = 'a--b' ORDER BY x", True),
        ("SELECT x FROM t WHERE n = 'a/*b' ORDER BY x", True),
        # Only the outer query's ORDER BY counts.
        ("SELECT x FROM (SELECT y FROM t ORDER BY y) z ORDER BY x", True),
        ("SELECT x FROM t WHERE x IN (SELECT x FROM t ORDER BY x LIMIT 2)", False),
        ("SELECT rank() OVER (ORDER BY x) FROM t", False),
        ("WITH c AS (SELECT x FROM t ORDER BY x) SELECT * FROM c", False),
    ],
)
def test_has_top_level_order_by(sql, expected):
    assert _has_top_level_order_by(sql) is expected
