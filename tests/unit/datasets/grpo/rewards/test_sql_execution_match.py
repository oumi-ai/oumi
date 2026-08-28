import sqlite3
import tempfile
from pathlib import Path

import pytest

from oumi.datasets.grpo.rewards.sql_execution_match import (
    _has_top_level_order_by,
    sql_execution_match,
)

SCHEMA = "CREATE TABLE t(x INTEGER);"
SEED = "INSERT INTO t VALUES (1),(2),(3);"


def _extra(**create_kwargs):
    """The DB spec as verl delivers it: run_sql's create_kwargs under tools_kwargs."""
    return {"tools_kwargs": {"run_sql": {"create_kwargs": create_kwargs}}}


EXTRA = _extra(schema_sql=SCHEMA, seed_sql=SEED)


def test_matching_sql_scores_one():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="Here you go:\n```sql\nSELECT count(*) FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert reward == 1.0


def test_wrong_sql_scores_zero():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x = 1\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert reward == 0.0


def test_ordered_result_wrong_order_scores_zero():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t ORDER BY x DESC\n```",
        ground_truth="SELECT x FROM t ORDER BY x ASC",
        extra_info=EXTRA,
    )
    assert reward == 0.0


def test_ordered_result_matches_in_row_order():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t ORDER BY x ASC\n```",
        ground_truth="SELECT x FROM t ORDER BY x ASC",
        extra_info=EXTRA,
    )
    assert reward == 1.0


def test_no_predicted_sql_scores_zero():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert reward == 0.0


def test_invalid_predicted_sql_scores_zero():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT this is not valid sql\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert reward == 0.0


def test_invalid_gold_sql_raises():
    with pytest.raises(sqlite3.Error):
        sql_execution_match(
            data_source="nl2sql",
            solution_str="```sql\nSELECT count(*) FROM t\n```",
            ground_truth="SELECT this is not valid sql",
            extra_info=EXTRA,
        )


def test_invalid_gold_sql_raises_even_when_prediction_is_invalid():
    with pytest.raises(sqlite3.Error):
        sql_execution_match(
            data_source="nl2sql",
            solution_str="```sql\nSELECT this is also invalid sql\n```",
            ground_truth="SELECT this is not valid sql",
            extra_info=EXTRA,
        )


def test_extracts_last_non_empty_line_when_no_fence():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="I think the query is:\nSELECT count(*) FROM t",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert reward == 1.0


def test_extracts_multiline_sql_without_fence():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="I think the query is:\nSELECT count(*)\nFROM t\nWHERE x > 0",
        ground_truth="SELECT count(*) FROM t WHERE x > 0",
        extra_info=EXTRA,
    )
    assert reward == 1.0


TWO_TABLES = _extra(
    schema_sql="CREATE TABLE t(x INTEGER); CREATE TABLE u(y INTEGER);",
    seed_sql="INSERT INTO t VALUES (1); INSERT INTO u VALUES (9);",
)


@pytest.mark.parametrize(
    "unfenced_sql",
    [
        "SELECT x FROM t\nUNION\nSELECT y FROM u",
        "SELECT x FROM t\nUNION ALL\nSELECT y FROM u",
        "WITH both AS (SELECT x FROM t UNION SELECT y FROM u)\nSELECT * FROM both",
        "SELECT x FROM t\nUNION\nSELECT y FROM u WHERE y IN (\nSELECT y FROM u\n)",
    ],
)
def test_unfenced_multi_statement_sql_is_extracted_whole(unfenced_sql):
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str=f"The answer is:\n{unfenced_sql}",
        ground_truth="SELECT x FROM t UNION SELECT y FROM u",
        extra_info=TWO_TABLES,
    )
    assert reward == 1.0


def test_a_trailing_union_arm_alone_does_not_score_a_match():
    # Extracting only the last arm would score this against gold `SELECT y FROM u`.
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="SELECT x FROM t\nUNION\nSELECT y FROM u",
        ground_truth="SELECT y FROM u",
        extra_info=TWO_TABLES,
    )
    assert reward == 0.0


def test_only_the_final_turn_of_a_tool_transcript_is_scored():
    transcript = (
        "SELECT y FROM u\n"
        "<tool_response>{'rows': [[9]]}</tool_response>\n"
        "SELECT x FROM t"
    )
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str=transcript,
        ground_truth="SELECT x FROM t",
        extra_info=TWO_TABLES,
    )
    assert reward == 1.0


def test_a_runaway_prediction_scores_zero_instead_of_hanging():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str=(
            "```sql\nWITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
            "SELECT count(*) FROM n\n```"
        ),
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert reward == 0.0


def _make_shared_db(tmp_path):
    """A pre-staged SQLite file (Spider-style) referenced by db_path, not inlined."""
    db = tmp_path / "shared.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA + SEED)
    conn.commit()
    conn.close()
    return db


def test_db_path_matching_scores_one_and_keeps_file(tmp_path):
    db = _make_shared_db(tmp_path)
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT count(*) FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert reward == 1.0
    assert db.exists()


def test_db_path_wrong_scores_zero(tmp_path):
    db = _make_shared_db(tmp_path)
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x = 1\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert reward == 0.0


def test_db_path_write_pred_scores_zero_without_mutating(tmp_path):
    db = _make_shared_db(tmp_path)
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nDELETE FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert reward == 0.0
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 3
    conn.close()


def test_vacuum_into_scores_zero_without_creating_file(tmp_path):
    db = _make_shared_db(tmp_path)
    destination = tmp_path / "escaped.sqlite"
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str=f"```sql\nVACUUM INTO '{destination}'\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert reward == 0.0
    assert not destination.exists()


def test_top_level_order_by_ignores_subqueries():
    assert _has_top_level_order_by("SELECT x FROM t ORDER BY x") is True
    assert (
        _has_top_level_order_by(
            "SELECT x FROM t WHERE x IN (SELECT x FROM t ORDER BY x LIMIT 2)"
        )
        is False
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 'ORDER BY (x)' FROM t", False),
        ('SELECT "ORDER BY" FROM t', False),
        ("SELECT x FROM t -- ORDER BY (x)\n", False),
        ("SELECT x FROM t /* ORDER BY (x) */", False),
        ("SELECT x FROM t ORDER /* gap */ BY x", True),
        # Keyword matching is case- and whitespace-insensitive.
        ("select x from t order by x", True),
        ("SELECT x FROM t OrDeR By x", True),
        ("SELECT x FROM t ORDER \n BY x", True),
        ("SELECT x FROM t ORDER \t BY x", True),
        ("SELECT x FROM t ORDERBY x", False),
        ("SELECT x FROM t ORDER x", False),
        # Nested: only the outer query's ORDER BY counts.
        ("SELECT x FROM (SELECT y FROM t ORDER BY y) z ORDER BY x", True),
        ("SELECT x FROM t WHERE a IN (SELECT b FROM u WHERE c IN (ORDER BY d))", False),
        ("SELECT rank() OVER (ORDER BY x) FROM t", False),
        ("WITH c AS (SELECT x FROM t ORDER BY x) SELECT * FROM c", False),
        # A comment marker inside a string literal must not swallow the rest of
        # the line -- stripping comments before quotes hid a real ORDER BY.
        ("SELECT x FROM t WHERE n = 'a--b' ORDER BY x", True),
        ("SELECT x FROM t WHERE n = '--' ORDER BY x", True),
        ("SELECT x FROM t WHERE n = 'a/*b' ORDER BY x", True),
        ("SELECT x FROM t ORDER BY x -- trailing note", True),
    ],
)
def test_top_level_order_by_ignores_quoted_text_and_comments(sql, expected):
    assert _has_top_level_order_by(sql) is expected


def test_subquery_order_by_does_not_force_positional_match():
    # The outer results are equivalent despite their different row order.
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x < 3 ORDER BY x DESC\n```",
        ground_truth="SELECT x FROM t WHERE x IN (SELECT x FROM t ORDER BY x LIMIT 2)",
        extra_info=EXTRA,
    )
    assert reward == 1.0


def test_non_utf8_text_does_not_abort_scoring(tmp_path):
    """Spider DBs carry latin-1 bytes; decoding must not raise mid-run."""
    db = tmp_path / "latin1.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE people(last_name TEXT)")
    # 'Albarracín' in latin-1 (0xED) stored with TEXT affinity — invalid UTF-8.
    conn.execute("INSERT INTO people VALUES (CAST(x'416c626172726163ed6e' AS TEXT))")
    conn.commit()
    conn.close()

    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT last_name FROM people\n```",
        ground_truth="SELECT last_name FROM people",
        extra_info=_extra(db_path=str(db)),
    )
    assert reward == 1.0


def test_non_utf8_bytes_that_differ_do_not_compare_equal(tmp_path):
    """Lossy decoding would map both rows to U+FFFD and score the wrong row 1.0."""
    db = tmp_path / "bytes.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE p(n TEXT)")
    conn.execute("INSERT INTO p VALUES (CAST(x'636166e9' AS TEXT))")
    conn.execute("INSERT INTO p VALUES (CAST(x'636166ff' AS TEXT))")
    conn.commit()
    conn.close()

    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT n FROM p WHERE rowid = 1\n```",
        ground_truth="SELECT n FROM p WHERE rowid = 2",
        extra_info=_extra(db_path=str(db)),
    )
    assert reward == 0.0


@pytest.mark.parametrize("gold_suffix", ["", " ORDER BY 1"])
def test_value_equality_does_not_depend_on_gold_order_by(gold_suffix):
    # avg() returns 2.0 and the gold returns 2; both branches must agree.
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT avg(x) FROM t\n```",
        ground_truth=f"SELECT 2{gold_suffix}",
        extra_info=EXTRA,
    )
    assert reward == 1.0


def test_duplicate_rows_must_match_in_multiplicity():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x = 1\n```",
        ground_truth="SELECT 1 UNION ALL SELECT 1",
        extra_info=EXTRA,
    )
    assert reward == 0.0


def test_recursive_cte_ground_truth_is_authorized():
    reward = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t\n```",
        ground_truth=(
            "WITH RECURSIVE n(i) AS "
            "(SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT i FROM n"
        ),
        extra_info=EXTRA,
    )
    assert reward == 1.0


def test_inline_snapshot_is_deleted_after_scoring():
    temp_dir = Path(tempfile.gettempdir())
    before = set(temp_dir.glob("oumi_snapshot_*.sqlite"))
    sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT count(*) FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert set(temp_dir.glob("oumi_snapshot_*.sqlite")) == before


@pytest.mark.parametrize(
    "create_kwargs",
    [
        {},
        {"db_path": "a.sqlite", "schema_sql": SCHEMA},
        {"seed_sql": SEED},
    ],
)
def test_invalid_db_spec_raises(create_kwargs):
    with pytest.raises(ValueError):
        sql_execution_match(
            data_source="nl2sql",
            solution_str="```sql\nSELECT count(*) FROM t\n```",
            ground_truth="SELECT count(*) FROM t",
            extra_info=_extra(**create_kwargs),
        )
