import sqlite3

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
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="Here you go:\n```sql\nSELECT count(*) FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert r == 1.0


def test_wrong_sql_scores_zero():
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x = 1\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert r == 0.0


def test_ordered_result_wrong_order_scores_zero():
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t ORDER BY x DESC\n```",
        ground_truth="SELECT x FROM t ORDER BY x ASC",
        extra_info=EXTRA,
    )
    assert r == 0.0


def test_ordered_result_matches_in_row_order():
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t ORDER BY x ASC\n```",
        ground_truth="SELECT x FROM t ORDER BY x ASC",
        extra_info=EXTRA,
    )
    assert r == 1.0


def test_no_predicted_sql_scores_zero():
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert r == 0.0


def test_invalid_predicted_sql_scores_zero():
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT this is not valid sql\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert r == 0.0


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
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="I think the query is:\nSELECT count(*) FROM t",
        ground_truth="SELECT count(*) FROM t",
        extra_info=EXTRA,
    )
    assert r == 1.0


def test_extracts_multiline_sql_without_fence():
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="I think the query is:\nSELECT count(*)\nFROM t\nWHERE x > 0",
        ground_truth="SELECT count(*) FROM t WHERE x > 0",
        extra_info=EXTRA,
    )
    assert r == 1.0


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
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT count(*) FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert r == 1.0
    assert db.exists()


def test_db_path_wrong_scores_zero(tmp_path):
    db = _make_shared_db(tmp_path)
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x = 1\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert r == 0.0


def test_db_path_write_pred_scores_zero_without_mutating(tmp_path):
    db = _make_shared_db(tmp_path)
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nDELETE FROM t\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert r == 0.0
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 3
    conn.close()


def test_vacuum_into_scores_zero_without_creating_file(tmp_path):
    db = _make_shared_db(tmp_path)
    destination = tmp_path / "escaped.sqlite"
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str=f"```sql\nVACUUM INTO '{destination}'\n```",
        ground_truth="SELECT count(*) FROM t",
        extra_info=_extra(db_path=str(db)),
    )
    assert r == 0.0
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
    ],
)
def test_top_level_order_by_ignores_quoted_text_and_comments(sql, expected):
    assert _has_top_level_order_by(sql) is expected


def test_subquery_order_by_does_not_force_positional_match():
    # The outer results are equivalent despite their different row order.
    r = sql_execution_match(
        data_source="nl2sql",
        solution_str="```sql\nSELECT x FROM t WHERE x < 3 ORDER BY x DESC\n```",
        ground_truth="SELECT x FROM t WHERE x IN (SELECT x FROM t ORDER BY x LIMIT 2)",
        extra_info=EXTRA,
    )
    assert r == 1.0
