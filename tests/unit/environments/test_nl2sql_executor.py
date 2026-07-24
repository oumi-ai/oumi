import sqlite3

from oumi.environments.examples.nl2sql import run_sql


def _conn():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1),(2),(3);")
    return c


def test_run_sql_returns_rows():
    res = run_sql(arguments={"query": "SELECT count(*) AS n FROM t"}, context=_conn())
    assert res.output == {"columns": ["n"], "rows": [[3]]}


def test_run_sql_hex_encodes_blobs():
    c = _conn()
    c.executescript("CREATE TABLE b(v BLOB); INSERT INTO b VALUES (x'DEAD');")
    res = run_sql(arguments={"query": "SELECT v FROM b"}, context=c)
    assert res.output == {"columns": ["v"], "rows": [["dead"]]}


def test_run_sql_returns_error_on_bad_sql():
    res = run_sql(arguments={"query": "SELECT * FROM no_such_table"}, context=_conn())
    assert isinstance(res.output, dict)
    assert "no_such_table" in str(res.output["error"])
