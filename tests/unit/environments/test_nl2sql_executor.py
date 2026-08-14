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

import sqlite3

from oumi.environments.examples.nl2sql import run_sql


def _conn():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1),(2),(3);"
    )
    return connection


def test_run_sql_returns_rows():
    result = run_sql(
        arguments={"query": "SELECT count(*) AS n FROM t"}, context=_conn()
    )
    assert result.output == {"columns": ["n"], "rows": [[3]]}


def test_run_sql_hex_encodes_blobs():
    connection = _conn()
    connection.executescript("CREATE TABLE b(v BLOB); INSERT INTO b VALUES (x'DEAD');")
    result = run_sql(arguments={"query": "SELECT v FROM b"}, context=connection)
    assert result.output == {"columns": ["v"], "rows": [["dead"]]}


def test_run_sql_returns_no_columns_for_a_statement_without_a_result_set():
    result = run_sql(arguments={"query": "CREATE TABLE z(a)"}, context=_conn())
    assert result.output == {"columns": [], "rows": []}


def test_run_sql_returns_error_on_bad_sql():
    result = run_sql(
        arguments={"query": "SELECT * FROM no_such_table"}, context=_conn()
    )
    assert "no_such_table" in str(result.output)
