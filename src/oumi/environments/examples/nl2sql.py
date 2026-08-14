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

"""Read-only `run_sql` tool executor for the database environment."""

from __future__ import annotations

import sqlite3
from typing import Any

from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_session import query_work_budget


def run_sql(arguments: dict[str, Any], context: sqlite3.Connection) -> ToolResult:
    """Execute the ``query`` argument against the rollout's bound SQLite connection.

    Returns:
        ``{"columns", "rows"}`` on success, ``{"error": <sqlite message>}`` if the
        query does not run, is unauthorized, or exceeds its work budget.
    """
    # Benchmark DBs carry non-UTF-8 text and the default factory raises
    # UnicodeDecodeError, which is not a sqlite3.Error and would kill the rollout.
    # backslashreplace keeps every byte recoverable and stays JSON-encodable.
    context.text_factory = lambda b: b.decode("utf-8", "backslashreplace")
    try:
        with query_work_budget(context):
            cursor = context.execute(arguments["query"])
            columns = [d[0] for d in cursor.description] if cursor.description else []
            # BLOBs come back as bytes, which ToolResult's JsonValue output rejects.
            rows = [
                [cell.hex() if isinstance(cell, bytes) else cell for cell in row]
                for row in cursor.fetchall()
            ]
    except sqlite3.Error as e:
        return ToolResult(output={"error": str(e)})
    return ToolResult(output={"columns": columns, "rows": rows})
