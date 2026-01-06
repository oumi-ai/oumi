# src/oumi/datasets/grpo/rewards/text2sql_reward.py

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional,Sequence
from func_timeout import func_timeout, FunctionTimedOut

from oumi.core.registry import register, RegistryType
from oumi.utils.logging import logger


THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"

@dataclass
class ExtractSqlResult:
    success: bool
    thoughts: Optional[List[str]]
    solution_sql: Optional[str]

def _extract_sql(output: str) -> ExtractSqlResult:
    """Extract SQL from model output.

    Priority:
      1. <sql>...</sql> block (SkyRL-style)
      2. Last ```sql ... ``` fenced block
    """
    if output.count(SOLUTION_START) != 1:
        return ExtractSqlResult(False, None, None)
    
    pre_solution, tail = output.split(SOLUTION_START, 1)

    if tail.count(SOLUTION_END) != 1:
        return ExtractSqlResult(False, None, None)

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return ExtractSqlResult(False, None, None)

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return ExtractSqlResult(False, None, None)

    for m in re.finditer(r"</observation>", pre_solution, re.I):
        rest = pre_solution[m.end() :].lstrip()
        if not rest.lower().startswith(THINK_START):
            return ExtractSqlResult(False, None, None)

    return ExtractSqlResult(True, thoughts, solution_text.strip())

@dataclass
class SqlResult:
    db_file: str
    sql: str
    rows: Optional[Sequence[tuple]]
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None

def _execute_sql(db_path: str, sql: str) -> Sequence[tuple]:
    """Execute SQL on a (read-only) sqlite3 database and return all rows."""
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            con.execute("BEGIN TRANSACTION;")
            cur.execute(sql)
            rows = cur.fetchall()
            con.rollback()
        logger.info('Successfully executed')
        return rows
    except Exception as e:
        logger.error(f"Error executing SQL: {e}, db file: {db_path}")
        return None

def _execute_sql_wrapper(
    db_file: str,
    sql: str,
    timeout: float,
    output_str: str | None = None,
) -> SqlResult:
    """
    High-level function: adds timeout, logging, and wraps into SqlResult.
    """
    try:
        rows = func_timeout(timeout, _execute_sql, args=(db_file, sql))
        logger.info("Successfully executed SQL on %s", db_file)
        return SqlResult(
            db_file=db_file,
            sql=sql,
            rows=rows,
            success=True,
            output=output_str,
        )

    except KeyboardInterrupt:
        sys.exit(0)

    except FunctionTimedOut:
        msg = "SQL timed out"
        logger.error("SQL:\n%s\nTime Out!", sql)
        logger.error("-" * 30)
        return SqlResult(
            db_file=db_file,
            sql=sql,
            rows=None,
            success=False,
            output=output_str,
            error=msg,
        )

    except Exception as e:
        msg = f"Error executing SQL: {e}"
        logger.error("%s, db file: %s", msg, db_file)
        return SqlResult(
            db_file=db_file,
            sql=sql,
            rows=None,
            success=False,
            output=output_str,
            error=msg,
        )


def _normalize_rows(rows: Iterable[Tuple[Any, ...]]) -> List[Tuple[str, ...]]:
    """Canonicalize rows for comparison: stringify + sort."""
    normalized = [tuple("" if v is None else str(v) for v in row) for row in rows]
    return sorted(normalized)


@register("text2sql", RegistryType.REWARD_FUNCTION)
def text2sql_reward(  # type: ignore[override]
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Dict[str, Any],
    *,
    format_score: float = 0.1,
    exec_score: float = 1.0,
) -> float:
    """Reward for Text2SQL.

    Expected ground_truth fields:
      - gold_sql: reference SQL string
      - db_path: path to sqlite DB (or pass via extra_info)
    """
    extract_sql_result = _extract_sql(solution_str)
    if not extract_sql_result.success:
        # No extractable SQL → no reward
        return 0.0

    ref_sql = ground_truth["gold_sql"]
    pred_sql = extract_sql_result.solution_sql
    db_path = extra_info.get("db_path")
    if not db_path:
        raise ValueError("db_path must be provided via extra_info")

    # Try executing predicted SQL; if it fails we still give a small format reward
    try:
        pred_rows = _normalize_rows(_execute_sql(db_path, pred_sql))
    except Exception:
        # Parsed but failed to execute → only format reward
        return format_score

    # Execute gold SQL (assumed to be valid)
    gold_rows = _normalize_rows(_execute_sql(db_path, gold_sql))

    return exec_score if pred_rows == gold_rows else format_score
