import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from oumi.core.types.tool_call import ToolResult

pytest.importorskip("verl")

from verl.tools.schemas import (  # pyright: ignore[reportMissingImports]
    OpenAIFunctionToolSchema,
)

from oumi.core.trainers.verl_agentic.oumi_verl_tool import OumiVerlTool


def run_sql(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    """Local SQL executor; keeps this test independent of the nl2sql example."""
    cursor = context.execute(arguments["query"])
    columns = [d[0] for d in cursor.description] if cursor.description else []
    rows = [list(r) for r in cursor.fetchall()]
    return ToolResult(output={"columns": columns, "rows": rows})


class _FakeAgentData(SimpleNamespace):
    """A weak-referenceable agent data stub."""


def test_execute_routes_into_shared_env(tmp_path):
    cfg_path = tmp_path / "env.yaml"
    cfg_path.write_text(
        "environments:\n- id: db\n  env_type: database\n"
        "  env_kwargs: {schema_sql: 'CREATE TABLE t(x);'}\n"
        "  tools:\n  - {id: run_sql, name: run_sql, description: run,\n"
        "     parameters: {type: object, properties: {query: {type: string}}, "
        "required: [query]},\n"
        f"     executor: {__name__}.run_sql, read_only: true}}\n"
    )
    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "run_sql",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    )
    tool = OumiVerlTool(config={"oumi_env_config": str(cfg_path)}, tool_schema=schema)
    ad = _FakeAgentData(
        request_id="execute-routes-into-shared-env",
        extra_fields={},
        tools_kwargs={
            "run_sql": {
                "create_kwargs": {
                    "schema_sql": "CREATE TABLE t(x INTEGER);",
                    "seed_sql": "INSERT INTO t VALUES (1),(2);",
                }
            }
        },
    )

    iid, _ = asyncio.run(tool.create())
    resp, reward, _metrics = asyncio.run(
        tool.execute(iid, {"query": "SELECT count(*) FROM t"}, agent_data=ad)
    )
    assert reward == 0.0
    assert "2" in resp.text
