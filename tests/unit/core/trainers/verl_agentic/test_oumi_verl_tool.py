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


def _tool_schema(name: str = "run_sql") -> OpenAIFunctionToolSchema:
    """Mirrors the run_sql tool entry written by _make_tool.

    `description` is required by verl's schema, so it cannot be omitted even
    where a test only cares about the tool name.
    """
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "run",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    )


def _make_tool(tmp_path) -> OumiVerlTool:
    cfg_path = tmp_path / "env.yaml"
    cfg_path.write_text(
        "environments:\n- id: db\n  name: db\n  description: test db\n"
        "  env_type: database\n"
        "  env_kwargs: {schema_sql: 'CREATE TABLE t(x);'}\n"
        "  tools:\n  - {id: run_sql, name: run_sql, description: run,\n"
        "     parameters: {type: object, properties: {query: {type: string}}, "
        "required: [query]},\n"
        f"     executor: {__name__}.run_sql, read_only: true}}\n"
    )
    return OumiVerlTool(
        config={"oumi_env_config": str(cfg_path)}, tool_schema=_tool_schema()
    )


def _agent_data(request_id: str) -> _FakeAgentData:
    return _FakeAgentData(
        request_id=request_id,
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


def test_execute_routes_into_shared_env(tmp_path):
    tool = _make_tool(tmp_path)
    ad = _agent_data("execute-routes-into-shared-env")

    iid, _ = asyncio.run(tool.create())
    resp, reward, _metrics = asyncio.run(
        tool.execute(iid, {"query": "SELECT count(*) FROM t"}, agent_data=ad)
    )
    assert reward == 0.0
    assert resp.text is not None
    assert "2" in resp.text


def test_invalid_env_config_is_rejected_at_construction(tmp_path):
    """The config is validated when the tool is built, not mid-rollout."""
    cfg_path = tmp_path / "env.yaml"
    cfg_path.write_text(
        "environments:\n- id: db\n  name: db\n  description: d\n  env_type: nope\n"
    )
    with pytest.raises(ValueError, match="Unknown env_type 'nope'"):
        OumiVerlTool(
            config={"oumi_env_config": str(cfg_path)}, tool_schema=_tool_schema()
        )


def test_tool_name_missing_from_env_config_is_rejected_at_construction(tmp_path):
    """The name is spelled out twice; a mismatch must not wait for the first call."""
    tool = _make_tool(tmp_path)

    with pytest.raises(ValueError, match="Known tools: \\['run_sql'\\]"):
        OumiVerlTool(config=tool.config, tool_schema=_tool_schema("run_sqll"))


def test_failed_tool_call_becomes_an_observation(tmp_path):
    """A bad tool call must not propagate and kill the rollout."""
    tool = _make_tool(tmp_path)
    ad = _agent_data("failed-tool-call-becomes-an-observation")

    iid, _ = asyncio.run(tool.create())
    resp, reward, _metrics = asyncio.run(
        tool.execute(iid, {"query": "NOT VALID SQL"}, agent_data=ad)
    )
    assert reward == 0.0
    assert resp.text is not None
    assert resp.text.startswith("Tool error:")
