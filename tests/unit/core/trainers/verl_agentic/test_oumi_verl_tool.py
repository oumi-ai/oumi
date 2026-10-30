import asyncio
import json
import sqlite3
from types import SimpleNamespace
from typing import Any

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


def _make_tool(tmp_path, execution_mode="sync") -> OumiVerlTool:
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
        config={"oumi_env_config": str(cfg_path), "execution_mode": execution_mode},
        tool_schema=_tool_schema(),
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


@pytest.mark.parametrize("execution_mode", ["sync", "thread"])
def test_execute_routes_into_shared_env(tmp_path, execution_mode):
    tool = _make_tool(tmp_path, execution_mode)
    ad = _agent_data(f"execute-routes-into-shared-env-{execution_mode}")

    iid, _ = asyncio.run(tool.create())
    resp, reward, _metrics = asyncio.run(
        tool.execute(iid, {"query": "SELECT count(*) FROM t"}, agent_data=ad)
    )
    assert reward == 0.0
    assert resp.text is not None and "2" in resp.text


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


@pytest.mark.parametrize("execution_mode", ["sync", "thread"])
def test_failed_tool_call_becomes_an_observation(tmp_path, execution_mode):
    """A bad tool call must not propagate and kill the rollout."""
    tool = _make_tool(tmp_path, execution_mode)
    ad = _agent_data(f"failed-tool-call-becomes-an-observation-{execution_mode}")

    iid, _ = asyncio.run(tool.create())
    resp, reward, _metrics = asyncio.run(
        tool.execute(iid, {"query": "NOT VALID SQL"}, agent_data=ad)
    )
    assert reward == 0.0
    assert resp.text is not None and resp.text.startswith("Tool error:")


def test_invalid_execution_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="execution_mode"):
        _make_tool(tmp_path, "invalid")


def test_native_verl_agent_loop_drives_isolated_threaded_tools(tmp_path):
    from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    tool = _make_tool(tmp_path, "thread")
    # Only exercise VERL's real create/execute/release dispatch, without a model.
    loop: Any = object.__new__(ToolAgentLoop)
    loop.tools = {"run_sql": tool}
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "left"

    async def run():
        owners = [
            AgentData(
                messages=[],
                image_data=[],
                video_data=[],
                metrics={},
                request_id=f"native-verl-{i}",
                tools_kwargs=_agent_data("unused").tools_kwargs,
            )
            for i in range(12)
        ]
        call = FunctionCall(
            name="run_sql", arguments=json.dumps({"query": "SELECT sum(x) FROM t"})
        )
        results = await asyncio.gather(
            *(loop._call_tool(call, o.tools_kwargs, o) for o in owners)
        )
        for response, reward, _ in results:
            assert response.text is not None
            assert json.loads(response.text)["rows"] == [[3]]
            assert reward == 0

    asyncio.run(run())
