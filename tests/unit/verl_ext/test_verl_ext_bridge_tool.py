"""Tests the bridge tool against stand-ins for verl's tool classes."""

import asyncio
import gc
import importlib.util
import json
import re
import sys
import threading
import time
import types
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import oumi_verl_ext
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.verl_grpo.bridge import BridgeServer
from oumi_verl_ext.bridge_client import BRIDGE_URL_ENV, BridgeError

ENV_CONFIG = "/configs/env.yaml"
TOOLS_KWARGS = {"run_sql": {"create_kwargs": {"db_path": "/data/db.sqlite"}}}
RESPONSES = {
    "/tool/check": {},
    "/tool/execute": {"text": "rows", "reward": 0.0, "metrics": {}},
    "/tool/close": {"closed": True},
}
SEEDED = {
    "schema_sql": "CREATE TABLE t(x INTEGER);",
    "seed_sql": "INSERT INTO t VALUES (1),(2);",
}
SCHEMA = SimpleNamespace(function=SimpleNamespace(name="run_sql"))


@dataclass
class _ToolResponse:
    text: str | None = None


class _BaseTool:
    """The part of verl's BaseTool the bridge tool relies on."""

    def __init__(self, config, tool_schema):
        self.config = config
        self.tool_schema = tool_schema
        self.name = tool_schema.function.name


class _AgentData:
    def __init__(self, request_id, tools_kwargs):
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def bridge_tool(monkeypatch):
    """Loads a fresh bridge tool module against stand-ins for verl's tool classes."""
    for module in (
        _module("verl"),
        _module("verl.tools"),
        _module("verl.tools.base_tool", BaseTool=_BaseTool),
        _module(
            "verl.tools.schemas",
            OpenAIFunctionToolSchema=object,
            ToolResponse=_ToolResponse,
        ),
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(oumi_verl_ext.__file__).with_name("bridge_tool.py")
    spec = importlib.util.spec_from_file_location("bridge_tool_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubBridge(ThreadingHTTPServer):
    """Answers each tool op with RESPONSES and records every request."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.requests: list[tuple[str, dict]] = []
        self.status = 200
        self.close_delay_s = 0.0

    def payloads(self, path: str) -> list[dict]:
        # Filtered by path: an earlier test's rollout can still be closing.
        return [
            payload for request_path, payload in self.requests if request_path == path
        ]


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        server = self.server
        assert isinstance(server, _StubBridge)
        length = int(self.headers["Content-Length"])
        server.requests.append((self.path, json.loads(self.rfile.read(length))))
        if self.path == "/tool/close":
            time.sleep(server.close_delay_s)
        body = RESPONSES[self.path] if server.status == 200 else {"error": "refused"}
        data = json.dumps(body).encode()
        self.send_response(server.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def bridge(monkeypatch):
    server = _StubBridge()
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    monkeypatch.setenv(BRIDGE_URL_ENV, f"http://127.0.0.1:{server.server_address[1]}")
    yield server
    server.shutdown()
    server.server_close()


def _tool(bridge_tool):
    return bridge_tool.OumiBridgeTool(
        config={"type": "native", "oumi_env_config": ENV_CONFIG}, tool_schema=SCHEMA
    )


def _execute(tool, agent_data):
    return asyncio.run(tool.execute("i", {"query": "SELECT 1"}, agent_data=agent_data))


def _eventually(condition) -> bool:
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_construction_checks_the_tool_with_the_bridge(bridge, bridge_tool):
    _tool(bridge_tool)
    check = {"env_config": ENV_CONFIG, "tool_name": "run_sql"}
    assert bridge.payloads("/tool/check") == [check]
    bridge.status = 400
    with pytest.raises(BridgeError, match="refused"):
        _tool(bridge_tool)


def test_a_missing_oumi_env_config_is_refused(bridge, bridge_tool):
    with pytest.raises(ValueError, match="'run_sql' needs 'oumi_env_config'"):
        bridge_tool.OumiBridgeTool(config={"type": "native"}, tool_schema=SCHEMA)
    assert bridge.payloads("/tool/check") == []


def test_create_mints_an_instance_id_locally(bridge, bridge_tool):
    tool = _tool(bridge_tool)
    create_kwargs = TOOLS_KWARGS["run_sql"]["create_kwargs"]
    minted, _ = asyncio.run(tool.create(create_kwargs=create_kwargs))
    given, _ = asyncio.run(tool.create("given", create_kwargs=create_kwargs))
    assert re.fullmatch(r"[0-9a-f]{32}", minted)
    assert given == "given"
    assert [path for path, _ in bridge.requests if path != "/tool/close"] == [
        "/tool/check"
    ]


def test_execute_posts_request_id_tools_kwargs_and_parameters(bridge, bridge_tool):
    agent_data = _AgentData("r-execute", TOOLS_KWARGS)
    response, reward, metrics = _execute(_tool(bridge_tool), agent_data)
    assert (response.text, reward, metrics) == ("rows", 0.0, {})
    assert bridge.payloads("/tool/execute") == [
        {
            "env_config": ENV_CONFIG,
            "tool_name": "run_sql",
            "instance_id": "i",
            "request_id": "r-execute",
            "tools_kwargs": TOOLS_KWARGS,
            "parameters": {"query": "SELECT 1"},
        }
    ]


def test_dropping_agent_data_posts_close_once(bridge, bridge_tool):
    agent_data = _AgentData("r-drop", TOOLS_KWARGS)
    for tool in (_tool(bridge_tool), _tool(bridge_tool)):
        for _ in range(2):
            _execute(tool, agent_data)
    assert {"request_id": "r-drop"} not in bridge.payloads("/tool/close")

    del agent_data
    gc.collect()

    assert _eventually(
        lambda: {"request_id": "r-drop"} in bridge.payloads("/tool/close")
    )
    time.sleep(0.2)
    assert bridge.payloads("/tool/close").count({"request_id": "r-drop"}) == 1


def test_dropping_agent_data_does_not_wait_for_the_close(bridge, bridge_tool):
    # verl frees agent_data on its event loop, which must not block on the bridge.
    bridge.close_delay_s = 1.0
    agent_data = _AgentData("r-slow-close", {})
    _execute(_tool(bridge_tool), agent_data)

    start = time.monotonic()
    del agent_data
    gc.collect()

    assert time.monotonic() - start < 0.5
    assert _eventually(
        lambda: {"request_id": "r-slow-close"} in bridge.payloads("/tool/close")
    )


@pytest.mark.parametrize("failure", ["down", "refused", "malformed"])
def test_execute_raises_when_the_bridge_is_down(
    bridge, bridge_tool, monkeypatch, caplog, failure
):
    tool = _tool(bridge_tool)
    if failure == "down":
        bridge.shutdown()
        bridge.server_close()
    elif failure == "refused":
        bridge.status = 500
    else:
        monkeypatch.setitem(RESPONSES, "/tool/execute", {"reward": 0.0})
    agent_data = _AgentData(f"r-{failure}", {})
    with pytest.raises(BridgeError):
        _execute(tool, agent_data)

    del agent_data
    gc.collect()

    # The close still goes out; against a dead bridge it is only logged.
    if failure == "down":
        assert _eventually(lambda: "r-down" in caplog.text)


def test_the_closer_keeps_running_after_an_unexpected_error(
    bridge, bridge_tool, monkeypatch, caplog
):
    post_json = bridge_tool.post_json

    def post_json_failing_one_close(path, payload, **kwargs):
        if payload == {"request_id": "r-unexpected"}:
            raise RuntimeError("unexpected close failure")
        return post_json(path, payload, **kwargs)

    monkeypatch.setattr(bridge_tool, "post_json", post_json_failing_one_close)
    tool = _tool(bridge_tool)
    for request_id in ("r-unexpected", "r-next"):
        agent_data = _AgentData(request_id, {})
        _execute(tool, agent_data)
        del agent_data
        gc.collect()

    assert _eventually(
        lambda: {"request_id": "r-next"} in bridge.payloads("/tool/close")
    )
    assert "RuntimeError: unexpected close failure" in caplog.text


def test_calc_reward_and_release_stay_local(bridge, bridge_tool):
    tool = _tool(bridge_tool)
    assert asyncio.run(tool.calc_reward("i")) == 0.0
    assert asyncio.run(tool.release("i")) is None
    assert [path for path, _ in bridge.requests if path != "/tool/close"] == [
        "/tool/check"
    ]


def _env_config(tmp_path) -> str:
    environment = EnvironmentParams(
        id="db",
        env_type="database",
        env_kwargs={"schema_sql": "CREATE TABLE t(x INTEGER);"},
        tools=[
            {
                "id": "run_sql",
                "name": "run_sql",
                "description": "run sql",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "executor": "oumi.environments.examples.nl2sql.run_sql",
                "read_only": True,
            }
        ],
    )
    path = tmp_path / "env.yaml"
    EnvironmentConfig(environments=[environment]).to_yaml(path)
    return str(path)


def _live_routers(bridge) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(bridge.url + "/health", timeout=10) as response:
        return json.loads(response.read())["live_routers"]


async def _call_tool(tool, agent_data):
    # The order verl's tool agent loop uses for every call.
    instance_id, _ = await tool.create(create_kwargs=SEEDED)
    try:
        return await tool.execute(
            instance_id, {"query": "SELECT x FROM t ORDER BY x"}, agent_data=agent_data
        )
    finally:
        await tool.release(instance_id)


def test_bridge_tool_runs_sql_through_the_real_bridge(
    bridge_tool, tmp_path, monkeypatch
):
    with BridgeServer() as bridge:
        monkeypatch.setenv(BRIDGE_URL_ENV, bridge.url)
        tool = bridge_tool.OumiBridgeTool(
            config={"type": "native", "oumi_env_config": _env_config(tmp_path)},
            tool_schema=SCHEMA,
        )
        agent_data = _AgentData("e2e", {"run_sql": {"create_kwargs": SEEDED}})

        response, reward, metrics = asyncio.run(_call_tool(tool, agent_data))

        assert json.loads(response.text) == {"columns": ["x"], "rows": [[1], [2]]}
        assert (reward, metrics) == (0.0, {})
        assert _live_routers(bridge) == 1

        del agent_data
        gc.collect()

        deadline = time.monotonic() + 10
        while _live_routers(bridge) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _live_routers(bridge) == 0
