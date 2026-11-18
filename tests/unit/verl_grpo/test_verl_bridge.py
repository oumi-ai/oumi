import http.client
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import RegistryType, register
from oumi.core.types.tool_call import ToolResult
from oumi.exceptions import OumiConfigError
from oumi.verl_grpo.bridge import BridgeServer, check_bridge_config, uses_bridge
from oumi_verl_ext.bridge_client import BRIDGE_REWARD_PATH

_SLOW_REWARD_S = 0.2
_SLOW_REWARD_STARTED = threading.Event()


@register("t_bridge_one", RegistryType.REWARD_FUNCTION)
def _one(data_source, solution_str, ground_truth, extra_info):
    return 1.0


@register("t_bridge_zero", RegistryType.REWARD_FUNCTION)
def _zero(data_source, solution_str, ground_truth, extra_info):
    return 0.0


@register("t_bridge_raises", RegistryType.REWARD_FUNCTION)
def _raises(data_source, solution_str, ground_truth, extra_info):
    raise RuntimeError("boom")


@register("t_bridge_slow", RegistryType.REWARD_FUNCTION)
def _slow(data_source, solution_str, ground_truth, extra_info):
    _SLOW_REWARD_STARTED.set()
    time.sleep(_SLOW_REWARD_S)
    return 1.0


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url):
    with _OPENER.open(url, timeout=10) as response:
        return json.loads(response.read())


def _post(url, payload, *, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    try:
        with _OPENER.open(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _payload(**overrides):
    payload = {
        "data_source": "ds",
        "solution_str": "sol",
        "ground_truth": "gt",
        "extra_info": {"index": 1},
        "reward_functions": ["t_bridge_one"],
        "reward_weights": None,
        "reward_function_kwargs": {},
    }
    payload.update(overrides)
    return payload


def test_bridge_serves_health_on_an_ephemeral_localhost_port():
    with BridgeServer() as bridge:
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+", bridge.url)
        assert not bridge.url.endswith(":0")
        assert _get(bridge.url + "/health") == {"status": "ok", "live_routers": 0}


def test_bridge_refuses_connections_after_exit():
    with BridgeServer() as bridge:
        url = bridge.url
    with pytest.raises(urllib.error.URLError):
        _get(url + "/health")


def test_reward_endpoint_returns_the_weighted_sum():
    with BridgeServer() as bridge:
        status, body = _post(
            bridge.url + "/reward",
            _payload(
                reward_functions=["t_bridge_one", "t_bridge_zero"],
                reward_weights=[2.0, 0.5],
            ),
        )
    assert status == 200
    assert body == {"scores": {"score": 2.0, "t_bridge_one": 1.0, "t_bridge_zero": 0.0}}


def test_a_raising_reward_function_is_a_500_naming_the_exception():
    with BridgeServer() as bridge:
        status, body = _post(
            bridge.url + "/reward", _payload(reward_functions=["t_bridge_raises"])
        )
    assert status == 500
    assert body == {"error": "RuntimeError: boom"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raw": b"not json"},
        {"raw": b"[1, 2]"},
        {"payload": {"solution_str": "s"}},
        {"payload": _payload(reward_functions="t_bridge_one")},
        {"payload": _payload(reward_functions=[])},
        {"payload": _payload(solution_str=3)},
        {"payload": _payload(reward_weights=[1.0, 2.0])},
        {"payload": _payload(reward_weights=["1"])},
        {"payload": _payload(reward_weights=[True])},
        {"payload": _payload(reward_function_kwargs={"t_bridge_one": 3})},
    ],
)
def test_a_malformed_reward_payload_is_a_400(kwargs):
    with BridgeServer() as bridge:
        status, body = _post(
            bridge.url + "/reward", kwargs.get("payload"), raw=kwargs.get("raw")
        )
    assert status == 400
    assert body["error"]


def test_exit_returns_while_a_reward_is_still_running():
    _SLOW_REWARD_STARTED.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with BridgeServer() as bridge:
            future = pool.submit(
                _post,
                bridge.url + "/reward",
                _payload(reward_functions=["t_bridge_slow"]),
            )
            assert _SLOW_REWARD_STARTED.wait(timeout=10)
            start = time.monotonic()
        assert time.monotonic() - start < _SLOW_REWARD_S
        future.result(timeout=10)


@pytest.mark.parametrize("path", ["/nope", "/tool/nope"])
def test_unknown_paths_are_a_404(path):
    with BridgeServer() as bridge:
        status, body = _post(bridge.url + path, {})
    assert status == 404
    assert "error" in body


@pytest.mark.parametrize("content_length", [None, "abc", "-1"])
def test_a_bad_content_length_is_a_400(content_length):
    with BridgeServer() as bridge:
        connection = http.client.HTTPConnection(
            bridge.url.removeprefix("http://"), timeout=10
        )
        connection.putrequest("POST", "/reward")
        if content_length is not None:
            connection.putheader("Content-Length", content_length)
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        assert "Content-Length" in json.loads(response.read())["error"]
        connection.close()


def test_bridge_serves_concurrent_rewards():
    with BridgeServer() as bridge:
        payload = _payload(reward_functions=["t_bridge_slow"])
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(
                pool.map(lambda _: _post(bridge.url + "/reward", payload), range(16))
            )
        elapsed = time.monotonic() - start
    assert all(status == 200 for status, _ in results)
    assert elapsed < 1.5


def _verl_config(
    path: str | None = BRIDGE_REWARD_PATH,
    names=("t_bridge_one",),
    weights=None,
    nnodes=1,
    reward_kwargs=None,
):
    if reward_kwargs is None:
        reward_kwargs = {"reward_functions": list(names), "reward_weights": weights}
    return OmegaConf.create(
        {
            "trainer": {"nnodes": nnodes},
            "reward": {
                "custom_reward_function": {
                    "path": path,
                    "reward_kwargs": reward_kwargs,
                }
            },
        }
    )


def test_uses_bridge_only_for_the_bridge_reward_path():
    assert uses_bridge(_verl_config())
    assert not uses_bridge(_verl_config(path="/some/file.py"))
    assert not uses_bridge(_verl_config(path=None))
    assert not uses_bridge(OmegaConf.create({"reward": {}}))


def test_check_bridge_config_accepts_a_valid_config():
    check_bridge_config(_verl_config(weights=[2.0]))


def test_check_bridge_config_skips_non_bridge_rewards():
    check_bridge_config(_verl_config(path="/some/file.py", names=("missing",)))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"names": ("t_bridge_missing",)}, "not registered"),
        ({"weights": [1.0, 2.0]}, "reward_weights"),
        ({"names": ()}, "reward_functions"),
        ({"nnodes": 2}, "single node"),
        ({"weights": 2.0}, "list of numbers"),
        ({"weights": ["a"]}, "list of numbers"),
        ({"weights": [True]}, "list of numbers"),
        ({"reward_kwargs": ["t_bridge_one"]}, "mapping"),
        ({"reward_kwargs": 3}, "mapping"),
    ],
)
def test_check_bridge_config_refuses(overrides, match):
    with pytest.raises(OumiConfigError, match=match):
        check_bridge_config(_verl_config(**overrides))


def run_sql(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    cursor = context.execute(arguments["query"])
    return ToolResult(
        output={
            "rows": [list(row) for row in cursor.fetchall()],
            "args": sorted(arguments),
        }
    )


def _env_config(tmp_path) -> str:
    tool = {
        "description": "run sql",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        "executor": f"{__name__}.run_sql",
        "read_only": True,
    }
    environment = EnvironmentParams(
        id="db",
        env_type="database",
        env_kwargs={
            "schema_sql": "CREATE TABLE t(x INTEGER);",
            "seed_sql": "INSERT INTO t VALUES (1),(2);",
        },
        tools=[{"id": name, "name": name, **tool} for name in ("run_sql", "sql_2")],
    )
    path = tmp_path / "env.yaml"
    EnvironmentConfig(environments=[environment]).to_yaml(path)
    return str(path)


def _execute(url, env_config, request_id, query="SELECT x FROM t", **overrides):
    payload = {
        "env_config": env_config,
        "tool_name": "run_sql",
        "instance_id": "i",
        "request_id": request_id,
        "tools_kwargs": {},
        "parameters": {"query": query, "limit": None},
    }
    return _post(url + "/tool/execute", payload | overrides)


def _live_routers(bridge) -> int:
    return _get(bridge.url + "/health")["live_routers"]


def test_execute_routes_run_sql_through_http(tmp_path):
    with BridgeServer() as bridge:
        status, body = _execute(bridge.url, _env_config(tmp_path), "r")
        assert _live_routers(bridge) == 1
    assert status == 200
    rows = json.dumps({"rows": [[1], [2]], "args": ["query"]})
    assert body == {"text": rows, "reward": 0.0, "metrics": {}}


def test_a_failing_tool_call_becomes_tool_error_text(tmp_path):
    with BridgeServer() as bridge:
        status, body = _execute(bridge.url, _env_config(tmp_path), "r", "NOT SQL")
    assert status == 200
    assert body["text"].startswith("Tool error:")


def test_conflicting_create_kwargs_are_a_400(tmp_path):
    tools_kwargs = {
        name: {"create_kwargs": {"schema_sql": f"CREATE TABLE {name}(x);"}}
        for name in ("run_sql", "sql_2")
    }
    with BridgeServer() as bridge:
        status, body = _execute(
            bridge.url, _env_config(tmp_path), "r", tools_kwargs=tools_kwargs
        )
    assert status == 400
    assert "conflicting create_kwargs" in body["error"]


def test_close_drops_live_routers_to_zero(tmp_path):
    with BridgeServer() as bridge:
        _execute(bridge.url, _env_config(tmp_path), "r")
        close = bridge.url + "/tool/close"
        assert _post(close, {"request_id": "r"}) == (200, {"closed": True})
        assert _live_routers(bridge) == 0
        assert _post(close, {"request_id": "r"}) == (200, {"closed": False})


def test_bridge_exit_closes_every_router(tmp_path, caplog):
    config = _env_config(tmp_path)
    with BridgeServer() as bridge:
        for request_id in ("a", "b"):
            _execute(bridge.url, config, request_id)
    assert "closed 2 live tool routers" in caplog.text


def test_concurrent_executes_on_eight_rollouts(tmp_path):
    config = _env_config(tmp_path)
    with BridgeServer() as bridge:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda i: _execute(bridge.url, config, f"r{i}"), range(8))
            )
        assert _live_routers(bridge) == 8
    assert [status for status, _ in results] == [200] * 8
    assert all('"rows": [[1], [2]]' in body["text"] for _, body in results)


def test_tool_lifecycle_ops_follow_the_contract(tmp_path):
    with BridgeServer() as bridge:

        def post(op, payload):
            return _post(f"{bridge.url}/tool/{op}", payload)

        check = {"env_config": _env_config(tmp_path), "tool_name": "run_sql"}
        assert post("check", check) == (200, {})
        create = {"tool_name": "run_sql", "instance_id": None, "create_kwargs": {}}
        status, body = post("create", create)
        assert status == 200
        assert re.fullmatch(r"[0-9a-f]{32}", body["instance_id"])
        given = create | {"instance_id": "given"}
        assert post("create", given) == (200, {"instance_id": "given"})
        assert post("calc_reward", {"instance_id": "given"}) == (200, {"reward": 0.0})
        assert post("release", {"instance_id": "given"}) == (200, {})


@pytest.mark.parametrize(
    "op,payload,match",
    [
        ("check", {"tool_name": "nope"}, "Known tools"),
        ("check", {"env_config": "missing.yaml", "tool_name": "run_sql"}, "not found"),
        ("check", {"tool_name": 3}, "tool_name"),
        ("execute", {"request_id": None}, "request_id"),
        ("execute", {"parameters": [1]}, "parameters"),
        ("execute", {"tools_kwargs": {"run_sql": 3}}, "tools_kwargs"),
        ("execute", {"tools_kwargs": {"nope": {"create_kwargs": {"a": 1}}}}, "nope"),
        ("create", {"instance_id": 3}, "instance_id"),
        ("release", {}, "instance_id"),
        ("close", {"request_id": None}, "request_id"),
    ],
)
def test_a_malformed_tool_request_is_a_400(tmp_path, op, payload, match):
    base = {
        "env_config": _env_config(tmp_path),
        "tool_name": "run_sql",
        "request_id": "r",
        "parameters": {"query": "SELECT 1"},
    }
    with BridgeServer() as bridge:
        status, body = _post(f"{bridge.url}/tool/{op}", base | payload)
    assert status == 400
    assert match in body["error"]


def _tool_verl_config(tmp_path, class_name, full_determinism=False):
    tool_config = tmp_path / "tools.yaml"
    tool_config.write_text(
        f"tools:\n- class_name: {class_name}\n  config: {{type: native}}\n"
    )
    rollout = {
        "full_determinism": full_determinism,
        "multi_turn": {"tool_config_path": str(tool_config)},
    }
    return OmegaConf.create(
        {"trainer": {"nnodes": 1}, "actor_rollout_ref": {"rollout": rollout}}
    )


def test_uses_bridge_for_a_tool_config_naming_the_bridge_tool(tmp_path):
    bridge_tool = "oumi_verl_ext.bridge_tool.OumiBridgeTool"
    assert uses_bridge(_tool_verl_config(tmp_path, bridge_tool))
    check_bridge_config(_tool_verl_config(tmp_path, bridge_tool))
    assert not uses_bridge(_tool_verl_config(tmp_path, "verl.tools.my_tool.MyTool"))


def test_full_determinism_with_the_bridge_tool_is_refused(tmp_path):
    verl_config = _tool_verl_config(
        tmp_path, "oumi_verl_ext.bridge_tool.OumiBridgeTool", full_determinism=True
    )
    with pytest.raises(OumiConfigError, match="full_determinism"):
        check_bridge_config(verl_config)
