import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from omegaconf import ListConfig

import oumi_verl_ext.bridge_reward as bridge_reward
from oumi.core.registry import RegistryType, register
from oumi.verl_grpo.bridge import BridgeServer
from oumi_verl_ext.bridge_client import BRIDGE_URL_ENV, BridgeError, to_json_value
from oumi_verl_ext.bridge_reward import oumi_registry_reward


@register("t_shim_one", RegistryType.REWARD_FUNCTION)
def _one(data_source, solution_str, ground_truth, extra_info):
    return 1.0


@register("t_shim_raises", RegistryType.REWARD_FUNCTION)
def _raises(data_source, solution_str, ground_truth, extra_info):
    raise RuntimeError("boom")


@register("t_shim_echo_ground_truth", RegistryType.REWARD_FUNCTION)
def _echo(data_source, solution_str, ground_truth, extra_info):
    return {"score": 1.0, "ground_truth": ground_truth}


@pytest.fixture
def stub_bridge(monkeypatch):
    """A stdlib server that answers /reward with a canned body."""
    state = {"body": {}, "status": 200, "delay_s": 0.0, "raw": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            time.sleep(state["delay_s"])
            data = state["raw"]
            if data is None:
                data = json.dumps(state["body"]).encode()
            self.send_response(state["status"])
            length = state.get("content_length", len(data))
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(BRIDGE_URL_ENV, f"http://127.0.0.1:{server.server_address[1]}")
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_registry_reward_scores_through_the_bridge(monkeypatch):
    with BridgeServer() as bridge:
        monkeypatch.setenv(BRIDGE_URL_ENV, bridge.url)
        out = oumi_registry_reward(
            "ds", "sol", "gt", {"index": 1}, reward_functions=["t_shim_one"]
        )
    assert out == {"score": 1.0, "t_shim_one": 1.0}


def test_registry_reward_encodes_numpy_and_omegaconf_values(monkeypatch):
    with BridgeServer() as bridge:
        monkeypatch.setenv(BRIDGE_URL_ENV, bridge.url)
        out = oumi_registry_reward(
            "countdown",
            "s",
            {"numbers": np.array([79, 8]), "target": np.int64(87)},
            {"x": np.float32(0.5)},
            reward_functions=ListConfig(["t_shim_echo_ground_truth"]),
        )
    assert out["t_shim_echo_ground_truth/ground_truth"] == {
        "numbers": [79, 8],
        "target": 87,
    }


def test_to_json_value_leaves_strings_and_scalars_alone():
    assert to_json_value("abc") == "abc"
    assert to_json_value(None) is None
    assert to_json_value((1, np.float32(0.5), {"a": np.bool_(True)})) == [
        1,
        0.5,
        {"a": True},
    ]


def test_to_json_value_encodes_values_nested_in_object_arrays():
    array = np.empty(2, dtype=object)
    array[0] = np.array([1, 2])
    array[1] = {"n": np.int64(3)}
    assert json.dumps(to_json_value(array)) == '[[1, 2], {"n": 3}]'


def test_registry_reward_wraps_an_unencodable_payload(stub_bridge):
    with pytest.raises(BridgeError, match="JSON-encodable"):
        oumi_registry_reward("ds", "s", {1, 2}, reward_functions=["t_shim_one"])


def test_registry_reward_wraps_a_truncated_response(stub_bridge):
    stub_bridge["raw"] = b'{"scores": {"score": 1.0}}'
    stub_bridge["content_length"] = 100
    with pytest.raises(BridgeError):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


def test_registry_reward_without_a_bridge_url_raises(monkeypatch):
    monkeypatch.delenv(BRIDGE_URL_ENV, raising=False)
    with pytest.raises(BridgeError, match=BRIDGE_URL_ENV):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


def test_registry_reward_raises_when_the_reward_function_raises(monkeypatch):
    with BridgeServer() as bridge:
        monkeypatch.setenv(BRIDGE_URL_ENV, bridge.url)
        with pytest.raises(BridgeError, match="RuntimeError"):
            oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_raises"])


def test_registry_reward_raises_when_the_bridge_is_down(monkeypatch):
    with BridgeServer() as bridge:
        url = bridge.url
    monkeypatch.setenv(BRIDGE_URL_ENV, url)
    with pytest.raises(BridgeError):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"scores": {}},
        {"scores": {"score": "1"}},
        {"scores": {"score": True}},
        {"scores": {"score": float("inf")}},
    ],
)
def test_registry_reward_rejects_a_bad_score(stub_bridge, body):
    stub_bridge["body"] = body
    with pytest.raises(BridgeError):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


def test_registry_reward_rejects_a_non_json_body(stub_bridge):
    stub_bridge["raw"] = b"<html>oops</html>"
    with pytest.raises(BridgeError, match="JSON"):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


def test_registry_reward_rejects_a_non_200_status(stub_bridge):
    stub_bridge["status"] = 400
    stub_bridge["body"] = {"error": "bad payload"}
    with pytest.raises(BridgeError, match="bad payload"):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


def test_registry_reward_rejects_a_2xx_that_is_not_200(stub_bridge):
    stub_bridge["status"] = 202
    stub_bridge["body"] = {"scores": {"score": 1.0}}
    with pytest.raises(BridgeError, match="202"):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])


def test_registry_reward_passes_a_nan_function_score_through(stub_bridge):
    stub_bridge["body"] = {"scores": {"score": 1.0, "f": float("nan")}}
    out = oumi_registry_reward("ds", "s", "gt", reward_functions=["f"])
    assert out["score"] == 1.0
    assert out["f"] != out["f"]


def test_registry_reward_times_out(stub_bridge, monkeypatch):
    monkeypatch.setattr(bridge_reward, "REWARD_TIMEOUT_S", 0.2)
    stub_bridge["delay_s"] = 1.0
    stub_bridge["body"] = {"scores": {"score": 1.0}}
    with pytest.raises(BridgeError):
        oumi_registry_reward("ds", "s", "gt", reward_functions=["t_shim_one"])
