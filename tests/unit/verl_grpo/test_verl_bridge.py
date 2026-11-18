import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf

from oumi.core.registry import RegistryType, register
from oumi.exceptions import OumiConfigError
from oumi.verl_grpo.bridge import BridgeServer, check_bridge_config, uses_bridge
from oumi_verl_ext.bridge_client import BRIDGE_REWARD_PATH

_SLOW_REWARD_S = 0.2


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
    with ThreadPoolExecutor(max_workers=1) as pool:
        with BridgeServer() as bridge:
            future = pool.submit(
                _post,
                bridge.url + "/reward",
                _payload(reward_functions=["t_bridge_slow"]),
            )
            time.sleep(0.05)
            start = time.monotonic()
        assert time.monotonic() - start < _SLOW_REWARD_S
        future.result(timeout=10)


def test_unknown_paths_are_a_404():
    with BridgeServer() as bridge:
        status, body = _post(bridge.url + "/nope", {})
    assert status == 404
    assert "error" in body


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
