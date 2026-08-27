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

import pytest
from pydantic import JsonValue

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.registry import REGISTRY, RegistryType
from oumi.environments.endpoint_environment import (
    EndpointCallError,
    EndpointEnvironment,
    EndpointEnvironmentKwargs,
    EndpointTransport,
)
from oumi.environments.utils import parse_env_kwargs

_URL = "https://tools.example.com/call"


def _tool() -> ToolParams:
    return ToolParams(
        id="place_order",
        name="Place order",
        description="Places an order.",
        parameters={
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
    )


class _RecordingTransport:
    """Captures each request and returns a canned response."""

    def __init__(
        self, response: JsonValue = None, error: Exception | None = None
    ) -> None:
        self.response: JsonValue = {"status": "ok"} if response is None else response
        self.error = error
        self.requests: list[dict] = []

    def __call__(
        self, *, url: str, payload: dict[str, JsonValue], timeout_seconds: float
    ) -> JsonValue:
        self.requests.append(
            {"url": url, "payload": payload, "timeout_seconds": timeout_seconds}
        )
        if self.error is not None:
            raise self.error
        return self.response


def _environment(
    transport: EndpointTransport, timeout_seconds: float = 5.0
) -> EndpointEnvironment:
    kwargs = EndpointEnvironmentKwargs(
        endpoint_url=_URL, timeout_seconds=timeout_seconds
    )
    kwargs.finalize_and_validate()
    return EndpointEnvironment(
        EnvironmentParams(id="env", env_type="endpoint", tools=[_tool()]),
        kwargs,
        transport,
    )


def test_call_sends_the_tool_call_and_returns_the_response():
    transport = _RecordingTransport()
    env = _environment(transport)

    result = env.call(
        "place_order",
        {"item": "X"},
        call_id="row42:3:0:place_order",
        session_id="row42",
    )

    assert result.output == {"status": "ok"}
    assert transport.requests == [
        {
            "url": _URL,
            "payload": {
                "name": "place_order",
                "arguments": {"item": "X"},
                "call_id": "row42:3:0:place_order",
                "session_id": "row42",
            },
            "timeout_seconds": 5.0,
        }
    ]


def test_step_identifies_calls_when_the_caller_does_not():
    transport = _RecordingTransport()
    env = _environment(transport)

    env.step([("place_order", {"item": "X"}), ("place_order", {"item": "Y"})])

    payloads = [request["payload"] for request in transport.requests]
    # Distinct calls in one conversation: same session, different call ids.
    assert payloads[0]["call_id"] != payloads[1]["call_id"]
    assert payloads[0]["session_id"] == payloads[1]["session_id"]


def test_call_rejects_arguments_that_do_not_match_the_tool_schema():
    transport = _RecordingTransport()
    env = _environment(transport)

    with pytest.raises(ToolArgumentError):
        env.call("place_order", {"item": 7}, call_id="c1")
    assert transport.requests == []


def test_call_rejects_an_unknown_tool():
    env = _environment(_RecordingTransport())

    with pytest.raises(ToolLookupError):
        env.call("cancel_order", {"item": "X"}, call_id="c1")


def test_call_reports_a_response_that_breaks_the_output_schema():
    env = _environment(_RecordingTransport(response={"state": "ok"}))

    with pytest.raises(EndpointCallError, match="output schema"):
        env.call("place_order", {"item": "X"}, call_id="c1")


def test_call_reports_a_transport_failure_as_a_tool_error():
    env = _environment(_RecordingTransport(error=TimeoutError("timed out")))

    with pytest.raises(EndpointCallError, match="timed out"):
        env.call("place_order", {"item": "X"}, call_id="c1")


def test_call_accepts_any_response_when_the_tool_declares_no_output_schema():
    kwargs = EndpointEnvironmentKwargs(endpoint_url=_URL)
    kwargs.finalize_and_validate()
    tool = _tool()
    tool.output_schema = None
    env = EndpointEnvironment(
        EnvironmentParams(id="env", env_type="endpoint", tools=[tool]),
        kwargs,
        _RecordingTransport(response=["anything"]),
    )

    assert env.call("place_order", {"item": "X"}, call_id="c1").output == ["anything"]


@pytest.mark.parametrize(
    "kwargs",
    [
        EndpointEnvironmentKwargs(endpoint_url=""),
        EndpointEnvironmentKwargs(endpoint_url=_URL, timeout_seconds=0),
        EndpointEnvironmentKwargs(endpoint_url=_URL, timeout_seconds=-1),
    ],
)
def test_kwargs_reject_an_unusable_endpoint_configuration(kwargs):
    with pytest.raises(ValueError):
        kwargs.finalize_and_validate()


def test_environment_is_registered_under_endpoint():
    assert REGISTRY.get("endpoint", RegistryType.ENVIRONMENT) is EndpointEnvironment


def test_kwargs_parse_from_a_plain_json_config():
    params = EnvironmentParams(
        id="env",
        env_type="endpoint",
        tools=[],
        env_kwargs={"endpoint_url": _URL, "timeout_seconds": 5},
    )

    kwargs = parse_env_kwargs(EndpointEnvironmentKwargs, params, env_label="E")

    assert kwargs.endpoint_url == _URL
    assert kwargs.timeout_seconds == 5


class _ClosingTransport(_RecordingTransport):
    """A transport owning resources it must release."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_close_releases_a_transport_that_owns_resources():
    transport = _ClosingTransport()

    _environment(transport).close()

    assert transport.closed


def test_close_is_a_no_op_for_a_transport_that_owns_nothing():
    _environment(_RecordingTransport()).close()


def test_a_tool_error_from_the_transport_passes_through_unchanged():
    def refusing_transport(
        *, url: str, payload: dict[str, JsonValue], timeout_seconds: float
    ) -> JsonValue:
        raise ToolError("out of stock")

    with pytest.raises(ToolError, match="^out of stock$") as excinfo:
        _environment(refusing_transport).call(
            "place_order", {"item": "X"}, call_id="c1"
        )

    # A refusal is the tool's answer, not an endpoint failure.
    assert not isinstance(excinfo.value, EndpointCallError)


# The environment fixes what a call means, never how it travels. These two
# transports carry the same call over other wire formats, unchanged, and map
# their protocol's in-band tool failure to ToolError.


class _JsonRpcTransport:
    """Sends a call as a self-contained JSON-RPC ``tools/call`` request."""

    def __init__(self, result: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.result = result or {"structuredContent": {"status": "ok"}}

    def __call__(
        self, *, url: str, payload: dict[str, JsonValue], timeout_seconds: float
    ) -> JsonValue:
        self.sent.append(
            {
                "jsonrpc": "2.0",
                "id": payload["call_id"],
                "method": "tools/call",
                "params": {
                    "name": payload["name"],
                    "arguments": payload["arguments"],
                },
            }
        )
        # The envelope is the transport's business; the environment sees the result.
        if self.result.get("isError"):
            raise ToolError(self.result["content"][0]["text"])
        return self.result["structuredContent"]


def test_a_json_rpc_transport_carries_the_call_unchanged():
    transport = _JsonRpcTransport()

    result = _environment(transport).call(
        "place_order", {"item": "X"}, call_id="c1", session_id="s1"
    )

    assert transport.sent == [
        {
            "jsonrpc": "2.0",
            "id": "c1",
            "method": "tools/call",
            "params": {"name": "place_order", "arguments": {"item": "X"}},
        }
    ]
    assert result.output == {"status": "ok"}


def test_a_json_rpc_transport_reports_is_error_as_the_tools_refusal():
    transport = _JsonRpcTransport(
        result={"isError": True, "content": [{"type": "text", "text": "out of stock"}]}
    )

    with pytest.raises(ToolError, match="^out of stock$"):
        _environment(transport).call("place_order", {"item": "X"}, call_id="c1")


class _OperationTransport:
    """Routes a call by looking its tool name up as an operation."""

    _OPERATIONS = {"place_order": {"method": "POST", "path": "/orders"}}

    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.status_code = status_code
        self.body = body or {"status": "ok"}

    def __call__(
        self, *, url: str, payload: dict[str, JsonValue], timeout_seconds: float
    ) -> JsonValue:
        operation = self._OPERATIONS[str(payload["name"])]
        self.sent.append(
            {
                "method": operation["method"],
                "url": url + operation["path"],
                "body": payload["arguments"],
            }
        )
        if self.status_code >= 400:
            raise ToolError(f"HTTP {self.status_code}: {self.body['error']}")
        return self.body


def test_an_operation_routing_transport_picks_the_route_from_the_tool_name():
    transport = _OperationTransport()

    result = _environment(transport).call("place_order", {"item": "X"}, call_id="c1")

    assert transport.sent == [
        {"method": "POST", "url": _URL + "/orders", "body": {"item": "X"}}
    ]
    assert result.output == {"status": "ok"}


def test_an_operation_routing_transport_reports_a_4xx_as_the_tools_refusal():
    transport = _OperationTransport(status_code=422, body={"error": "item unavailable"})

    with pytest.raises(ToolError, match="^HTTP 422: item unavailable$") as excinfo:
        _environment(transport).call("place_order", {"item": "X"}, call_id="c1")

    assert not isinstance(excinfo.value, EndpointCallError)
