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

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
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
    EndpointProtocol,
    EndpointStatusError,
    JsonHttpProtocol,
    RemoteToolCall,
    RequestsJsonClient,
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


class _RecordingProtocol:
    """Captures each call and returns a canned response."""

    def __init__(
        self, response: JsonValue = None, error: Exception | None = None
    ) -> None:
        self.response: JsonValue = {"status": "ok"} if response is None else response
        self.error = error
        self.calls: list[RemoteToolCall] = []

    def call(self, request: RemoteToolCall) -> JsonValue:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class _RecordingHttpClient:
    """Captures each posted body and returns a canned response."""

    def __init__(self, response: JsonValue = None) -> None:
        self.response: JsonValue = {"status": "ok"} if response is None else response
        self.posted: list[JsonValue] = []

    def post_json(self, payload: JsonValue) -> JsonValue:
        self.posted.append(payload)
        return self.response


def _environment(protocol: EndpointProtocol) -> EndpointEnvironment:
    return EndpointEnvironment(
        EnvironmentParams(id="env", env_type="endpoint", tools=[_tool()]), protocol
    )


def test_call_hands_the_identified_call_to_the_protocol():
    protocol = _RecordingProtocol()
    env = _environment(protocol)

    result = env.call(
        "place_order",
        {"item": "X"},
        call_id="row42:3:0:place_order",
        session_id="row42",
    )

    assert result.output == {"status": "ok"}
    assert protocol.calls == [
        RemoteToolCall(
            name="place_order",
            arguments={"item": "X"},
            call_id="row42:3:0:place_order",
            session_id="row42",
        )
    ]


def test_step_identifies_calls_when_the_caller_does_not():
    protocol = _RecordingProtocol()
    env = _environment(protocol)

    env.step([("place_order", {"item": "X"}), ("place_order", {"item": "Y"})])

    # Distinct calls in one conversation: same session, different call ids.
    assert protocol.calls[0].call_id != protocol.calls[1].call_id
    assert protocol.calls[0].session_id == protocol.calls[1].session_id


def test_call_rejects_arguments_that_do_not_match_the_tool_schema():
    protocol = _RecordingProtocol()
    env = _environment(protocol)

    with pytest.raises(ToolArgumentError):
        env.call("place_order", {"item": 7}, call_id="c1", session_id="s1")
    assert protocol.calls == []


def test_call_rejects_an_unknown_tool():
    env = _environment(_RecordingProtocol())

    with pytest.raises(ToolLookupError):
        env.call("cancel_order", {"item": "X"}, call_id="c1", session_id="s1")


def test_call_reports_a_response_that_breaks_the_output_schema():
    env = _environment(_RecordingProtocol(response={"state": "ok"}))

    with pytest.raises(EndpointCallError, match="output schema"):
        env.call("place_order", {"item": "X"}, call_id="c1", session_id="s1")


def test_call_reports_a_protocol_failure_as_an_endpoint_error():
    env = _environment(_RecordingProtocol(error=TimeoutError("timed out")))

    with pytest.raises(EndpointCallError, match="timed out"):
        env.call("place_order", {"item": "X"}, call_id="c1", session_id="s1")


def test_an_endpoint_failure_is_not_a_tool_error():
    """The two lanes must stay distinguishable by type, not just by message."""
    assert not issubclass(EndpointCallError, ToolError)


def test_a_tool_error_from_the_protocol_passes_through_unchanged():
    env = _environment(_RecordingProtocol(error=ToolError("out of stock")))

    with pytest.raises(ToolError, match="^out of stock$") as excinfo:
        env.call("place_order", {"item": "X"}, call_id="c1", session_id="s1")

    # A refusal is the tool's answer, not an endpoint failure.
    assert not isinstance(excinfo.value, EndpointCallError)


def test_call_accepts_any_response_when_the_tool_declares_no_output_schema():
    tool = _tool()
    tool.output_schema = None
    env = EndpointEnvironment(
        EnvironmentParams(id="env", env_type="endpoint", tools=[tool]),
        _RecordingProtocol(response=["anything"]),
    )

    assert env.call(
        "place_order", {"item": "X"}, call_id="c1", session_id="s1"
    ).output == ["anything"]


@pytest.mark.parametrize(
    "kwargs",
    [
        EndpointEnvironmentKwargs(endpoint_url=""),
        EndpointEnvironmentKwargs(endpoint_url=_URL, timeout_seconds=0),
        EndpointEnvironmentKwargs(endpoint_url=_URL, timeout_seconds=-1),
        EndpointEnvironmentKwargs(endpoint_url=_URL, max_retries=-1),
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
    assert kwargs.max_retries == EndpointEnvironmentKwargs().max_retries


def test_json_http_protocol_sends_the_call_and_its_ids_as_one_body():
    client = _RecordingHttpClient()

    result = _environment(JsonHttpProtocol(client)).call(
        "place_order", {"item": "X"}, call_id="c1", session_id="s1"
    )

    assert client.posted == [
        {
            "name": "place_order",
            "arguments": {"item": "X"},
            "call_id": "c1",
            "session_id": "s1",
        }
    ]
    assert result.output == {"status": "ok"}


def test_from_params_posts_to_the_configured_endpoint(monkeypatch):
    sent: dict[str, object] = {}

    class _Response:
        ok = True

        def json(self) -> JsonValue:
            return {"status": "ok"}

    def fake_post(self, url, *, json, timeout):
        sent.update({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    env = EndpointEnvironment.from_params(
        EnvironmentParams(
            id="env",
            env_type="endpoint",
            tools=[_tool()],
            env_kwargs={"endpoint_url": _URL, "timeout_seconds": 7},
        )
    )

    result = env.call("place_order", {"item": "X"}, call_id="c1", session_id="s1")

    assert sent["url"] == _URL
    assert sent["timeout"] == 7
    assert sent["json"] == {
        "name": "place_order",
        "arguments": {"item": "X"},
        "call_id": "c1",
        "session_id": "s1",
    }
    assert result.output == {"status": "ok"}


# The environment fixes what a call means, never how it travels. These two
# protocols carry the same call over other wire formats, unchanged, and map
# their format's in-band tool failure to ToolError.


class _JsonRpcProtocol:
    """Sends a call as a self-contained JSON-RPC ``tools/call`` request."""

    def __init__(self, http_client: _RecordingHttpClient, result: dict | None = None):
        self._http_client = http_client
        self.result = result or {"structuredContent": {"status": "ok"}}

    def call(self, request: RemoteToolCall) -> JsonValue:
        self._http_client.post_json(
            {
                "jsonrpc": "2.0",
                "id": request.call_id,
                "method": "tools/call",
                "params": {"name": request.name, "arguments": request.arguments},
            }
        )
        # The envelope is the protocol's business; the environment sees the result.
        if self.result.get("isError"):
            raise ToolError(self.result["content"][0]["text"])
        return self.result["structuredContent"]


def test_a_json_rpc_protocol_carries_the_call_unchanged():
    client = _RecordingHttpClient()

    result = _environment(_JsonRpcProtocol(client)).call(
        "place_order", {"item": "X"}, call_id="c1", session_id="s1"
    )

    assert client.posted == [
        {
            "jsonrpc": "2.0",
            "id": "c1",
            "method": "tools/call",
            "params": {"name": "place_order", "arguments": {"item": "X"}},
        }
    ]
    assert result.output == {"status": "ok"}


def test_a_json_rpc_protocol_reports_is_error_as_the_tools_refusal():
    protocol = _JsonRpcProtocol(
        _RecordingHttpClient(),
        result={"isError": True, "content": [{"type": "text", "text": "out of stock"}]},
    )

    with pytest.raises(ToolError, match="^out of stock$"):
        _environment(protocol).call(
            "place_order", {"item": "X"}, call_id="c1", session_id="s1"
        )


class _OperationProtocol:
    """Routes a call by looking its tool name up as an operation."""

    _OPERATIONS = {"place_order": {"method": "POST", "path": "/orders"}}

    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.status_code = status_code
        self.body = body or {"status": "ok"}

    def call(self, request: RemoteToolCall) -> JsonValue:
        # The protocol picks the method and path; the client owns the origin.
        operation = self._OPERATIONS[request.name]
        self.sent.append(
            {
                "method": operation["method"],
                "path": operation["path"],
                "body": request.arguments,
            }
        )
        if self.status_code >= 400:
            raise ToolError(f"HTTP {self.status_code}: {self.body['error']}")
        return self.body


def test_an_operation_routing_protocol_picks_the_route_from_the_tool_name():
    protocol = _OperationProtocol()

    result = _environment(protocol).call(
        "place_order", {"item": "X"}, call_id="c1", session_id="s1"
    )

    assert protocol.sent == [
        {"method": "POST", "path": "/orders", "body": {"item": "X"}}
    ]
    assert result.output == {"status": "ok"}


def test_an_operation_routing_protocol_reports_a_4xx_as_the_tools_refusal():
    protocol = _OperationProtocol(status_code=422, body={"error": "item unavailable"})

    with pytest.raises(ToolError, match="^HTTP 422: item unavailable$") as excinfo:
        _environment(protocol).call(
            "place_order", {"item": "X"}, call_id="c1", session_id="s1"
        )

    assert not isinstance(excinfo.value, EndpointCallError)


class _FlakyHandler(BaseHTTPRequestHandler):
    """Fails with a retryable status until ``failures`` is exhausted."""

    failures = 2
    received: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).received.append(json.loads(body))
        if len(type(self).received) <= type(self).failures:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        answer = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    def log_message(self, *args: object) -> None:
        """Silence the default stderr request log."""


@pytest.fixture
def flaky_endpoint() -> Iterator[str]:
    """Run a server that answers 503 twice before succeeding."""
    _FlakyHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/tools"
    finally:
        server.shutdown()
        server.server_close()


def test_a_retried_call_resends_the_identical_body(flaky_endpoint):
    """What makes call_id useful: the endpoint can recognize the resend."""
    client = RequestsJsonClient(flaky_endpoint, timeout_seconds=5.0)

    result = _environment(JsonHttpProtocol(client)).call(
        "place_order", {"item": "X"}, call_id="c1", session_id="s1"
    )
    client.close()

    assert result.output == {"status": "ok"}
    assert len(_FlakyHandler.received) == 3
    # Every attempt carries the same ids, so a deduplicating endpoint acts once.
    assert all(body == _FlakyHandler.received[0] for body in _FlakyHandler.received)
    assert _FlakyHandler.received[0]["call_id"] == "c1"


def test_retries_can_be_turned_off(flaky_endpoint):
    client = RequestsJsonClient(flaky_endpoint, timeout_seconds=5.0, max_retries=0)

    with pytest.raises(EndpointCallError):
        _environment(JsonHttpProtocol(client)).call(
            "place_order", {"item": "X"}, call_id="c1", session_id="s1"
        )
    client.close()

    assert len(_FlakyHandler.received) == 1


class _StatusClient:
    """Answers every call with one non-2xx status."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    def post_json(self, payload: JsonValue) -> JsonValue:
        raise EndpointStatusError(
            self._status_code, f"Endpoint returned HTTP {self._status_code}."
        )


@pytest.mark.parametrize("status_code", [400, 403, 404, 422])
def test_a_4xx_is_the_tool_rejecting_the_call(status_code):
    """The model can act on this, so it must not look like a broken endpoint."""
    env = _environment(JsonHttpProtocol(_StatusClient(status_code)))

    with pytest.raises(ToolError) as excinfo:
        env.call("place_order", {"item": "X"}, call_id="c1", session_id="s1")

    assert not isinstance(excinfo.value, EndpointCallError)


@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
def test_a_transient_or_server_status_is_an_endpoint_failure(status_code):
    env = _environment(JsonHttpProtocol(_StatusClient(status_code)))

    with pytest.raises(EndpointCallError):
        env.call("place_order", {"item": "X"}, call_id="c1", session_id="s1")


def test_the_default_client_names_the_status_it_got(flaky_endpoint):
    """Without the status, the protocol cannot tell a refusal from a failure."""
    _FlakyHandler.failures = 99
    client = RequestsJsonClient(flaky_endpoint, timeout_seconds=5.0, max_retries=0)

    with pytest.raises(EndpointStatusError) as excinfo:
        client.post_json({"name": "place_order"})
    client.close()
    _FlakyHandler.failures = 2

    assert excinfo.value.status_code == 503
