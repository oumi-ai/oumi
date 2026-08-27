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

"""Environment that forwards tool calls to a remote HTTP endpoint."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import jsonschema
import requests
from pydantic import JsonValue

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError, ToolLookupError, ToolParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.utils import parse_env_kwargs

_DEFAULT_TIMEOUT_SECONDS = 30.0


class EndpointCallError(ToolError):
    """Raised when the endpoint cannot be reached or answers unusably."""


@dataclass
class EndpointEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for :class:`EndpointEnvironment`."""

    endpoint_url: str = ""
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    def __finalize_and_validate__(self) -> None:
        """Validate the endpoint configuration."""
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


@dataclass(frozen=True)
class RemoteToolCall:
    """One tool call to send, identified within its conversation."""

    name: str
    arguments: dict[str, Any]
    call_id: str
    session_id: str


class JsonHttpClient(Protocol):
    """POSTs a JSON body to one fixed endpoint and decodes the JSON answer.

    What :class:`JsonHttpProtocol` sends over. The client owns the URL, the
    credential, and the egress policy, so a protocol never chooses where a call
    goes or what it is allowed to reach. A protocol needing more than a JSON
    POST declares its own client rather than widening this one.
    """

    def post_json(self, payload: JsonValue) -> JsonValue:
        """POST ``payload`` and return the decoded response body."""


class EndpointProtocol(Protocol):
    """Turns one tool call into a request, and its answer into the tool's output.

    Implementations own the wire format: the shape of the request, and which
    answers count as the tool refusing rather than the endpoint failing.

    Raise :class:`ToolError` (or a subclass) when the tool itself refused the
    call — an in-band answer that reaches the model verbatim. Raise anything
    else when the endpoint could not answer at all.

    A protocol also brings the client that suits it. MCP over Streamable HTTP
    answers a call with either JSON or an event stream, so it declares its own
    client rather than reusing :class:`JsonHttpClient`.

    Example:
        ::

            class McpProtocol:
                def __init__(self, mcp_client):
                    self._mcp_client = mcp_client

                def call(self, request):
                    result = self._mcp_client.call_tool(
                        name=request.name, arguments=request.arguments
                    )
                    if result.get("isError"):
                        raise ToolError(result["content"][0]["text"])
                    return result["structuredContent"]

                def close(self):
                    self._mcp_client.close()
    """

    def call(self, request: RemoteToolCall) -> JsonValue:
        """Execute one tool call and return the tool's output."""

    def close(self) -> None:
        """Release any protocol-level resources, such as a negotiated session."""


class RequestsJsonClient:
    """Default client: a plain JSON POST with no egress policy of its own."""

    def __init__(self, url: str, timeout_seconds: float) -> None:
        """Send every call to ``url``, waiting at most ``timeout_seconds``."""
        self._url = url
        self._timeout_seconds = timeout_seconds

    def post_json(self, payload: JsonValue) -> JsonValue:
        """POST ``payload`` and return the decoded response body."""
        response = requests.post(self._url, json=payload, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()


class JsonHttpProtocol:
    """Sends each tool call as one JSON POST whose response body is the output.

    The request carries the call's name and arguments alongside the ids
    identifying it. A retry resends the identical body, so an endpoint that
    deduplicates on ``{session_id}:{call_id}`` performs a side effect once no
    matter how often it is re-sent.
    """

    def __init__(self, http_client: JsonHttpClient) -> None:
        """Send over ``http_client``, which owns the URL and the credential."""
        self._http_client = http_client

    def call(self, request: RemoteToolCall) -> JsonValue:
        """POST one call and return the response body as the tool's output."""
        return self._http_client.post_json(
            {
                "name": request.name,
                "arguments": request.arguments,
                "call_id": request.call_id,
                "session_id": request.session_id,
            }
        )

    def close(self) -> None:
        """Hold nothing: the client's lifetime belongs to whoever built it."""


@register_environment("endpoint")
class EndpointEnvironment(BaseEnvironment):
    """Environment that executes each tool call against a remote endpoint.

    The endpoint owns the tool's behavior; this environment owns the contract:
    it validates arguments against the tool's schema, hands the call to a
    protocol, and validates the answer against the tool's ``output_schema``.
    The protocol owns the wire format, so a different one leaves this class
    untouched.

    Each call is identified by a ``session_id`` naming the conversation and a
    ``call_id`` naming the call within it, so a protocol can make retries
    deduplicable end to end.
    """

    tool_params_cls = ToolParams

    def __init__(self, params: EnvironmentParams, protocol: EndpointProtocol) -> None:
        """Bind the env to the protocol that reaches its endpoint."""
        self._params = params
        self._protocol = protocol
        self._tools_by_id: dict[str, ToolParams] = {
            tool.id: tool for tool in params.tools
        }
        self._session_id = uuid.uuid4().hex

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> EndpointEnvironment:
        """Build the env from its configured kwargs, over a plain JSON POST."""
        kwargs = parse_env_kwargs(
            EndpointEnvironmentKwargs, params, env_label="EndpointEnvironment"
        )
        client = RequestsJsonClient(kwargs.endpoint_url, kwargs.timeout_seconds)
        return cls(params, JsonHttpProtocol(client))

    def close(self) -> None:
        """Release the protocol's resources, if it holds any."""
        self._protocol.close()

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute a batch of tool calls; results are returned in input order.

        Identifies the calls itself. Callers that already have stable ids for a
        conversation and its calls should use :meth:`call` so retries stay
        deduplicable end to end.
        """
        return [
            self.call(tool_id, arguments, call_id=uuid.uuid4().hex)
            for tool_id, arguments in calls
        ]

    def call(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        call_id: str,
        session_id: str | None = None,
    ) -> ToolResult:
        """Execute one tool call against the endpoint.

        Raises:
            ToolLookupError: If the environment does not serve ``tool_id``.
            ToolArgumentError: If ``arguments`` do not match the tool's schema.
            ToolError: As raised by the protocol, when the tool itself refused
                the call.
            EndpointCallError: If the endpoint is unreachable or its response
                does not match the tool's output schema.
        """
        tool = self._tools_by_id.get(tool_id)
        if tool is None:
            raise ToolLookupError(
                f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
                f"Available tools: {sorted(self._tools_by_id)}"
            )
        tool.validate_arguments(arguments)

        request = RemoteToolCall(
            name=tool_id,
            arguments=arguments,
            call_id=call_id,
            session_id=session_id or self._session_id,
        )
        try:
            output = self._protocol.call(request)
        except ToolError:
            # The protocol speaks the wire format; its tool-level verdict is
            # already precise, so wrapping it would only blur the message.
            raise
        except Exception as error:
            raise EndpointCallError(
                f"Tool '{tool_id}' endpoint call failed: {error}"
            ) from error

        self._validate_output(tool, output)
        return ToolResult(output=output)

    @staticmethod
    def _validate_output(tool: ToolParams, output: JsonValue) -> None:
        """Check a response against the tool's output schema, if it declares one.

        Raises:
            EndpointCallError: If the response does not match the schema.
        """
        if tool.output_schema is None:
            return
        try:
            jsonschema.validate(output, tool.output_schema)
        except jsonschema.ValidationError as error:
            raise EndpointCallError(
                f"Tool '{tool.id}' endpoint response does not match its output "
                f"schema: {error.message}"
            ) from error
