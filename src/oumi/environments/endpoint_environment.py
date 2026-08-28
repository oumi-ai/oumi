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
from requests.adapters import HTTPAdapter, Retry

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError, ToolLookupError, ToolParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.utils import parse_env_kwargs

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 3
_RETRY_BACKOFF_FACTOR = 0.5
_RETRY_STATUSES = (429, 500, 502, 503, 504)
# 408 and 429 are the endpoint asking to be tried again, not the tool refusing.
_TRANSIENT_CLIENT_STATUSES = (408, 429)


class EndpointStatusError(Exception):
    """Raised by a client when the endpoint answered with a non-2xx status.

    Carries the status so a protocol can tell the tool refusing the call apart
    from the endpoint failing to serve it. Clients raise this instead of their
    own HTTP error type, which is what keeps that decision in the protocol.
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Name the status the endpoint answered with."""
        super().__init__(message)
        self.status_code = status_code


class EndpointCallError(Exception):
    """Raised when the endpoint cannot be reached or answers unusably.

    Deliberately not a :class:`ToolError`: the tool never answered, so a caller
    can tell an endpoint failure apart from a tool refusing the call and choose
    whether to retry it, fail the row, or tell the model something generic.
    """


@dataclass
class EndpointEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for :class:`EndpointEnvironment`."""

    endpoint_url: str = ""
    """The endpoint every tool call is sent to."""

    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    """How long to wait for the endpoint to answer one call."""

    max_retries: int = _DEFAULT_MAX_RETRIES
    """Retries for a connection failure or a retryable status.

    A retry re-sends the identical body, so an endpoint deduplicating on
    ``{session_id}:{call_id}`` performs the side effect once. Set to 0 for an
    endpoint that does not deduplicate.
    """

    def __finalize_and_validate__(self) -> None:
        """Validate the endpoint configuration."""
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")


@dataclass(frozen=True)
class RemoteToolCall:
    """One tool call to send, identified within its conversation."""

    name: str
    """Id of the tool being called."""

    arguments: dict[str, Any]
    """Arguments, already validated against the tool's schema."""

    call_id: str
    """Identifies this call. Stable across retries of the same call."""

    session_id: str
    """Identifies the conversation the call belongs to."""


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
    client rather than reusing :class:`JsonHttpClient`. The client holds the
    connection, so releasing it is the job of whoever built it.

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
    """

    def call(self, request: RemoteToolCall) -> JsonValue:
        """Execute one tool call and return the tool's output."""


class RequestsJsonClient:
    """Default client: a plain JSON POST with no egress policy of its own."""

    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        """Send every call to ``url``, waiting at most ``timeout_seconds``."""
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()
        # urllib3 leaves POST out of allowed_methods because a resend can repeat
        # a side effect; the call's ids are what make resending safe here.
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=max_retries,
                backoff_factor=_RETRY_BACKOFF_FACTOR,
                status_forcelist=_RETRY_STATUSES,
                allowed_methods=frozenset({"POST"}),
                # Hand back the exhausted response so the status reaches the
                # protocol, rather than urllib3 raising its own pool error.
                raise_on_status=False,
            )
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def post_json(self, payload: JsonValue) -> JsonValue:
        """POST ``payload`` and return the decoded response body."""
        response = self._session.post(
            self._url, json=payload, timeout=self._timeout_seconds
        )
        if not response.ok:
            raise EndpointStatusError(
                response.status_code, f"Endpoint returned HTTP {response.status_code}."
            )
        return response.json()

    def close(self) -> None:
        """Release the pooled connections. Not part of :class:`JsonHttpClient`."""
        self._session.close()


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
        """POST one call and return the response body as the tool's output.

        Raises:
            ToolError: If the endpoint answered 4xx, which is the tool rejecting
                the call rather than the endpoint failing to serve it.
            EndpointStatusError: For any other non-2xx status.
        """
        try:
            return self._http_client.post_json(
                {
                    "name": request.name,
                    "arguments": request.arguments,
                    "call_id": request.call_id,
                    "session_id": request.session_id,
                }
            )
        except EndpointStatusError as error:
            if self._is_tool_refusal(error.status_code):
                raise ToolError(str(error)) from error
            raise

    @staticmethod
    def _is_tool_refusal(status_code: int) -> bool:
        """Whether this status is the tool rejecting the call."""
        return (
            400 <= status_code < 500 and status_code not in _TRANSIENT_CLIENT_STATUSES
        )


@register_environment("endpoint")
class EndpointEnvironment(BaseEnvironment):
    """Environment that executes each tool call against a remote endpoint.

    The endpoint owns the tool's behavior; this environment owns the contract:
    it validates arguments against the tool's schema, hands the call to a
    protocol, and validates the answer against the tool's ``output_schema``.
    The protocol owns the wire format, so a different one leaves this class
    untouched.

    Each call is identified by a caller-supplied ``session_id`` naming the
    conversation and a ``call_id`` naming the call within it, so a protocol can
    make retries deduplicable end to end.

    Shareable across samples, so the harness never closes it. Whoever builds
    the protocol's client owns releasing it.
    """

    tool_params_cls = ToolParams

    def __init__(self, params: EnvironmentParams, protocol: EndpointProtocol) -> None:
        """Bind the env to the protocol that reaches its endpoint."""
        self._params = params
        self._protocol = protocol
        self._tools_by_id: dict[str, ToolParams] = {
            tool.id: tool for tool in params.tools
        }

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> EndpointEnvironment:
        """Build the env from its configured kwargs, over a plain JSON POST."""
        kwargs = parse_env_kwargs(
            EndpointEnvironmentKwargs, params, env_label="EndpointEnvironment"
        )
        client = RequestsJsonClient(
            kwargs.endpoint_url, kwargs.timeout_seconds, kwargs.max_retries
        )
        return cls(params, JsonHttpProtocol(client))

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute a batch of tool calls; results are returned in input order.

        Identifies the calls itself, treating the batch as one conversation.
        Callers holding stable ids for a conversation and its calls should use
        :meth:`call` so retries stay deduplicable end to end.
        """
        session_id = uuid.uuid4().hex
        return [
            self.call(
                tool_id, arguments, call_id=uuid.uuid4().hex, session_id=session_id
            )
            for tool_id, arguments in calls
        ]

    def call(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        call_id: str,
        session_id: str,
    ) -> ToolResult:
        """Execute one tool call against the endpoint.

        ``session_id`` names the conversation and ``call_id`` the call within it.
        Both come from the caller: this env is shared across samples, so it
        cannot know which conversation a call belongs to.

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
            session_id=session_id,
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
