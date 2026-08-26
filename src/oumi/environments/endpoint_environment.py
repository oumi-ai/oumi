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


class EndpointTransport(Protocol):
    """Sends one tool call and returns the endpoint's decoded JSON response.

    Implementations own the wire concerns the environment deliberately does not:
    authentication, egress policy, TLS, and timeouts. Raise any exception to
    report a failed call; the environment reports it as a tool error.
    """

    def __call__(
        self, *, url: str, payload: dict[str, JsonValue], timeout_seconds: float
    ) -> JsonValue:
        """Send ``payload`` to ``url`` and return the decoded response body."""


def _post_json(
    *, url: str, payload: dict[str, JsonValue], timeout_seconds: float
) -> JsonValue:
    """Default transport: a plain JSON POST with no egress policy of its own."""
    import requests

    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


@register_environment("endpoint")
class EndpointEnvironment(BaseEnvironment):
    """Environment that executes each tool call as one POST to an endpoint.

    The endpoint owns the tool's behavior; this environment owns the contract:
    it validates arguments against the tool's schema, sends one request per
    call, and validates the response against the tool's ``output_schema``.

    Each request carries a ``session_id`` identifying the conversation and a
    ``call_id`` identifying the call within it. A retry resends the identical
    request, so an endpoint that deduplicates on ``{session_id}:{call_id}``
    performs a side effect once no matter how often it is re-sent.
    """

    tool_params_cls = ToolParams

    def __init__(
        self,
        params: EnvironmentParams,
        kwargs: EndpointEnvironmentKwargs,
        transport: EndpointTransport = _post_json,
    ) -> None:
        """Bind the env to its endpoint and the transport that reaches it."""
        self._params = params
        self._kwargs = kwargs
        self._transport = transport
        self._tools_by_id: dict[str, ToolParams] = {
            tool.id: tool for tool in params.tools
        }
        self._session_id = uuid.uuid4().hex

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> EndpointEnvironment:
        """Build the env from its configured kwargs, over the default transport."""
        kwargs = parse_env_kwargs(
            EndpointEnvironmentKwargs, params, env_label="EndpointEnvironment"
        )
        return cls(params, kwargs)

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

        payload = self._build_payload(tool_id, arguments, call_id, session_id)
        try:
            output = self._transport(
                url=self._kwargs.endpoint_url,
                payload=payload,
                timeout_seconds=self._kwargs.timeout_seconds,
            )
        except Exception as error:
            raise EndpointCallError(
                f"Tool '{tool_id}' endpoint call failed: {error}"
            ) from error

        self._validate_output(tool, output)
        return ToolResult(output=output)

    def _build_payload(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        call_id: str,
        session_id: str | None,
    ) -> dict[str, JsonValue]:
        """Shape one call into the request body the endpoint is sent.

        This and :meth:`_validate_output` are the two ends of the wire contract:
        a different contract replaces both and leaves the rest of ``call``
        untouched.
        """
        return {
            "name": tool_id,
            "arguments": arguments,
            "call_id": call_id,
            "session_id": session_id or self._session_id,
        }

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
