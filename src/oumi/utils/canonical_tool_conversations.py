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

"""The canonical tool-call conversation.

One conversation, wherever a tool-call example is needed. The chat-template
probe in :mod:`oumi.core.tokenizers.utils` renders it to work out which
tool-call form a template accepts, and the tests render it to check which
tokens land in the loss. Both read the same shape, so a template quirk that
breaks one is visible to the other instead of hiding behind a second,
differently-shaped copy.

Every value is a unique sentinel, so a caller can locate it in rendered text
or in a token span without matching on incidental words.
"""

import json
from typing import Final

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import (
    FunctionCall,
    FunctionDefinition,
    JSONSchema,
    ToolCall,
    ToolDefinition,
)

SYSTEM_TEXT: Final[str] = "SYSTEM_PROMPT_TEXT"
USER_TEXT_1: Final[str] = "USER_ASKS_WEATHER_AND_FLIGHTS"
USER_TEXT_2: Final[str] = "USER_ASKS_TOKYO"
ASSISTANT_TEXT_1: Final[str] = "ASSISTANT_ANSWERS_PARIS"
ASSISTANT_TEXT_2: Final[str] = "ASSISTANT_ANSWERS_TOKYO"
TOOL_RESULT_1: Final[str] = "TOOL_RESULT_PARIS_WEATHER"
TOOL_RESULT_2: Final[str] = "TOOL_RESULT_BOSTON_FLIGHTS"
TOOL_RESULT_3: Final[str] = "TOOL_RESULT_TOKYO_WEATHER"

# The value of the first tool call's only argument. Distinctive enough that
# finding it in rendered output means the template really did render that
# argument, and that finding it beside an escaped quote means the template
# encoded the arguments a second time.
ARGUMENT_SENTINEL: Final[str] = "OUMI_PROBE_7f3a"

# Index of the first assistant turn carrying tool calls. That turn holds two
# parallel calls and no content, which is the shape most template quirks show
# up on.
FIRST_TOOL_CALL_INDEX: Final[int] = 2


def _call(call_id: str, name: str, **arguments) -> ToolCall:
    """Builds a tool call, deliberately leaving ``type`` at its default.

    Conversations built in code rather than parsed from OpenAI-format JSON
    routinely omit ``type``, and some templates read it directly, so the
    canonical conversation exercises that path rather than papering over it.
    """
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


def _tool(name: str, *parameters: str) -> ToolDefinition:
    return ToolDefinition(
        function=FunctionDefinition(
            name=name,
            description=f"{name} description",
            parameters=JSONSchema(
                type="object",
                properties={p: JSONSchema(type="string") for p in parameters},
                required=list(parameters),
            ),
        )
    )


def canonical_tool_conversation() -> Conversation:
    """Multi-turn conversation with parallel tool calls and interleaved results.

    Returns a fresh instance per call, so a caller that adapts it for a
    template cannot disturb anyone else's copy.
    """
    return Conversation(
        tools=[_tool("get_weather", "city"), _tool("search_flights", "origin")],
        messages=[
            Message(role=Role.SYSTEM, content=SYSTEM_TEXT),
            Message(role=Role.USER, content=USER_TEXT_1),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    _call("weathr001", "get_weather", city=ARGUMENT_SENTINEL),
                    _call("flight001", "search_flights", origin="Boston"),
                ],
            ),
            Message(role=Role.TOOL, tool_call_id="weathr001", content=TOOL_RESULT_1),
            Message(role=Role.TOOL, tool_call_id="flight001", content=TOOL_RESULT_2),
            Message(role=Role.ASSISTANT, content=ASSISTANT_TEXT_1),
            Message(role=Role.USER, content=USER_TEXT_2),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[_call("weathr002", "get_weather", city="Tokyo")],
            ),
            Message(role=Role.TOOL, tool_call_id="weathr002", content=TOOL_RESULT_3),
            Message(role=Role.ASSISTANT, content=ASSISTANT_TEXT_2),
        ],
    )
