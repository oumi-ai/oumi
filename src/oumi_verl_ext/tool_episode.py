# Copyright 2026 - Oumi
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

"""Episode rules for oumi's verl tool agent loop, in the stdlib so CI runs them."""

import copy
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

TOOL_AGENT_LOOP_NAME: Final = "oumi_tool_agent"
# The converter's column and extra_info key, and the loop's kwarg.
TOOLS_JSON_KEY: Final = "tools_json"
TRANSCRIPT_KEY: Final = "oumi_tool_transcript_json"
RENDERED_TOOLS_KEY: Final = "oumi_tool_schemas_json"
CALLS_KEY: Final = "oumi_tool_calls"
DROPPED_CALLS_KEY: Final = "oumi_tool_calls_dropped"
MALFORMED_CALLS_KEY: Final = "oumi_tool_calls_malformed"
ENDS_EPISODE_CONFIG_KEY: Final = "ends_episode"


@dataclass(frozen=True)
class EpisodeTools:
    """The tools one episode offers."""

    schemas: list[dict[str, Any]]  # rendered in the prompt, in offer order
    executable: frozenset[str]  # any other offered name ends the episode

    @property
    def ending(self) -> frozenset[str]:
        """Offered names that are not executable."""
        names = frozenset(tool["function"]["name"] for tool in self.schemas)
        return names - self.executable


def parse_row_tools(tools_json: str | None) -> list[dict[str, Any]]:
    """A row's tools_json as tool dicts; [] when None or empty.

    Raises:
        ValueError: naming tools_json, never echoing it, if it isn't a JSON list
            of {"type": "function", "function": {"name": ...}} objects.
    """
    if not tools_json:
        return []
    try:
        tools = json.loads(tools_json)
    except (TypeError, ValueError):
        tools = None
    if not isinstance(tools, list) or not all(
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
        for tool in tools
    ):
        raise ValueError(
            f"'{TOOLS_JSON_KEY}' must be a JSON list of function tools, each with "
            "a function name."
        )
    return tools


def choose_episode_tools(
    row_tools: list[dict[str, Any]],
    run_tools: Mapping[str, dict[str, Any]],
    ends_episode_tools: frozenset[str],
) -> EpisodeTools:
    """Evaluation's toolset rule.

    With run tools attached, the episode offers exactly the run's tools; each renders
    from the row's dict of the same name when the row carries one (lossless), else
    from ``run_tools[name]``; ``ends_episode_tools`` (the run tools whose config sets
    ends_episode) are not executable. With none attached, the row's tools render
    unchanged and none is executable.
    """
    if not run_tools:
        return EpisodeTools(schemas=list(row_tools), executable=frozenset())
    row_tools_by_name = {tool["function"]["name"]: tool for tool in row_tools}
    return EpisodeTools(
        schemas=[row_tools_by_name.get(name, tool) for name, tool in run_tools.items()],
        executable=frozenset(run_tools) - ends_episode_tools,
    )


def verl_parser_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """A copy of a tool dict that verl's OpenAIFunctionToolSchema accepts.

    Fills in what verl requires and oumi's JSONSchema may leave out or give as a
    list: function.description, parameters and its properties, a string
    parameters.type ("object"), and every JSON type for a property without one
    (anyOf- or enum-only). A null required is dropped. The rendered dict is unchanged.

    Raises:
        ValueError: naming the tool, never its content, if the copy still breaks the
            shape of v0.9.1 verl/tools/schemas.py:21-56.
    """
    parser_tool = copy.deepcopy(tool)
    function = parser_tool["function"]
    error = ValueError(
        f"Tool '{function.get('name')}' has a schema verl's tool parser can't read."
    )
    if function.get("description") is None:
        function["description"] = ""
    if function.get("parameters") is None:
        function["parameters"] = {}
    parameters = function["parameters"]
    if not isinstance(parameters, dict):
        raise error
    if not isinstance(parameters.get("type"), str):
        parameters["type"] = "object"
    if parameters.get("properties") is None:
        parameters["properties"] = {}
    if parameters.get("required", []) is None:
        del parameters["required"]
    properties = parameters["properties"]
    if not isinstance(properties, dict):
        raise error
    for prop in properties.values():
        if isinstance(prop, dict) and prop.get("type") is None:
            prop["type"] = "string number integer boolean object array null".split()
    if not (
        isinstance(parser_tool.get("type"), str)
        and isinstance(function.get("name"), str)
        and isinstance(function["description"], str)
        and isinstance(function.get("strict", False), bool)
        and _is_str_list(parameters.get("required", []))
        and all(
            isinstance(prop, dict)
            and (isinstance(prop["type"], str) or _is_str_list(prop["type"]))
            and isinstance(prop.get("description"), str | None)
            and isinstance(prop.get("enum"), list | None)
            for prop in properties.values()
        )
    ):
        raise error
    return parser_tool


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


@dataclass
class CallCounts:
    """One episode's tool-call counters."""

    calls: int = 0
    dropped: int = 0
    malformed: int = 0

    def add_turn(
        self,
        *,
        attempted: int,
        call_names: Sequence[str],
        max_parallel_calls: int,
        ending: frozenset[str] = frozenset(),
        cut: bool = False,
        capped: bool = False,
    ) -> bool:
        """Counts one assistant turn; True when it ends the episode.

        attempted: call-start markers in the turn's text. call_names: the parsed
        calls' names, in order. Only the first max_parallel_calls are kept (verl
        records and runs only those); the rest count as dropped. A kept call to an
        ``ending`` tool ends the episode, and the turn's other kept calls, which
        don't run, count as dropped. cut: verl ended the turn before parsing it, so
        its attempted calls count as dropped, not malformed, and it returns False.
        capped: the turn came once user turns reached max_user_turns, so it returns
        True and its kept calls, which don't run, count as dropped, except a call to
        an ``ending`` tool, which counts as the call that ended the episode.
        """
        if cut:
            self.dropped += attempted
            return False
        kept = call_names[:max_parallel_calls]
        self.calls += len(call_names)
        # qwen3_coder can parse several calls from one marker.
        self.malformed += max(attempted - len(call_names), 0)
        self.dropped += len(call_names) - len(kept)
        ends = any(name in ending for name in kept)
        if not (ends or capped):
            return False
        self.dropped += len(kept) - int(ends)
        return True


def judge_transcript(
    messages: list[dict[str, Any]],
    truncated_turn_text: str | None,
    special_tokens: Collection[str],
) -> list[dict[str, Any]]:
    """The messages up to and including the last assistant message.

    A turn the response budget cut before verl appended it comes last, as an
    assistant message holding ``truncated_turn_text``: the decoded text of only that
    turn's tokens inside the final response_ids[:response_length], so the judge never
    scores tokens the policy doesn't train on. Each ``special_tokens``
    string is removed from assistant content, since verl's parsers decode with
    skip_special_tokens=False; an end-of-turn-only turn becomes empty content.
    """
    transcript = list(messages)
    if truncated_turn_text is not None:
        transcript.append({"role": "assistant", "content": truncated_turn_text})
    while transcript and transcript[-1]["role"] != "assistant":
        transcript.pop()
    for index, message in enumerate(transcript):
        if message["role"] == "assistant" and isinstance(message["content"], str):
            content = message["content"]
            for token in special_tokens:
                content = content.replace(token, "")
            transcript[index] = {**message, "content": content}
    return transcript


def episode_extra_fields(
    transcript: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
    counts: CallCounts,
) -> dict[str, object]:
    """The judge's transcript and rendered tools as JSON strings, and the counters."""
    return {
        TRANSCRIPT_KEY: json.dumps(transcript),
        RENDERED_TOOLS_KEY: json.dumps(schemas),
        CALLS_KEY: counts.calls,
        DROPPED_CALLS_KEY: counts.dropped,
        MALFORMED_CALLS_KEY: counts.malformed,
    }
