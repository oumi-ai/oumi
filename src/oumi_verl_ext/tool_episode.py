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

"""The judge's view of a verl tool episode, in the stdlib so CI runs it."""

from collections.abc import Collection
from typing import Any, Final

TOOL_AGENT_LOOP_NAME: Final = "oumi_tool_agent"
MALFORMED_CALLS_KEY: Final = "oumi_tool_calls_malformed"
DROPPED_CALLS_KEY: Final = "oumi_tool_calls_dropped"


def call_problems(
    turn_text: str, call_start: str | None, parsed_calls: int, max_parallel_calls: int
) -> tuple[int, int]:
    """Counts a parsed turn's malformed and dropped tool calls.

    Malformed calls are call-start markers the parser couldn't read; a parser with
    no such marker (gpt-oss, kimi) shows none. Dropped calls are the ones past
    ``max_parallel_calls``, which verl discards without running.
    """
    attempted = turn_text.count(call_start) if call_start else parsed_calls
    # qwen3_coder can parse several calls from one marker.
    return max(attempted - parsed_calls, 0), max(parsed_calls - max_parallel_calls, 0)


def stopped_by_turn_limit(
    response_tokens: int,
    response_length: int,
    assistant_turns: int,
    max_assistant_turns: int | None,
) -> bool:
    """Whether ``max_assistant_turns`` stopped verl before it parsed a turn.

    verl checks the response budget first, so a turn that hits both is a budget cut.
    """
    return response_tokens < response_length and bool(
        max_assistant_turns and assistant_turns >= max_assistant_turns
    )


def judge_transcript(
    messages: list[dict[str, Any]],
    unparsed_turn_text: str | None,
    special_tokens: Collection[str],
) -> list[dict[str, Any]]:
    """The messages up to the last assistant message, for the judge.

    ``unparsed_turn_text`` is a final turn verl stopped before parsing. Special
    tokens are removed from assistant content, since verl decodes with them.
    """
    transcript = list(messages)
    if unparsed_turn_text is not None:
        transcript.append({"role": "assistant", "content": unparsed_turn_text})
    while transcript and transcript[-1]["role"] != "assistant":
        transcript.pop()
    for index, message in enumerate(transcript):
        if message["role"] == "assistant" and isinstance(message["content"], str):
            content = message["content"]
            for token in special_tokens:
                content = content.replace(token, "")
            transcript[index] = {**message, "content": content}
    return transcript


def full_tool_schemas(
    config_schemas: list[dict[str, Any]], schemas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``schemas`` with each tool's ``tool_schema`` from the tool config file.

    verl parses that tool_schema into models that keep only the keys they declare,
    so nested properties, array items, defaults and the like are lost. A tool the
    file gives no tool_schema, such as a function tool, keeps verl's schema.
    """
    by_name = {schema["function"]["name"]: schema for schema in config_schemas}
    return [by_name.get(schema["function"]["name"], schema) for schema in schemas]
