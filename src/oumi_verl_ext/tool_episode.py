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

import json
from collections.abc import Collection
from typing import Any, Final

TOOL_AGENT_LOOP_NAME: Final = "oumi_tool_agent"
TRANSCRIPT_KEY: Final = "oumi_tool_transcript_json"
RENDERED_TOOLS_KEY: Final = "oumi_tool_schemas_json"


def judge_transcript(
    messages: list[dict[str, Any]],
    unparsed_turn_text: str | None,
    special_tokens: Collection[str],
) -> list[dict[str, Any]]:
    """The messages up to and including the last assistant message.

    A final turn verl ended before appending it comes last, as an assistant message
    holding ``unparsed_turn_text``: the decoded text of only that turn's tokens
    inside the final response_ids[:response_length], so the judge never scores
    tokens the policy doesn't train on. Each ``special_tokens`` string is removed
    from assistant content, since verl's parsers decode with
    skip_special_tokens=False; an end-of-turn-only turn becomes empty content.
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


def episode_extra_fields(
    transcript: list[dict[str, Any]], schemas: list[dict[str, Any]]
) -> dict[str, str]:
    """The judge's transcript and the rendered tools, as JSON strings."""
    return {
        TRANSCRIPT_KEY: json.dumps(transcript),
        RENDERED_TOOLS_KEY: json.dumps(schemas),
    }
