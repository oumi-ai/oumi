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

"""Backend-free core for simulated-user (conversational) rollouts."""

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

from oumi.core.types.conversation import Conversation, Message, Role

DONE_SENTINEL = "[[END]]"
DEFAULT_MAX_TURNS = 6


@dataclasses.dataclass
class RolloutState:
    """Per-conversation state of a simulated user.

    Attributes:
        persona: System prompt describing who the simulated user is.
        max_turns: Safety cap on the number of simulated-user replies.
        goal: What the simulated user wants out of the conversation.
        turn_idx: Number of simulated-user replies generated so far.
    """

    persona: str
    max_turns: int
    goal: str = ""
    turn_idx: int = 0


def build_user_turn_prompt(state: RolloutState, history: list[Message]) -> Conversation:
    """Builds the prompt for the simulated user's next reply.

    Args:
        state: The simulated user. `state.turn_idx` is the reply being generated.
        history: The user/assistant turns so far.

    Returns:
        A conversation to send to the simulator's inference engine.
    """
    system_prompt = state.persona
    if state.goal:
        system_prompt += f"\n\nYour goal for this conversation: {state.goal}"
    instruction = (
        f"You are the USER generating reply {state.turn_idx}. You may generate "
        f"at most {state.max_turns} replies; this limit is a safety cap, not a "
        "target. Respond naturally to the assistant's latest message. Pursue "
        "your goal without repeating yourself or inventing facts. Do not "
        "prolong the conversation to reach the turn limit. If your goal has "
        f"been satisfied, respond naturally and append {DONE_SENTINEL}. Reply "
        "with ONLY your next message and stay in character."
    )
    return Conversation(
        messages=[
            Message(role=Role.SYSTEM, content=system_prompt),
            *history,
            Message(role=Role.USER, content=instruction),
        ]
    )


def messages_to_history(messages: list[dict[str, Any]]) -> list[Message]:
    """Keeps only the user and assistant text turns of a chat-format message list.

    The simulated user is a person talking to the assistant, so it never sees
    system prompts or tool traffic. Chat APIs would also reject a tool turn that
    does not follow an assistant message carrying structured `tool_calls`.

    Args:
        messages: Messages as `{"role", "content"}` dicts.

    Returns:
        The conversation as the simulated user saw it.
    """
    return [
        Message(role=Role(m["role"]), content=m["content"])
        for m in messages
        if m["role"] in (Role.USER.value, Role.ASSISTANT.value) and m.get("content")
    ]


async def next_user_turn(
    state: RolloutState,
    messages: list[dict[str, Any]],
    generate: Callable[[Conversation], Awaitable[str]],
) -> tuple[bool, str]:
    """Generates the simulated user's next reply and increments `state.turn_idx`.

    Args:
        state: The simulated user.
        messages: The conversation so far, as `{"role", "content"}` dicts.
        generate: Returns the simulator model's reply to a prompt.

    Returns:
        `(done, text)`. `done` is whether the reply contains the done sentinel.
        `text` is the reply with the sentinel stripped.
    """
    state.turn_idx += 1
    text = await generate(build_user_turn_prompt(state, messages_to_history(messages)))
    return DONE_SENTINEL in text, text.replace(DONE_SENTINEL, "").strip()
