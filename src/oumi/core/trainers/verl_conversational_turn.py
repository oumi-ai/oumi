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

"""verl-free core for the conversational (simulated-user) rollout."""

import dataclasses
from collections.abc import Callable

from oumi.core.types.conversation import Conversation, Message, Role

DEFAULT_DONE_SENTINEL = "[[END]]"


@dataclasses.dataclass
class RolloutState:
    """Rollout state; ``max_turns`` is a safety cap on simulated-user replies."""

    persona: str
    max_turns: int
    goal: str = ""
    turn_idx: int = 0


def build_user_turn_prompt(
    persona: str,
    history: list[Message],
    current_turn: int,
    max_turns: int,
    done_sentinel: str = DEFAULT_DONE_SENTINEL,
    *,
    goal: str = "",
) -> Conversation:
    """Builds the prompt for the next simulated-user turn."""
    system_prompt = persona
    if goal:
        system_prompt += f"\n\nYour goal for this conversation: {goal}"

    messages: list[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
    messages.extend(history)
    messages.append(
        Message(
            role=Role.USER,
            content=(
                f"You are the USER generating reply {current_turn}. You may generate "
                f"at most {max_turns} replies; this limit is a safety cap, not a "
                "target. Respond naturally to the assistant's latest message. Pursue "
                "your goal without repeating yourself or inventing facts. Do not "
                "prolong the conversation to reach the turn limit. If your goal has "
                "been satisfied, respond naturally and append "
                f"{done_sentinel}. Reply with ONLY your next message and stay in "
                "character."
            ),
        )
    )
    return Conversation(messages=messages)


def messages_to_history(messages: list[dict]) -> list[Message]:
    """Converts verl messages to user and assistant history."""
    return [
        Message(role=Role(m["role"]), content=(m.get("content") or ""))
        for m in messages
        if m.get("role") != "system"
    ]


def next_user_turn(
    state: RolloutState,
    messages: list[dict],
    infer_fn: Callable[[Conversation], str],
    done_sentinel: str = DEFAULT_DONE_SENTINEL,
) -> tuple[bool, str, float]:
    """Produces the next simulated-user turn, or ends the rollout at the cap."""
    if state.turn_idx >= state.max_turns:
        return True, "", 0.0
    state.turn_idx += 1
    prompt = build_user_turn_prompt(
        state.persona,
        messages_to_history(messages),
        state.turn_idx,
        state.max_turns,
        done_sentinel,
        goal=state.goal,
    )
    text = infer_fn(prompt)
    if done_sentinel in text:
        return True, text.replace(done_sentinel, "").strip(), 0.0
    return False, text.strip(), 0.0
