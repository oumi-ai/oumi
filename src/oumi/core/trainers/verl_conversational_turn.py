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
    """Per-rollout user-sim state, keyed by verl's instance_id."""

    persona: str
    max_turns: int
    goal: str | None = None
    turn_idx: int = 0


def build_user_turn_prompt(
    persona: str, history: list[Message], current_turn: int, target_turns: int
) -> Conversation:
    """Assemble the user-sim prompt: persona (system) + history + turn instruction."""
    messages: list[Message] = [Message(role=Role.SYSTEM, content=persona)]
    messages.extend(history)
    messages.append(
        Message(
            role=Role.USER,
            content=(
                f"You are the USER (turn {current_turn} of at most {target_turns}). "
                "Continue the conversation in character and reply with ONLY your next "
                "message. If your goal is fully met and you have nothing more to ask, "
                f"end your reply with {DEFAULT_DONE_SENTINEL}."
            ),
        )
    )
    return Conversation(messages=messages)


def messages_to_history(messages: list[dict]) -> list[Message]:
    """Map verl's [{'role','content'}, ...] into Oumi Messages."""
    return [
        Message(role=Role(m["role"]), content=(m.get("content") or ""))
        for m in messages
    ]


def next_user_turn(
    state: RolloutState,
    messages: list[dict],
    infer_fn: Callable[[Conversation], str],
    done_sentinel: str = DEFAULT_DONE_SENTINEL,
) -> tuple[bool, str, float]:
    """Produce the next user turn (or end)."""
    state.turn_idx += 1
    if state.turn_idx >= state.max_turns:  # hard cap → no more user turns
        return True, "", 0.0
    prompt = build_user_turn_prompt(
        state.persona, messages_to_history(messages), state.turn_idx, state.max_turns
    )
    text = infer_fn(prompt)
    if done_sentinel in text:  # user-decided end
        return True, text.replace(done_sentinel, "").strip(), 0.0
    return False, text.strip(), 0.0
