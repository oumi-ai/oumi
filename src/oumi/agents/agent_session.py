# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Stateful chat session that wires an LLM to one or more environments."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from oumi.agents.exceptions import (
    AgentSessionError,
    InvalidToolArgumentsError,
    UnknownToolError,
)
from oumi.agents.tool_result_format import compact_tool_output
from oumi.agents.tool_router import ToolRouter
from oumi.builders.environments import build_environment
from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.agent_harness_config import AgentHarnessConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import ToolCall
from oumi.environments.base_environment import BaseEnvironment

EventKind = Literal[
    "user",
    "inference_start",
    "inference_end",
    "tool_call",
    "tool_result",
    "assistant_text",
    "turn_end",
]
# Event payloads:
#   user               { "text": str }
#   inference_start    {}
#   inference_end      { "usage": dict | None }
#   tool_call          { "name": str, "arguments": str, "call_id": str }
#   tool_result        { "name": str, "payload": Any, "is_error": bool,
#                        "duration_ms": float }
#   assistant_text     { "text": str }
#   turn_end           { "tool_calls": int, "session_total_tokens": int }

EventHandler = Callable[[EventKind, dict], None]


class AgentSession:
    """Stateful chat session for ``oumi agent``.

    Construct via :meth:`from_config`; use :meth:`send` to push a user
    message and receive the assistant's final reply (after any tool-call
    cycles have resolved). Use as a context manager so environments are
    closed cleanly on exit.
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        envs: dict[str, BaseEnvironment],
        router: ToolRouter,
        system_prompt: str = "",
        max_tool_result_chars: int = 8192,
        on_event: EventHandler | None = None,
    ) -> None:
        """Initialize an agent session over the given engine and environments."""
        self._engine = engine
        self._envs = envs
        self._router = router
        self._max_tool_result_chars = max_tool_result_chars
        self._on_event: EventHandler = on_event or (lambda _kind, _data: None)
        self._session_total_tokens = 0

        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        self._conv = Conversation(messages=messages, tools=router.tool_specs)

    @classmethod
    def from_config(cls, config: AgentHarnessConfig) -> AgentSession:
        """Build an :class:`AgentSession` from a validated harness config."""
        config.finalize_and_validate()

        engine_type = config.inference.engine or InferenceEngineType.NATIVE
        try:
            engine = build_inference_engine(
                engine_type=engine_type,
                model_params=config.inference.model,
                remote_params=config.inference.remote_params,
                generation_params=config.inference.generation,
            )
        except Exception as e:
            raise AgentSessionError(
                f"Failed to build inference engine '{engine_type}': {e}"
            ) from e

        envs: dict[str, BaseEnvironment] = {}
        try:
            for env_params in config.environment.environments:
                envs[env_params.id] = build_environment(env_params)
        except Exception as e:
            for env in envs.values():
                env.close()
            raise AgentSessionError(f"Failed to build environments: {e}") from e

        router = ToolRouter.from_environment_config(config.environment)
        return cls(
            engine=engine,
            envs=envs,
            router=router,
            system_prompt=config.system_prompt,
        )

    @property
    def conversation(self) -> Conversation:
        """The full message log so far. Treat as read-only."""
        return self._conv

    def close(self) -> None:
        """Close every environment owned by this session."""
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                # Best-effort: one env's failure must not block the rest.
                continue

    def __enter__(self) -> AgentSession:
        """Enter context; return self."""
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:  # noqa: ANN001
        """Exit context; close all environments."""
        self.close()

    def set_event_handler(self, handler: EventHandler | None) -> None:
        """Attach (or detach) an event hook. Handlers run synchronously."""
        self._on_event = handler or (lambda _kind, _data: None)

    def send(self, user_message: str) -> str:
        """Send a user message; loop until the assistant stops calling tools."""
        self._on_event("user", {"text": user_message})
        self._conv.messages.append(Message(role=Role.USER, content=user_message))
        seen_calls: set[tuple[str, str]] = set()
        tool_calls_this_turn = 0

        while True:
            self._on_event("inference_start", {})
            assistant, usage = self._step_engine()
            self._on_event("inference_end", {"usage": usage})
            if not assistant.tool_calls:
                text = assistant.compute_flattened_text_content()
                self._on_event("assistant_text", {"text": text})
                self._on_event(
                    "turn_end",
                    {
                        "tool_calls": tool_calls_this_turn,
                        "session_total_tokens": self._session_total_tokens,
                    },
                )
                return text
            for call in assistant.tool_calls:
                tool_calls_this_turn += 1
                self._on_event(
                    "tool_call",
                    {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                        "call_id": call.id,
                    },
                )
                started = time.monotonic()
                tool_msg, payload, is_error = self._handle_tool_call(call, seen_calls)
                duration_ms = (time.monotonic() - started) * 1000.0
                self._conv.messages.append(tool_msg)
                self._on_event(
                    "tool_result",
                    {
                        "name": call.function.name,
                        "payload": payload,
                        "is_error": is_error,
                        "duration_ms": duration_ms,
                    },
                )

    def _step_engine(self) -> tuple[Message, dict[str, int] | None]:
        """One round-trip to the inference engine."""
        [updated] = self._engine.infer([self._conv])
        new_messages = updated.messages[len(self._conv.messages) :]
        if not new_messages:
            raise AgentSessionError(
                "Inference engine returned no new messages — cannot continue."
            )
        for msg in new_messages:
            self._conv.messages.append(msg)

        usage = updated.metadata.get("usage") if updated.metadata else None
        if isinstance(usage, dict):
            self._session_total_tokens += int(usage.get("total_tokens", 0))
        return self._conv.messages[-1], usage if isinstance(usage, dict) else None

    def _handle_tool_call(
        self,
        call: ToolCall,
        seen_calls: set[tuple[str, str]],
    ) -> tuple[Message, Any, bool]:
        """Validate, dedup, execute one tool call.

        Returns ``(tool_message, payload, is_error)``.
        """
        wire_name = call.function.name
        raw_args = call.function.arguments or "{}"

        try:
            env_id, tool = self._router.route(wire_name)
        except UnknownToolError as e:
            return self._tool_error(call.id, "unknown_tool", str(e))

        dedup_key = (wire_name, raw_args)
        if dedup_key in seen_calls:
            return self._tool_error(
                call.id,
                "duplicate_call",
                f"Tool '{wire_name}' was already called with these exact "
                f"arguments in this turn.",
            )
        seen_calls.add(dedup_key)

        try:
            arguments = self._router.parse_and_validate_arguments(tool, raw_args)
        except InvalidToolArgumentsError as e:
            return self._tool_error(call.id, "invalid_arguments", str(e))

        env = self._envs[env_id]
        try:
            result = env.step(tool.id, arguments)
        except Exception as e:
            return self._tool_error(call.id, type(e).__name__, str(e))

        payload = compact_tool_output(
            result.output, max_chars=self._max_tool_result_chars
        )
        msg = Message(
            role=Role.TOOL,
            content=json.dumps(payload, default=str),
            tool_call_id=call.id,
        )
        is_error = isinstance(payload, dict) and payload.get("status") == "error"
        return msg, payload, is_error

    @staticmethod
    def _tool_error(
        tool_call_id: str, error_kind: str, message: str
    ) -> tuple[Message, dict[str, Any], bool]:
        payload: dict[str, Any] = {
            "status": "error",
            "error": error_kind,
            "message": message,
        }
        msg = Message(
            role=Role.TOOL,
            content=json.dumps(payload),
            tool_call_id=tool_call_id,
        )
        return msg, payload, True
