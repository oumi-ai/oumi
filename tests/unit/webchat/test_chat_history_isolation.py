import asyncio
import json
import time
import types

import pytest

from aiohttp import web

from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.handlers.chat_handler import ChatHandler
from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation, Message, Role


class DummyEngine:
    def infer(self, input, inference_config):
        # input is expected to be a list with one Conversation
        conv: Conversation = input[-1]
        # Return a new conversation with the same messages plus one assistant reply
        msgs = list(conv.messages)
        msgs.append(Message(role=Role.ASSISTANT, content="ok"))
        return [Conversation(messages=msgs)]


@pytest.mark.asyncio
async def test_rest_chat_uses_only_client_messages(monkeypatch):
    """When calling /v1/chat/completions with session_id, ensure backend does not
    prepend stale session history and only uses client-provided messages for context
    and session persistence."""

    # Minimal config
    cfg = InferenceConfig()
    sm = SessionManager(cfg, system_prompt=None)
    handler = ChatHandler(sm, system_prompt=None, db=None, enhanced_features_available=False)

    # Seed the session with prior history that must NOT leak
    session_id = "sess_leak_test"
    session = await sm.get_or_create_session_safe(session_id, db=None)
    session.conversation_history.extend([
        {"role": "user", "content": "old-1", "timestamp": time.time()},
        {"role": "assistant", "content": "old-2", "timestamp": time.time()},
    ])

    # Force our dummy engine
    session.command_context.inference_engine = DummyEngine()

    # Build a fake web.Request object with .json()
    class FakeRequest:
        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

    # Client sends exactly one user message for a fresh conversation turn
    payload = {
        "model": "unit-model",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "session_id": session_id,
        "stream": False,
    }

    req = FakeRequest(payload)

    # Invoke handler
    resp: web.Response = await handler.handle_chat_completions(req)
    assert resp.status == 200
    data = json.loads(resp.text)
    # Response should include assistant content from DummyEngine
    assert data["choices"][0]["message"]["content"] == "ok"

    # Verify the backend session history contains exactly the client messages + assistant
    updated = await sm.get_or_create_session_safe(session_id, db=None)
    hist = updated.conversation_history
    assert len(hist) == 2, f"Unexpected history len: {len(hist)}; hist={hist}"
    assert hist[0]["role"] == "user" and hist[0]["content"] == "hello"
    assert hist[1]["role"] == "assistant" and hist[1]["content"] == "ok"


@pytest.mark.asyncio
async def test_rest_chat_with_prior_turns_keeps_only_client_snapshot(monkeypatch):
    """If UI sends multiple messages (existing conversation turns), backend should
    persist exactly that snapshot + assistant, not any earlier session content."""

    cfg = InferenceConfig()
    sm = SessionManager(cfg, system_prompt=None)
    handler = ChatHandler(sm, system_prompt=None, db=None, enhanced_features_available=False)

    session_id = "sess_multi_turn"
    session = await sm.get_or_create_session_safe(session_id, db=None)
    # Stale content that must not be included
    session.conversation_history.extend([
        {"role": "user", "content": "stale-u", "timestamp": time.time()},
        {"role": "assistant", "content": "stale-a", "timestamp": time.time()},
    ])
    session.command_context.inference_engine = DummyEngine()

    payload = {
        "model": "unit-model",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
        "session_id": session_id,
        "stream": False,
    }

    class FakeRequest:
        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

    req = FakeRequest(payload)
    resp: web.Response = await handler.handle_chat_completions(req)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert data["choices"][0]["message"]["content"] == "ok"

    updated = await sm.get_or_create_session_safe(session_id, db=None)
    hist = updated.conversation_history
    # Should be u1, a1, u2, ok
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in hist] == ["u1", "a1", "u2", "ok"]

