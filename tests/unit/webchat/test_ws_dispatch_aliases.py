# Copyright 2025 - Oumi

import json
import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.handlers.ws_handler import WebSocketHandler
from tests.utils.chat_test_utils import create_test_inference_config


class FakeEngine:
    def generate_response(self, conversation: Conversation) -> Conversation:
        # Deterministic assistant reply to keep assertions simple
        return Conversation(messages=[Message(role=Role.ASSISTANT, content="OK")])


class DummyWS:
    def __init__(self):
        self.sent = []

    async def send_str(self, s: str):
        try:
            self.sent.append(json.loads(s))
        except Exception:
            self.sent.append({"raw": s})


@pytest.mark.asyncio
async def test_ws_dispatch_accepts_both_chat_types():
    cfg = create_test_inference_config()
    sm = SessionManager(cfg)
    ws_handler = WebSocketHandler(sm)

    session = await sm.get_or_create_session_safe("DISPATCH_1")
    engine = FakeEngine()
    session.inference_engine = engine
    session.command_context.inference_engine = engine

    ws = DummyWS()

    # Send with alias "chat"
    await ws_handler.handle_websocket_message(
        session, {"type": "chat", "message": "Hello"}, ws
    )
    # Now send with canonical "chat_message"
    await ws_handler.handle_websocket_message(
        session, {"type": "chat_message", "message": "World"}, ws
    )

    # Expect 2 user + 2 assistant turns
    roles = [m.get("role") for m in session.conversation_history]
    contents = [m.get("content") for m in session.conversation_history]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert contents[0] == "Hello"
    assert contents[2] == "World"

