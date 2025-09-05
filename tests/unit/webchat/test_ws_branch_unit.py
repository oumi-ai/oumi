# Copyright 2025 - Oumi

import asyncio
import json
import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.handlers.ws_handler import WebSocketHandler
from oumi.webchat.handlers.branch_handler import BranchHandler
from oumi.core.commands.conversation_branches import ConversationBranchManager
from tests.utils.chat_test_utils import create_test_inference_config


class FakeEngine:
    def generate_response(self, conversation: Conversation) -> Conversation:
        return Conversation(messages=[Message(role=Role.ASSISTANT, content="OK")])


class DummyWS:
    def __init__(self):
        self.messages = []

    async def send_str(self, s: str):
        try:
            self.messages.append(json.loads(s))
        except Exception:
            self.messages.append({"raw": s})


@pytest.mark.asyncio
async def test_session_manager_returns_same_session_object():
    cfg = create_test_inference_config()
    sm = SessionManager(cfg)
    s1 = await sm.get_or_create_session_safe("S")
    s2 = await sm.get_or_create_session_safe("S")
    assert id(s1) == id(s2)


@pytest.mark.asyncio
async def test_ws_handler_appends_to_target_branch_when_branch_id_specified():
    cfg = create_test_inference_config()
    sm = SessionManager(cfg)
    ws_handler = WebSocketHandler(sm)

    session = await sm.get_or_create_session_safe("S1")
    # Inject fake engine
    engine = FakeEngine()
    session.inference_engine = engine
    session.command_context.inference_engine = engine

    # Create branch and switch
    ok, msg, new_branch = session.branch_manager.create_branch(from_branch_id="main", name="side")
    assert ok and new_branch
    ok, msg, b = session.branch_manager.switch_branch(new_branch.id)
    assert ok

    ws = DummyWS()
    await ws_handler.handle_chat_message(
        session,
        {"type": "chat", "message": "Hello", "branch_id": new_branch.id},
        ws,
    )

    # Assert current branch is target and both turns present
    assert session.branch_manager.current_branch_id == new_branch.id
    assert session.conversation_history[-2]["role"] == "user"
    assert session.conversation_history[-2]["content"] == "Hello"
    assert session.conversation_history[-1]["role"] == "assistant"
    assert session.conversation_history[-1]["content"]

    # Branch conversation is in sync with session
    branch = session.branch_manager.get_current_branch()
    assert branch.conversation_history == session.conversation_history


@pytest.mark.asyncio
async def test_get_conversation_returns_branch_copy():
    cfg = create_test_inference_config()
    sm = SessionManager(cfg)
    bh = BranchHandler(sm, db=None)

    session = await sm.get_or_create_session_safe("S2")
    # Create a branch and seed it
    ok, msg, new_branch = session.branch_manager.create_branch(from_branch_id="main", name="alpha")
    assert ok and new_branch
    ok, msg, _ = session.branch_manager.switch_branch(new_branch.id)
    assert ok

    seeded = [
        {"role": "user", "content": "U", "timestamp": 1.0},
        {"role": "assistant", "content": "A", "timestamp": 2.0},
    ]
    session.branch_manager.branches[new_branch.id].conversation_history = seeded.copy()

    class DummyReq:
        def __init__(self, session_id, branch_id):
            self.query = {"session_id": session_id, "branch_id": branch_id}

    resp = await bh.handle_get_conversation_api(DummyReq("S2", new_branch.id))
    import json as _json
    data = _json.loads(resp.text)
    conv = data.get("conversation", [])
    assert conv == seeded
