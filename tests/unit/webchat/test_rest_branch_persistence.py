import json
import time
import pytest

from aiohttp import web

from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.handlers.chat_handler import ChatHandler
from oumi.webchat.persistence import WebchatDB


class DummyEngine:
    def infer(self, input, inference_config):
        # Mirror request conversation with an assistant reply appended
        conv: Conversation = input[-1]
        msgs = list(conv.messages)
        msgs.append(Message(role=Role.ASSISTANT, content="ok"))
        return [Conversation(messages=msgs)]


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_rest_chat_persists_to_target_branch(tmp_path):
    cfg = InferenceConfig()
    sm = SessionManager(cfg, system_prompt=None)
    db_path = tmp_path / "webchat.sqlite"
    db = WebchatDB(str(db_path))
    handler = ChatHandler(sm, system_prompt=None, db=db, enhanced_features_available=False)

    # Prepare a session and create a branch to target
    session_id = "sess_branch_persist"
    session = await sm.get_or_create_session_safe(session_id, db=None)
    # Inject engine
    session.command_context.inference_engine = DummyEngine()

    ok, _, new_branch = session.branch_manager.create_branch(from_branch_id="main", name="alpha")
    assert ok and new_branch

    # Prepare REST payload targeting the new branch
    payload = {
        "model": "unit-model",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "session_id": session_id,
        "branch_id": new_branch.id,
        "stream": False,
    }
    req = FakeRequest(payload)

    resp: web.Response = await handler.handle_chat_completions(req)
    assert resp.status == 200

    # The session should be on the targeted branch
    updated = await sm.get_or_create_session_safe(session_id, db=None)
    assert updated.branch_manager.current_branch_id == new_branch.id

    # Dual-write should have created a conversation and appended two messages to the target branch
    info = db.get_session_info(session_id)
    assert info is not None
    conv_id = info["current_conversation_id"]
    assert conv_id

    # Ensure branch exists in DB and has 2 messages
    branches = db.get_session_branches(session_id)
    branch_ids = {b["id"] for b in branches}
    assert new_branch.id in branch_ids

    msgs = db.get_branch_messages(new_branch.id)
    # Expect last two messages (user + assistant) persisted
    assert len(msgs) >= 2
    assert msgs[-2]["role"] == "user" and msgs[-2]["content"] == "hello"
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == "ok"

