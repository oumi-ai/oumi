import json
import pytest

from aiohttp import web

from oumi.core.configs import InferenceConfig
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.handlers.branch_handler import BranchHandler


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_sync_requires_session_id_returns_400():
    cfg = InferenceConfig()
    sm = SessionManager(cfg)
    bh = BranchHandler(sm, db=None)

    # Missing session_id in body
    req = FakeRequest({
        "conversation": [
            {"role": "user", "content": "hi", "timestamp": 1.0}
        ],
        # no session_id
    })

    resp: web.Response = await bh.handle_sync_conversation_api(req)
    assert resp.status == 400
    data = json.loads(resp.text)
    assert 'error' in data
    # Ensure we include a helpful hint per implementation
    assert 'hint' in data


@pytest.mark.asyncio
async def test_sync_updates_only_target_branch():
    cfg = InferenceConfig()
    sm = SessionManager(cfg)
    bh = BranchHandler(sm, db=None)

    session_id = 'SYNC_TARGET'
    session = await sm.get_or_create_session_safe(session_id)

    # Seed main branch with distinct content
    session.branch_manager.branches['main'].conversation_history = [
        {"role": "user", "content": "MAIN-KEEP", "timestamp": 0.5}
    ]

    # Create target branch; by design it inherits source branch history at branch time
    ok, msg, new_branch = session.branch_manager.create_branch(from_branch_id='main', name='b1')
    assert ok and new_branch
    assert session.branch_manager.branches[new_branch.id].conversation_history == [
        {"role": "user", "content": "MAIN-KEEP", "timestamp": 0.5}
    ]

    # Prepare sync payload targeting new branch
    new_conv = [
        {"role": "user", "content": "U1", "timestamp": 1.0},
        {"role": "assistant", "content": "A1", "timestamp": 2.0},
    ]
    req = FakeRequest({
        "session_id": session_id,
        "conversation": new_conv,
        "branch_id": new_branch.id,
    })

    resp: web.Response = await bh.handle_sync_conversation_api(req)
    assert resp.status == 200

    # Verify target branch updated, main untouched
    updated = await sm.get_or_create_session_safe(session_id)
    tgt_hist = updated.branch_manager.branches[new_branch.id].conversation_history
    main_hist = updated.branch_manager.branches['main'].conversation_history

    assert tgt_hist == new_conv
    assert main_hist == [{"role": "user", "content": "MAIN-KEEP", "timestamp": 0.5}]
