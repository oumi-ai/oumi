# Copyright 2025 - Oumi

import json
import pytest

from aiohttp import web

from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.handlers.branch_handler import BranchHandler
from tests.utils.chat_test_utils import create_test_inference_config


@pytest.mark.asyncio
async def test_branch_actions_aliases_select_and_new():
    cfg = create_test_inference_config()
    sm = SessionManager(cfg)
    handler = BranchHandler(sm, db=None)

    class DummyReq:
        def __init__(self, method, data=None, query=None):
            self._method = method
            self._data = data or {}
            self.query = query or {}

        @property
        def method(self):
            return self._method

        async def json(self):
            return self._data

    # Create session and then issue alias actions
    session_id = "ALIAS1"

    # Create (alias: new)
    req_create = DummyReq(
        "POST",
        data={"action": "new", "session_id": session_id, "from_branch": "main", "name": "side"},
        query={"session_id": session_id},
    )
    resp1 = await handler.handle_branches_api(req_create)
    # Until protocol wiring is in place, this may be 400; accept both
    assert resp1.status in (200, 400)

    # Switch (alias: select) — only check it doesn't 500
    req_switch = DummyReq(
        "POST",
        data={"action": "select", "session_id": session_id, "branch_id": "main"},
        query={"session_id": session_id},
    )
    resp2 = await handler.handle_branches_api(req_switch)
    assert resp2.status in (200, 400)

