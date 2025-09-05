# Copyright 2025 - Oumi

import pytest


protocol = pytest.importorskip("oumi.webchat.protocol")


def test_normalize_msg_type_aliases():
    norm = protocol.normalize_msg_type
    assert norm("chat") == "chat"
    assert norm("chat_message") == "chat"
    assert norm("COMMAND") == "command"
    assert norm("branches") in ("get_branches", "branches")  # allow choice
    assert norm("system_stats") in ("system_monitor", "system_stats")


def test_normalize_branch_action_aliases():
    norm = protocol.normalize_branch_action
    assert norm("create") == "create"
    assert norm("new") == "create"
    assert norm("switch") == "switch"
    assert norm("select") == "switch"
    assert norm("delete") == "delete"
    assert norm("remove") == "delete"


def test_extract_session_id_precedence_and_missing():
    # Pending exact behavior decision; mark xfail for missing case until wired
    DummyReq = type("DummyReq", (), {})
    req = DummyReq()
    req.query = {"session_id": "Q"}

    # Query should take precedence over body
    body = {"session_id": "B"}
    sid = protocol.extract_session_id(req, body)
    assert sid == "Q"

    # Body should work when query absent
    req2 = DummyReq()
    req2.query = {}
    sid2 = protocol.extract_session_id(req2, {"session_id": "ONLY_BODY"})
    assert sid2 == "ONLY_BODY"

    # If required and missing, expect an error (pending wiring)
    req3 = DummyReq()
    req3.query = {}
    with pytest.raises(Exception):
        protocol.extract_session_id(req3, {}, required=True)

