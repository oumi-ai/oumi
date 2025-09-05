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

"""Integration test to ensure WS chat updates active branch history."""

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager

import aiohttp
import pytest
import websockets

from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.webchat.server import OumiWebServer


class _FakeEngine:
    """Minimal fake inference engine returning a deterministic assistant reply."""

    def generate_response(self, conversation: Conversation) -> Conversation:
        return Conversation(messages=[Message(role=Role.ASSISTANT, content="OK")])


@asynccontextmanager
async def _run_server(monkeypatch: pytest.MonkeyPatch):
    """Spin up the aiohttp app on an ephemeral port for the test."""
    # Use a tiny config and stub the engine builder to avoid heavy models
    cfg = InferenceConfig()

    # Patch build_inference_engine used by WebChatSession
    from oumi.webchat.core import session as session_mod

    monkeypatch.setattr(session_mod, "build_inference_engine", lambda **kwargs: _FakeEngine())

    # Use a temp DB path to avoid touching user files
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/webchat.sqlite"
        server = OumiWebServer(cfg, system_prompt=None, db_path=db_path)
        app = server.create_app()

        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        # Discover the bound port
        sockets = site._server.sockets  # type: ignore[attr-defined]
        port = sockets[0].getsockname()[1]

        try:
            yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_ws_chat_syncs_active_branch_history(monkeypatch: pytest.MonkeyPatch):
    """Send a WS chat on a non-main branch and verify branch history updated."""
    async with _run_server(monkeypatch) as (http_base, ws_base):
        session_id = "test_sync"

        # Connect WebSocket and read initial session payload
        ws_uri = f"{ws_base}/v1/oumi/ws?session_id={session_id}"
        async with websockets.connect(ws_uri) as ws:
            init = json.loads(await ws.recv())
            assert init.get("type") == "session_init"
            assert init.get("current_branch") == "main"

            # Create a new branch off main
            async with aiohttp.ClientSession() as http:
                create_resp = await http.post(
                    f"{http_base}/v1/oumi/branches",
                    json={
                        "action": "create",
                        "session_id": session_id,
                        "from_branch": "main",
                        "name": "side",
                    },
                )
                assert create_resp.status == 200
                create_json = await create_resp.json()
                branch_id = create_json.get("branch", {}).get("id")
                assert branch_id and branch_id.startswith("branch_")

                # Switch to the new branch
                switch_resp = await http.post(
                    f"{http_base}/v1/oumi/branches",
                    json={
                        "action": "switch",
                        "session_id": session_id,
                        "branch_id": branch_id,
                    },
                )
                assert switch_resp.status == 200

                # Verify the current branch is set before sending chat
                verify_resp = await http.get(
                    f"{http_base}/v1/oumi/branches",
                    params={"session_id": session_id},
                )
                assert verify_resp.status == 200
                verify_json = await verify_resp.json()
                assert verify_json.get("current_branch") == branch_id, (
                    f"Branch switch didn't stick: expected {branch_id}, got {verify_json.get('current_branch')}"
                )

                # Send a chat message over WS
                user_text = "Hello from branch!"
            await ws.send(json.dumps({
                "type": "chat",
                "message": user_text,
                "session_id": session_id,
            }))
            # Poll backend conversation for the active branch and verify user turn appears
            found = False
            async with aiohttp.ClientSession() as http:
                for _ in range(40):  # up to ~10s total
                    conv_resp = await http.get(
                        f"{http_base}/v1/oumi/conversation",
                        params={"session_id": session_id, "branch_id": branch_id},
                    )
                    assert conv_resp.status == 200
                    conv_json = await conv_resp.json()
                    conversation = conv_json.get("conversation", [])
                    # Accept either full user+assistant or at least the user turn
                    if conversation and conversation[-1].get("role") == "user" and conversation[-1].get("content") == user_text:
                        found = True
                        break
                    if len(conversation) >= 2 and conversation[-2].get("content") == user_text and conversation[-1].get("role") == "assistant":
                        found = True
                        break
                    await asyncio.sleep(0.25)
            assert found, "Branch conversation did not update with user message in time"
