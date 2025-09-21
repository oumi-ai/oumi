"""Id-first node regeneration handler for Oumi WebChat."""

from typing import Any, Dict, Optional
import time

from aiohttp import web

from oumi.utils.logging import logger
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.protocol import extract_session_id
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.webchat.utils.id_utils import generate_message_id


class RegenHandler:
    def __init__(self, session_manager: SessionManager, db=None):
        self.session_manager = session_manager
        self.db = db

    async def handle_regen_node_api(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        # Inputs
        try:
            session_id = extract_session_id(request, data)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        branch_id = data.get("branch_id")
        assistant_id = data.get("assistant_id")
        user_message_id = data.get("user_message_id")
        prompt = data.get("prompt")
        history_mode = data.get("history_mode", "last_user")  # last_user | full | none
        is_electron = bool(data.get("electron"))

        # Get session
        session = await self.session_manager.get_or_create_session_safe(session_id, self.db)

        # Optionally switch branch
        async def _maybe_switch(s):
            if branch_id and branch_id != s.branch_manager.current_branch_id:
                try:
                    s.branch_manager.sync_conversation_history(s.conversation_history)
                    ok, msg, br = s.branch_manager.switch_branch(branch_id)
                    if ok and br:
                        s.conversation_history.clear()
                        s.conversation_history.extend(br.conversation_history)
                        logger.debug(f"regen_node: switched to branch {branch_id}")
                except Exception as e:
                    logger.warning(f"regen_node: branch switch failed: {e}")
            return s

        session = await self.session_manager.execute_session_operation(session_id, _maybe_switch)

        # Resolve target and build prompt if needed
        resolved_index: Optional[int] = None
        resolved_prompt: Optional[str] = None

        if prompt and isinstance(prompt, str):
            resolved_prompt = prompt
        elif user_message_id:
            for i, m in enumerate(session.conversation_history):
                if m.get("id") == user_message_id and m.get("role") == "user":
                    resolved_index = i
                    resolved_prompt = str(m.get("content", ""))
                    break
        elif assistant_id:
            # Find assistant and its preceding user
            for i, m in enumerate(session.conversation_history):
                if m.get("id") == assistant_id and m.get("role") == "assistant":
                    resolved_index = i
                    # Search backwards for preceding user
                    for j in range(i - 1, -1, -1):
                        if session.conversation_history[j].get("role") == "user":
                            resolved_prompt = str(session.conversation_history[j].get("content", ""))
                            break
                    break

        if not resolved_prompt:
            return web.json_response({"error": "Unable to resolve prompt for regeneration"}, status=400)

        # If assistant index resolved, truncate in-place
        if resolved_index is not None:
            try:
                del session.conversation_history[resolved_index:]
            except Exception:
                session.conversation_history[resolved_index:] = []

        # Build inference conversation
        convo_msgs = []
        # Optional system prompt
        if hasattr(session, 'system_prompt') and session.system_prompt:
            convo_msgs.append(Message(role=Role.SYSTEM, content=session.system_prompt))

        if history_mode == "full":
            for m in session.conversation_history:
                r = Role.USER if m.get("role") == "user" else (Role.ASSISTANT if m.get("role") == "assistant" else Role.SYSTEM)
                c = str(m.get("content", ""))
                if c:
                    convo_msgs.append(Message(role=r, content=c))
        else:
            # last_user or none -> just single user message with resolved_prompt
            convo_msgs.append(Message(role=Role.USER, content=resolved_prompt))

        full_conversation = Conversation(messages=convo_msgs)

        # Choose engine/config respecting /swap
        session_config = session.config
        session_engine = session.inference_engine
        if hasattr(session.command_context, 'config') and session.command_context.config:
            session_config = session.command_context.config
        if hasattr(session.command_context, 'inference_engine') and session.command_context.inference_engine:
            session_engine = session.command_context.inference_engine

        # Inference
        try:
            model_response = session_engine.infer(input=[full_conversation], inference_config=session_config)
        except Exception as e:
            logger.error(f"regen_node: inference error: {e}")
            return web.json_response({"error": f"Inference failed: {e}"}, status=500)

        response_content = ""
        if model_response:
            last_conv = model_response[-1] if isinstance(model_response, list) else model_response
            for msg in reversed(last_conv.messages):
                if msg.role == Role.ASSISTANT and isinstance(msg.content, str):
                    response_content = msg.content
                    break
        if not response_content:
            response_content = "No response generated"

        # Append assistant to session history
        session.conversation_history.append({
            "id": generate_message_id(),
            "role": "assistant",
            "content": response_content,
            "timestamp": time.time(),
        })

        # Sync branch snapshot
        try:
            current_branch = session.branch_manager.get_current_branch()
            current_branch.conversation_history = session.conversation_history.copy()
            current_branch.last_active = time.time()
        except Exception as e:
            logger.debug(f"regen_node: branch sync failed: {e}")

        # Persist and align ids (best-effort)
        new_id = session.conversation_history[-1].get("id")
        if self.db:
            try:
                self.db.ensure_session(session_id)
                conv_id = self.db.ensure_conversation(session_id)
                session.current_conversation_id = conv_id
                if not getattr(session, 'is_hydrated_from_db', False):
                    session.is_hydrated_from_db = True
                self.db.ensure_branch(conv_id, session.branch_manager.current_branch_id, name=session.branch_manager.current_branch_id)
                db_id = self.db.append_message_to_branch(
                    conv_id,
                    session.branch_manager.current_branch_id,
                    role="assistant",
                    content=response_content,
                    created_at=float(session.conversation_history[-1].get("timestamp", time.time())),
                    force_new=is_electron,
                )
                session.conversation_history[-1]["id"] = db_id
                new_id = db_id
                self.db.set_session_current_branch(session_id, conv_id, session.branch_manager.current_branch_id)
            except Exception as pe:
                logger.warning(f"regen_node: persistence failed: {pe}")

        # Broadcast update
        try:
            await session.broadcast_to_websockets(
                {
                    "type": "conversation_update",
                    "conversation": session.serialize_conversation(),
                    "branches": session.get_enhanced_branch_info(self.db),
                    "current_branch": session.branch_manager.current_branch_id,
                    "timestamp": time.time(),
                }
            )
        except Exception:
            pass

        return web.json_response(
            {
                "success": True,
                "assistant": {"id": new_id, "content": response_content},
                "target": {"assistant_id": assistant_id, "user_message_id": user_message_id, "resolved_index": resolved_index},
                "broadcast": True,
            }
        )

