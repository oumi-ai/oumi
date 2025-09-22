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

"""Command execution handler for Oumi WebChat server."""

import io
import time
import traceback
from typing import Dict, Optional, Any

from aiohttp import web
from rich.console import Console

from oumi.core.commands.command_parser import ParsedCommand
from oumi.utils.logging import logger
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.utils.id_utils import generate_message_id


class CommandHandler:
    """Handles command execution requests for Oumi WebChat."""
    
    def __init__(
        self, 
        session_manager: SessionManager,
        db = None
    ):
        """Initialize command handler.
        
        Args:
            session_manager: Session manager for WebChat sessions
            db: Optional WebchatDB instance for persistence
        """
        self.session_manager = session_manager
        self.db = db
    
    async def handle_command_api(self, request: web.Request) -> web.Response:
        """Handle command execution via REST API.

        Args:
            request: Web request with command execution parameters

        Returns:
            JSON response with command execution result
        """
        # Parse JSON
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Extract inputs with safe defaults
        session_id = data.get("session_id", "default")
        command = data.get("command", "")
        args = data.get("args", [])
        branch_id = data.get("branch_id")
        message_id = data.get("message_id")
        index_hint = data.get("index")
        payload = data.get("payload")  # e.g., new content for edit
        is_electron = bool(data.get("electron"))

        logger.info(f"🌐 API: Received command '{command}' with args: {args} for session: {session_id}")

        try:
            # Get or create the session
            session = await self.session_manager.get_or_create_session_safe(session_id, self.db)

            # Optional branch switch
            if branch_id and branch_id != session.branch_manager.current_branch_id:
                async def _switch_branch(s):
                    try:
                        s.branch_manager.sync_conversation_history(s.conversation_history)
                        ok, _msg, br = s.branch_manager.switch_branch(branch_id)
                        if ok and br:
                            s.conversation_history.clear()
                            s.conversation_history.extend(br.conversation_history)
                            logger.debug(f"[CMD] switched branch -> {branch_id}")
                            logger.debug(f"[CMD] post-switch history len={len(s.conversation_history)} obj={id(s.conversation_history)}")
                    except Exception as e:
                        logger.warning(f"[CMD] branch switch failed: {e}")
                    return s
                session = await self.session_manager.execute_session_operation(session_id, _switch_branch)

            # Helper: resolve id/index
            def _normalize_target():
                convo = session.conversation_history or []
                resolved_id = None
                resolved_index = None
                if message_id:
                    for i, m in enumerate(convo):
                        if m.get("id") == message_id:
                            resolved_id = message_id
                            resolved_index = i
                            break
                if resolved_id is None and index_hint is not None:
                    try:
                        i = int(index_hint)
                        if 0 <= i < len(convo):
                            resolved_index = i
                            resolved_id = convo[i].get("id")
                    except Exception:
                        pass
                return resolved_id, resolved_index

            # Capture console output during command execution
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, width=80)
            original_console = session.command_context.console
            session.command_context.console = temp_console

            # Log pre-command snapshot
            try:
                try:
                    pre_len = len(session.conversation_history)
                    first_id = session.conversation_history[0].get("id") if pre_len else None
                    last_id = session.conversation_history[-1].get("id") if pre_len else None
                    logger.info(f"[CMD] Pre-command history len={pre_len} first={first_id} last={last_id} obj={id(session.conversation_history)}")
                except Exception:
                    pass
                
                # Normalize id-first commands to index-based arg
                resolved_id = None
                resolved_index = None
                if command in {"delete", "regen", "edit"}:
                    resolved_id, resolved_index = _normalize_target()
                    if resolved_index is None:
                        return web.json_response({
                            "success": False,
                            "error": f"Unable to resolve target for {command}. Please refresh and retry.",
                            "should_continue": False
                        }, status=400)
                    if command in {"delete", "regen"}:
                        args = [str(resolved_index)]
                    else:  # edit
                        new_content = payload if payload is not None else (args[1] if len(args) > 1 else "")
                        args = [str(resolved_index), new_content]
                    logger.info(f"[CMD] normalized {command} -> index={resolved_index}, id={resolved_id}")

                # Execute via router
                parsed_command = ParsedCommand(
                    command=command,
                    args=args,
                    kwargs={},
                    raw_input=f"/{command}({','.join(args)})",
                )
                logger.info(f"🌐 API: Executing command '{command}' via command router...")
                result = session.command_router.handle_command(parsed_command)
                try:
                    post_len = len(session.conversation_history)
                    first_id = session.conversation_history[0].get("id") if post_len else None
                    last_id = session.conversation_history[-1].get("id") if post_len else None
                    logger.info(f"[CMD] Post-command history len={post_len} first={first_id} last={last_id} obj={id(session.conversation_history)}")
                except Exception:
                    pass

                # Combine message + captured console
                console_output = string_buffer.getvalue().strip()
                full_message = result.message or ""
                if console_output:
                    full_message = (full_message + "\n\n" if full_message else "") + console_output
            finally:
                session.command_context.console = original_console

            # Build base response
            response_data: Dict[str, Any] = {
                "success": result.success,
                "message": full_message,
                "should_continue": result.should_continue,
            }

            # Always include routing identifiers for clients to target updates precisely
            try:
                response_data["session_id"] = session_id
                response_data["branch_id"] = session.branch_manager.current_branch_id
                if self.db is not None:
                    # Do not force-create new conversations here; ensure_conversation returns existing id
                    response_data["conversation_id"] = self.db.ensure_conversation(session_id)
                else:
                    # Fallback to session.session_id as stable handle when DB is unavailable
                    response_data["conversation_id"] = getattr(session, "current_conversation_id", None) or session.session_id
            except Exception:
                pass

            # Target metadata for id-first commands
            if command in {"delete", "regen", "edit"}:
                # Recompute in case history changed; safe best-effort
                t_id, t_idx = _normalize_target()
                response_data["target"] = {"message_id": t_id, "index": t_idx}
                if command == "edit" and result.success and payload is not None and t_idx is not None:
                    response_data["updated"] = {"message_id": t_id, "index": t_idx, "content": payload}
                    response_data["broadcast"] = True

            # Enrich responses for specific commands
            # Always include current model info when available to help clients annotate messages
            try:
                if hasattr(session.command_context, 'config') and session.command_context.config:
                    cfg = session.command_context.config
                    response_data.setdefault("model_info", {
                        "name": getattr(cfg.model, "model_name", None),
                        "engine": str(cfg.engine) if getattr(cfg, 'engine', None) else None,
                    })
                elif hasattr(session, 'config') and session.config:
                    cfg = session.config
                    response_data.setdefault("model_info", {
                        "name": getattr(cfg.model, "model_name", None),
                        "engine": str(cfg.engine) if getattr(cfg, 'engine', None) else None,
                    })
            except Exception as _mi_err:
                pass

            if command in ["branches", "list_branches"]:
                response_data["branches"] = session.get_enhanced_branch_info(self.db)
                response_data["current_branch"] = session.branch_manager.current_branch_id
            elif command == "show":
                response_data["conversation"] = session.serialize_conversation()
            elif command == "swap" and result.success:
                # model_info was already added above, but keep compatibility
                pass

            # For state-mutating commands, include the latest conversation snapshot directly
            if command in {"clear", "delete", "regen", "edit"} and result.success:
                response_data["conversation"] = session.serialize_conversation()
                response_data["branches"] = session.get_enhanced_branch_info(self.db)
                response_data["current_branch"] = session.branch_manager.current_branch_id
                response_data.setdefault("broadcast", True)

            # Follow-up inference for regen
            if result.success and result.should_continue and getattr(result, 'user_input_override', None):
                logger.info(f"🌐 API: Command '{command}' requires follow-up inference with user_input_override")
                if getattr(result, 'is_regeneration', False):
                    try:
                        await session._perform_regeneration_inference(result.user_input_override)
                    except Exception as e:
                        logger.error(f"❌ Error during regeneration inference: {e}")
                        await session.broadcast_to_websockets({"type": "error", "message": f"Regeneration failed: {str(e)}", "timestamp": time.time()})

            # Persist EDIT in-place and keep branch/session identity consistent
            if result.success and command == "edit":
                try:
                    # Ensure conversation exists in DB
                    conv_id = None
                    if self.db is not None:
                        conv_id = self.db.ensure_conversation(session_id)
                        self.db.ensure_branch(conv_id, session.branch_manager.current_branch_id, name=session.branch_manager.current_branch_id)
                    # Update edited row by sequence index when DB available
                    if self.db is not None and resolved_index is not None:
                        edited = session.conversation_history[resolved_index]
                        updated_id = self.db.update_branch_message(
                            conv_id,
                            session.branch_manager.current_branch_id,
                            seq=resolved_index,
                            role=edited.get("role", "user"),
                            content=str(edited.get("content", "")),
                            created_at=float(edited.get("timestamp", time.time())),
                            metadata=None,
                        )
                        if updated_id:
                            try:
                                session.conversation_history[resolved_index]["id"] = updated_id
                            except Exception:
                                pass
                        logger.info(f"[CMD] edit persisted: conv={conv_id} branch={session.branch_manager.current_branch_id} idx={resolved_index} id={session.conversation_history[resolved_index].get('id')}")
                    # Keep branch using same list object
                    try:
                        current_branch = session.branch_manager.get_current_branch()
                        current_branch.conversation_history = session.conversation_history
                        from datetime import datetime as _dt
                        current_branch.last_active = _dt.now()
                    except Exception as sync_err:
                        logger.debug(f"edit branch sync failed: {sync_err}")
                except Exception as e:
                    logger.warning(f"[CMD] edit id alignment failed: {e}")

            # Persist regen’s assistant tail id when DB available
            if result.success and command == "regen" and self.db is not None:
                try:
                    conv_id = self.db.ensure_conversation(session_id)
                    self.db.ensure_branch(conv_id, session.branch_manager.current_branch_id, name=session.branch_manager.current_branch_id)
                    if session.conversation_history and session.conversation_history[-1].get("role") == "assistant":
                        last = session.conversation_history[-1]
                        new_db_id = self.db.append_message_to_branch(
                            conv_id,
                            session.branch_manager.current_branch_id,
                            role=last.get("role", "assistant"),
                            content=str(last.get("content", "")),
                            created_at=float(last.get("timestamp", time.time())),
                            metadata=(__import__('json').dumps(last.get("metadata", {})) if last.get("metadata") else None),
                            force_new=is_electron,
                        )
                        try:
                            session.conversation_history[-1]["id"] = new_db_id
                        except Exception:
                            pass
                        # Graph dual-write for message tail (message-id schema)
                        try:
                            from oumi.webchat.chatgraph_migration.graph_store import GraphStore as _GS
                            _GS(self.db.db_path).add_edge_for_message_tail(conv_id, new_db_id)
                        except Exception as _gge:
                            logger.warning(f"[Graph] message-tail dual-write (regen) failed: {_gge}")
                        logger.info(f"[CMD] regen persisted: conv={conv_id} branch={session.branch_manager.current_branch_id} idx={len(session.conversation_history)-1} id={new_db_id}")
                        response_data.setdefault("updated", {})
                        response_data["updated"].update({"message_id": new_db_id, "index": len(session.conversation_history) - 1, "content": last.get("content", "")})
                        response_data["broadcast"] = True
                except Exception as e:
                    logger.warning(f"[CMD] regen id alignment failed: {e}")

            # Broadcast conversation updates
            if command in ["clear", "delete", "regen", "edit"] and result.success:
                if command == "delete":
                    try:
                        current_branch = session.branch_manager.get_current_branch()
                        current_branch.conversation_history = session.conversation_history
                        from datetime import datetime as _dt
                        current_branch.last_active = _dt.now()
                    except Exception as sync_err:
                        logger.debug(f"delete branch sync failed: {sync_err}")
                logger.info(f"🌐 API: Broadcasting conversation update for command '{command}'")
                await session.broadcast_to_websockets({
                    "type": "conversation_update",
                    "conversation": session.serialize_conversation(),
                    "branches": session.get_enhanced_branch_info(self.db),
                    "current_branch": session.branch_manager.current_branch_id,
                    "timestamp": time.time(),
                })
                try:
                    clen = len(session.conversation_history)
                    logger.info(f"[CMD] broadcast conversation_update complete len={clen}")
                except Exception:
                    pass

            # Persist clear/delete by replacing branch mapping
            if self.db and command in ["clear", "delete"]:
                try:
                    conv_id = self.db.ensure_conversation(session_id)
                    self.db.replace_branch_history(
                        conv_id,
                        session.branch_manager.current_branch_id,
                        session.conversation_history,
                    )
                    logger.info(f"[CMD] {command} persisted via replace_branch_history: conv={conv_id} branch={session.branch_manager.current_branch_id} count={len(session.conversation_history)}")
                except Exception as pe:
                    logger.warning(f"⚠️ Dual-write persistence (command result) failed: {pe}")

            return web.json_response(response_data)

        except Exception as e:
            logger.error(f"API command execution error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return web.json_response({"error": f"Command failed: {str(e)}"}, status=500)
