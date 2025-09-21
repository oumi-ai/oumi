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
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        session_id = data.get("session_id", "default")
        command = data.get("command", "")
        args = data.get("args", [])
        # Optional id-first fields
        branch_id = data.get("branch_id")
        message_id = data.get("message_id")
        index_hint = data.get("index")
        payload = data.get("payload")  # e.g., new content for edit
        
        logger.info(f"🌐 API: Received command '{command}' with args: {args} for session: {session_id}")
        
        session = await self.session_manager.get_or_create_session_safe(session_id, self.db)

        # Optionally switch branch to ensure cache alignment
        if branch_id and branch_id != session.branch_manager.current_branch_id:
            async def _switch_branch(s):
                try:
                    s.branch_manager.sync_conversation_history(s.conversation_history)
                    ok, msg, br = s.branch_manager.switch_branch(branch_id)
                    if ok and br:
                        s.conversation_history.clear()
                        s.conversation_history.extend(br.conversation_history)
                        logger.debug(f"[CMD] switched branch -> {branch_id}")
                except Exception as e:
                    logger.warning(f"[CMD] branch switch failed: {e}")
                return s
            session = await self.session_manager.execute_session_operation(session_id, _switch_branch)

        def _normalize_target():
            """Resolve message target via id-first, with index fallback.

            Returns (resolved_id, resolved_index, diag_dict)
            """
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
            diag = {
                "requested_index": index_hint,
                "requested_message_id": message_id,
                "resolved_index": resolved_index,
                "resolved_message_id": resolved_id,
                "msg_count": len(convo),
                "ids_first_last": (convo[0].get("id") if convo else None, convo[-1].get("id") if convo else None),
            }
            # Log inconsistencies
            if message_id and resolved_id is None:
                logger.warning(f"[ID_MISS] message_id='{message_id}' not found | {diag}")
            if (index_hint is not None and resolved_index is None) or (
                isinstance(index_hint, int) and not (0 <= index_hint < len(convo))
            ):
                logger.warning(f"[IDX_OOB] index='{index_hint}' out of range | {diag}")
            logger.debug(f"[RESOLVE] {diag}")
            return resolved_id, resolved_index, diag
        
        try:
            # Create a string buffer to capture console output
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, width=80)
            
            # Temporarily replace the session's console
            original_console = session.command_context.console
            session.command_context.console = temp_console
            
            try:
                # If id/index provided for id-first commands, adapt args
                id_first_cmds = {"delete", "regen", "edit"}
                resolved_id = None
                resolved_index = None
                if command in id_first_cmds:
                    resolved_id, resolved_index, _diag = _normalize_target()
                    if resolved_index is not None:
                        # For delete/regen/edit, ensure index is first arg
                        if command in {"delete", "regen"}:
                            args = [str(resolved_index)]
                        elif command == "edit":
                            # edit requires new content payload
                            new_content = payload if payload is not None else (args[1] if len(args) > 1 else "")
                            args = [str(resolved_index), new_content]
                        logger.info(f"[CMD] normalized {command} -> index={resolved_index}, id={resolved_id}")
                    else:
                        logger.warning(f"[CMD] Could not resolve target for {command}; proceeding with original args={args}")

                # Execute command via command router
                parsed_command = ParsedCommand(
                    command=command,
                    args=args,
                    kwargs={},
                    raw_input=f"/{command}({','.join(args)})",
                )
                logger.info(f"🌐 API: Executing command '{command}' via command router...")
                result = session.command_router.handle_command(parsed_command)
                logger.info(f"🌐 API: Command '{command}' result: success={result.success}, message='{result.message}', should_continue={result.should_continue}")
                
                # Capture the console output
                console_output = string_buffer.getvalue()
                
                # Combine command result message with console output
                full_message = result.message or ""
                if console_output.strip():
                    if full_message:
                        full_message += "\n\n" + console_output.strip()
                    else:
                        full_message = console_output.strip()
            
            finally:
                # Restore the original console
                session.command_context.console = original_console
            
            response_data = {
                "success": result.success,
                "message": full_message,
                "should_continue": result.should_continue,
            }
            # Attach optional structured updates for clients
            if command in {"delete", "regen", "edit"}:
                resolved_id, resolved_index, _ = _normalize_target()
                response_data["target"] = {
                    "message_id": resolved_id,
                    "index": resolved_index,
                }
                if command == "edit" and result.success and payload is not None and resolved_index is not None:
                    response_data["updated"] = {
                        "message_id": resolved_id,
                        "index": resolved_index,
                        "content": payload,
                    }
                    response_data["broadcast"] = True
            
            # Add specific data for certain commands
            if command in ["branches", "list_branches"]:
                response_data["branches"] = session.get_enhanced_branch_info(self.db)
                response_data["current_branch"] = (
                    session.branch_manager.current_branch_id
                )
            
            elif command == "show":
                response_data["conversation"] = session.serialize_conversation()
            
            elif command == "swap" and result.success:
                # CRITICAL FIX: Update model_info when model swap is successful
                try:
                    if hasattr(session.command_context, 'config') and session.command_context.config:
                        new_config = session.command_context.config
                        # Include the swapped model information in the response
                        response_data["model_info"] = {
                            "name": getattr(new_config.model, "model_name", "oumi-model"),
                            "engine": str(new_config.engine)
                        }
                        logger.info(f"✅ Included swapped model info in response: {response_data['model_info']['name']}")
                    else:
                        logger.warning("⚠️ Cannot include model_info: session config not available")
                except Exception as e:
                    logger.error(f"❌ Error adding model_info to response: {e}")
            
            # Handle commands that require follow-up inference (e.g., regen)
            if result.success and result.should_continue and hasattr(result, 'user_input_override') and result.user_input_override:
                logger.info(f"🌐 API: Command '{command}' requires follow-up inference with user_input_override")
                logger.info(f"🌐 API: User input override: {result.user_input_override[:100]}...")
                
                # For regeneration commands, perform inference directly
                is_regeneration = getattr(result, 'is_regeneration', False)
                if is_regeneration:
                    try:
                        # Perform inference using the user_input_override (edited content)
                        await session._perform_regeneration_inference(result.user_input_override)
                    except Exception as e:
                        logger.error(f"❌ Error during regeneration inference: {e}")
                        await session.broadcast_to_websockets(
                            {"type": "error", "message": f"Regeneration failed: {str(e)}", "timestamp": time.time()}
                        )
            
            # If edit succeeded, align the edited message's ID with DB canonical id
            if (
                result.success and command == "edit" and self.db is not None and resolved_index is not None
            ):
                try:
                    conv_id = self.db.ensure_conversation(session_id)
                    # Ensure branch exists
                    self.db.ensure_branch(
                        conv_id, session.branch_manager.current_branch_id, name=session.branch_manager.current_branch_id
                    )
                    # Persist only the edited message and capture DB id
                    edited = session.conversation_history[resolved_index]
                    new_db_id = self.db.append_message_to_branch(
                        conv_id,
                        session.branch_manager.current_branch_id,
                        role=edited.get("role", "user"),
                        content=str(edited.get("content", "")),
                        created_at=float(edited.get("timestamp", time.time())),
                    )
                    # Overwrite in-memory id with canonical DB id
                    try:
                        session.conversation_history[resolved_index]["id"] = new_db_id
                    except Exception:
                        pass
                    # Attach to response payload for the client to update mapping
                    response_data.setdefault("updated", {})
                    response_data["updated"].update(
                        {
                            "message_id": new_db_id,
                            "index": resolved_index,
                            "content": edited.get("content", ""),
                        }
                    )
                    response_data["broadcast"] = True
                except Exception as e:
                    logger.warning(f"[CMD] edit id alignment failed: {e}")

            # Broadcast conversation updates for commands that modify state
            if command in ["clear", "delete", "regen", "edit"] and result.success:
                logger.info(f"🌐 API: Broadcasting conversation update for command '{command}'")
                await session.broadcast_to_websockets(
                    {
                        "type": "conversation_update",
                        "conversation": session.serialize_conversation(),
                        "branches": session.get_enhanced_branch_info(self.db),
                        "current_branch": session.branch_manager.current_branch_id,
                        "timestamp": time.time()
                    }
                )
            
            # Save command results to persistence if available and supported
            # Avoid bulk-add for edit to prevent duplication; targeted persist above.
            if self.db and session.is_hydrated_from_db and command in ["clear", "delete", "regen"]:
                try:
                    # If the command modified conversation history, sync to database
                    conv_id = self.db.ensure_conversation(session_id)
                    self.db.bulk_add_branch_history(
                        conv_id, 
                        session.branch_manager.current_branch_id, 
                        session.conversation_history
                    )
                except Exception as pe:
                    logger.warning(f"⚠️ Dual-write persistence (command result) failed: {pe}")
            
            return web.json_response(response_data)
        
        except Exception as e:
            logger.error(f"API command execution error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return web.json_response({"error": f"Command failed: {str(e)}"}, status=500)
