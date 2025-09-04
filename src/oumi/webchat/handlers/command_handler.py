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
        
        logger.info(f"🌐 API: Received command '{command}' with args: {args} for session: {session_id}")
        
        session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
        
        try:
            # Create a string buffer to capture console output
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, width=80)
            
            # Temporarily replace the session's console
            original_console = session.command_context.console
            session.command_context.console = temp_console
            
            try:
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
            if self.db and session.is_hydrated_from_db and command in ["clear", "delete", "regen", "edit"]:
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