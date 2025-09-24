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

"""System information and health endpoints for Oumi WebChat server."""

import gc
import time
from typing import Dict, Optional, Any

from aiohttp import web

from oumi.utils.logging import logger
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.utils.fallbacks import model_name_fallback


class SystemHandler:
    """Handles system information endpoints for Oumi WebChat."""
    
    def __init__(
        self, 
        session_manager: SessionManager,
        enhanced_features_available: bool = False
    ):
        """Initialize system handler.
        
        Args:
            session_manager: Session manager for WebChat sessions
            enhanced_features_available: Whether enhanced API features are available
        """
        self.session_manager = session_manager
        self.enhanced_features_available = enhanced_features_available
        self.response_formatter = None
        
        # Initialize enhanced response formatter if available
        if enhanced_features_available:
            try:
                from oumi.webchat.api_responses import ResponseFormatter
                self.response_formatter = ResponseFormatter()
            except ImportError:
                logger.warning("Enhanced response formatter could not be imported")
        
        # Model info for /v1/models endpoint
        model_id = getattr(session_manager.default_config.model, "model_name", None)
        if not model_id:
            model_id = model_name_fallback("default_config.model.model_name")
            logger.warning(f"Default config model_name missing; using fallback '{model_id}'.")
        self.model_info = {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "oumi",
        }
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint.
        
        Args:
            request: Web request
            
        Returns:
            JSON response with health status
        """
        logger.debug("🔍 Health endpoint called!")
        
        if self.enhanced_features_available and self.response_formatter:
            from oumi.webchat.api_responses import create_json_response
            return create_json_response(
                self.response_formatter.success({
                    "status": "healthy",
                    "server": "oumi-webchat",
                    "version": "1.0.0",
                    "features": ["websocket", "sse", "chat", "branches", "commands"],
                    "enhanced_features": True,
                    "timestamp": time.time(),
                    "session_count": len(self.session_manager.sessions)
                })
            )
        else:
            return web.json_response({
                "status": "healthy",
                "server": "oumi-webchat",
                "version": "1.0.0",
                "features": ["websocket", "chat", "branches", "commands"],
                "enhanced_features": False,
                "timestamp": time.time(),
                "session_count": len(self.session_manager.sessions)
            })
    
    async def handle_models(self, request: web.Request) -> web.Response:
        """List available models endpoint with enhanced config metadata.
        
        Args:
            request: Web request with optional session_id
            
        Returns:
            JSON response with model information
        """
        # Get session-specific model info if available
        session_id = request.query.get("session_id", "default")
        
        try:
            session = await self.session_manager.get_or_create_session_safe(session_id)
            
            # Check if session has a swapped model
            if (hasattr(session.command_context, 'config') and 
                session.command_context.config is not None):
                active_config = session.command_context.config
                logger.debug(f"🔄 Using session's swapped config for /v1/models")
            else:
                active_config = session.config
                logger.debug(f"🔄 Using session's initial config for /v1/models")
            
            # Extract complete model metadata from active config
            model_name = getattr(active_config.model, "model_name", None)
            if not model_name:
                model_name = model_name_fallback("active_config.model.model_name")
                logger.warning(f"Active config model_name missing; using fallback '{model_name}'.")
            engine = str(active_config.engine) if active_config.engine else "NATIVE"
            context_length = getattr(active_config.model, "model_max_length", 4096)
            
            # Create enhanced model info with config metadata
            is_omni = self._is_omni_model(model_name)

            enhanced_model_info = {
                "id": model_name,
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "oumi",
                # Add config metadata for frontend display
                "config_metadata": {
                    "model_name": model_name,
                    "engine": engine,
                    "context_length": context_length,
                    "display_name": model_name.split('/')[-1] if '/' in model_name else model_name,
                    "description": f"{model_name} ({engine} engine)",
                    "model_family": self._extract_model_family(model_name),
                    "is_active_config": True,  # Flag to indicate this is the active config
                    "is_omni_capable": is_omni,
                }
            }
            
            logger.debug(f"📋 Returning enhanced model info: {enhanced_model_info['config_metadata']['display_name']}")
            return web.json_response({"object": "list", "data": [enhanced_model_info]})
            
        except Exception as e:
            logger.error(f"Error getting enhanced model info: {e}")
            # Fallback to basic model info
            fallback_info = dict(self.model_info)
            fallback_metadata = dict(fallback_info.get("config_metadata", {}))
            if "is_omni_capable" not in fallback_metadata:
                fallback_metadata["is_omni_capable"] = self._is_omni_model(fallback_info.get("id", ""))
            fallback_info["config_metadata"] = fallback_metadata
            return web.json_response({"object": "list", "data": [fallback_info]})
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name for UI categorization.
        
        Args:
            model_name: Full model name to extract family from
            
        Returns:
            Model family identifier
        """
        model_lower = model_name.lower()
        
        # Extract family based on common patterns
        if 'llama' in model_lower:
            if '3.1' in model_lower:
                return 'llama3_1'
            elif '3.2' in model_lower:
                return 'llama3_2' 
            elif '3.3' in model_lower:
                return 'llama3_3'
            elif '4' in model_lower:
                return 'llama4'
            else:
                return 'llama'
        elif 'qwen' in model_lower:
            if '2.5' in model_lower:
                return 'qwen2_5'
            elif '3' in model_lower:
                return 'qwen3'
            else:
                return 'qwen'
        elif 'gemma' in model_lower:
            return 'gemma3'
        elif 'phi' in model_lower:
            if '4' in model_lower:
                return 'phi4'
            else:
                return 'phi3'
        elif 'deepseek' in model_lower:
            return 'deepseek_r1'
        elif 'gpt' in model_lower:
            return 'gpt_oss'
        else:
            return 'unknown'

    def _is_omni_model(self, model_name: str) -> bool:
        model_lower = (model_name or "").lower()
        return 'omni' in model_lower and 'qwen' in model_lower
    
    async def handle_system_stats_api(self, request: web.Request) -> web.Response:
        """Handle getting system stats from backend session.
        
        Args:
            request: Web request with session_id
            
        Returns:
            JSON response with system stats
        """
        # Trace id
        try:
            trace_id = request.get('trace_id') or request.headers.get('X-Trace-ID')
        except Exception:
            trace_id = None
        session_id = request.query.get("session_id")
        if not session_id:
            payload = {"error": "session_id is required"}
            if trace_id:
                payload["trace_id"] = trace_id
            return web.json_response(payload, status=400)
        
        session = await self.session_manager.get_or_create_session_safe(session_id)
        
        # Update context usage based on current conversation
        self.session_manager.update_context_usage(session_id)
        
        # Get stats from SystemMonitor
        stats = session.system_monitor.get_stats()
        
        # Get session manager metrics
        session_metrics = self.session_manager.get_session_metrics()
        
        response_data = {
            # System stats
            "cpu_percent": stats.cpu_percent,
            "ram_used_gb": stats.ram_used_gb,
            "ram_total_gb": stats.ram_total_gb,
            "ram_percent": stats.ram_percent,
            "gpu_vram_used_gb": stats.gpu_vram_used_gb,
            "gpu_vram_total_gb": stats.gpu_vram_total_gb,
            "gpu_vram_percent": stats.gpu_vram_percent,
            
            # Context usage
            "context_used_tokens": stats.context_used_tokens,
            "context_max_tokens": stats.context_max_tokens,
            "context_percent": stats.context_percent,
            "conversation_turns": stats.conversation_turns,
            
            # Session metrics
            "active_sessions": session_metrics["active_sessions"],
            "avg_wait_time": session_metrics["avg_wait_time"],
            "max_wait_time": session_metrics["max_wait_time"],
            "contention_count": session_metrics["contention_count"]
        }
        if trace_id:
            response_data["trace_id"] = trace_id
        
        logger.debug(f"System stats for session {session_id}: {stats.context_used_tokens}/{stats.context_max_tokens} tokens, {stats.conversation_turns} turns")
        
        return web.json_response(response_data)
    
    async def handle_clear_model_api(self, request: web.Request) -> web.Response:
        """Handle clearing/unloading the current model from memory.
        
        Args:
            request: Web request with session_id
            
        Returns:
            JSON response with clearing operation result
        """
        # Trace id
        try:
            trace_id = request.get('trace_id') or request.headers.get('X-Trace-ID')
        except Exception:
            trace_id = None
        try:
            session_id = request.query.get("session_id", "default")
            session = await self.session_manager.get_or_create_session_safe(session_id)
            
            logger.info(f"[trace:{trace_id}] 🧹 Clearing model from memory for session {session_id}")
            
            # Clear the inference engine if it exists
            if hasattr(session, 'inference_engine') and session.inference_engine is not None:
                # Call dispose method if available
                if hasattr(session.inference_engine, 'dispose'):
                    session.inference_engine.dispose()
                elif hasattr(session.inference_engine, 'close'):
                    session.inference_engine.close()
                
                # Clear the engine reference
                session.inference_engine = None
                logger.info("✅ Inference engine cleared")
            
            # Clear the command context engine if it exists
            if hasattr(session.command_context, 'inference_engine') and session.command_context.inference_engine is not None:
                # Call dispose method if available
                if hasattr(session.command_context.inference_engine, 'dispose'):
                    session.command_context.inference_engine.dispose()
                elif hasattr(session.command_context.inference_engine, 'close'):
                    session.command_context.inference_engine.close()
                
                # Clear the engine reference
                session.command_context.inference_engine = None
                logger.info("✅ Command context inference engine cleared")
            
            # Force garbage collection and CUDA cache clearing
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("✅ CUDA cache cleared")
            except ImportError:
                logger.debug("PyTorch not available, skipping CUDA cache clear")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")
            
            logger.info("✅ Model clearing completed successfully")
            
            resp = {"success": True, "message": "Model cleared from memory successfully"}
            if trace_id:
                resp["trace_id"] = trace_id
            return web.json_response(resp)
            
        except Exception as e:
            logger.error(f"[trace:{trace_id}] ❌ Error clearing model: {e}")
            payload = {"success": False, "error": f"Failed to clear model: {str(e)}"}
            if trace_id:
                payload["trace_id"] = trace_id
            return web.json_response(payload, status=500)
