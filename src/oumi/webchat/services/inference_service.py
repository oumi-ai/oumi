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

"""Inference service for WebChat server."""

import gc
import time
from typing import Any, Dict, List, Optional, Union

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.infer import infer, get_engine
from oumi.utils.logging import logger


class InferenceService:
    """Manages inference engines and model generation."""
    
    def __init__(self, default_config: InferenceConfig):
        """Initialize inference service.
        
        Args:
            default_config: Default inference configuration.
        """
        self.default_config = default_config
        self._default_engine = None
        self._engines_cache = {}  # Cache for inference engines keyed by config hash
    
    @property
    def default_engine(self):
        """Get or lazily initialize the default inference engine.
        
        Returns:
            Default inference engine.
        """
        if self._default_engine is None:
            logger.info("🔄 Initializing default inference engine...")
            self._default_engine = self._create_engine(self.default_config)
            logger.info("✅ Default inference engine initialized")
        return self._default_engine
    
    def _get_config_key(self, config: InferenceConfig) -> str:
        """Create a unique key for the config to use in engine cache.
        
        Args:
            config: Inference configuration.
            
        Returns:
            String key representing the config.
        """
        # Create a key based on model name and engine type
        model_name = getattr(config.model, "model_name", "unknown")
        engine_type = str(config.engine) if config.engine else "NATIVE"
        return f"{model_name}_{engine_type}"
    
    def _create_engine(self, config: InferenceConfig):
        """Create a new inference engine from config.
        
        Args:
            config: Inference configuration.
            
        Returns:
            Inference engine instance.
        """
        try:
            return build_inference_engine(
                engine_type=config.engine or InferenceEngineType.NATIVE,
                model_params=config.model,
                remote_params=config.remote_params,
                generation_params=config.generation,
            )
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    def get_engine(self, config: InferenceConfig = None):
        """Get or create an inference engine for the given config.
        
        Args:
            config: Inference configuration, or None for default.
            
        Returns:
            Inference engine instance.
        """
        if config is None:
            return self.default_engine
        
        config_key = self._get_config_key(config)
        if config_key in self._engines_cache:
            logger.info(f"🔄 Using cached engine for {config_key}")
            return self._engines_cache[config_key]
        
        logger.info(f"🔄 Creating new engine for {config_key}")
        engine = self._create_engine(config)
        self._engines_cache[config_key] = engine
        return engine
    
    def generate_response(
        self, 
        conversation: Conversation, 
        config: Optional[InferenceConfig] = None
    ) -> Conversation:
        """Generate a response for the given conversation.
        
        Args:
            conversation: Input conversation.
            config: Optional custom config to use.
            
        Returns:
            Conversation with model response.
        """
        config = config or self.default_config
        engine = self.get_engine(config)
        
        # Log input conversation for debugging
        logger.debug(f"Generating response for conversation with {len(conversation.messages)} messages")
        for i, msg in enumerate(conversation.messages[-5:]):  # Log last 5 messages
            logger.debug(f"Message {i}: {msg.role} - {str(msg.content)[:100]}...")
        
        # Generate response
        start_time = time.time()
        try:
            # Use the engine.infer method for consistency with other code
            result = engine.infer(
                input=[conversation],
                inference_config=config
            )
            elapsed = time.time() - start_time
            logger.info(f"✅ Generated response in {elapsed:.2f}s")
            
            # Return the last conversation
            if result:
                return result[-1] if isinstance(result, list) else result
            return conversation
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            
            # Fallback to unified infer function
            try:
                logger.info(f"🔄 Falling back to unified infer() function")
                last_user_message = ""
                for msg in reversed(conversation.messages):
                    if msg.role == Role.USER:
                        last_user_message = msg.content
                        break
                
                if not last_user_message:
                    raise ValueError("No user message found in conversation")
                
                results = infer(
                    config=config,
                    inputs=[last_user_message],
                    inference_engine=engine
                )
                
                if results and len(results) > 0:
                    return results[0]
                else:
                    raise ValueError("No results from infer()")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                # Create a response with an error message
                error_msg = f"Failed to generate response: {str(e)}"
                result_conversation = Conversation(messages=list(conversation.messages))
                result_conversation.messages.append(Message(role=Role.ASSISTANT, content=error_msg))
                return result_conversation
    
    def clear_engine(self, config: Optional[InferenceConfig] = None):
        """Clear a specific inference engine from memory.
        
        Args:
            config: Configuration of the engine to clear, or None for default.
        """
        if config is None:
            if self._default_engine is not None:
                try:
                    if hasattr(self._default_engine, "dispose"):
                        self._default_engine.dispose()
                    elif hasattr(self._default_engine, "close"):
                        self._default_engine.close()
                except Exception as e:
                    logger.warning(f"Error disposing default engine: {e}")
                finally:
                    self._default_engine = None
            return
        
        config_key = self._get_config_key(config)
        if config_key in self._engines_cache:
            engine = self._engines_cache.pop(config_key)
            try:
                if hasattr(engine, "dispose"):
                    engine.dispose()
                elif hasattr(engine, "close"):
                    engine.close()
            except Exception as e:
                logger.warning(f"Error disposing engine {config_key}: {e}")
    
    def clear_all_engines(self):
        """Clear all inference engines from memory."""
        # Clear default engine
        if self._default_engine is not None:
            try:
                if hasattr(self._default_engine, "dispose"):
                    self._default_engine.dispose()
                elif hasattr(self._default_engine, "close"):
                    self._default_engine.close()
            except Exception as e:
                logger.warning(f"Error disposing default engine: {e}")
            finally:
                self._default_engine = None
        
        # Clear all cached engines
        for config_key, engine in list(self._engines_cache.items()):
            try:
                if hasattr(engine, "dispose"):
                    engine.dispose()
                elif hasattr(engine, "close"):
                    engine.close()
            except Exception as e:
                logger.warning(f"Error disposing engine {config_key}: {e}")
        
        self._engines_cache.clear()
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ CUDA cache cleared")
        except ImportError:
            logger.debug("PyTorch not available, skipping CUDA cache clear")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")
        
        logger.info("✅ All inference engines cleared")