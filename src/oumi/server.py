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

import time
from typing import Optional

from aiohttp import web
from aiohttp.web import Response

from oumi.core.configs import InferenceConfig
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.infer import get_engine, infer
from oumi.utils.logging import logger
from oumi.webchat.utils.fallbacks import model_name_fallback


class OpenAICompatibleServer:
    """HTTP server that implements OpenAI-compatible API endpoints for Oumi inference."""

    def __init__(
        self,
        config: InferenceConfig,
        system_prompt: Optional[str] = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.inference_engine: BaseInferenceEngine = get_engine(config)

        # Model info for /v1/models endpoint
        model_id = getattr(config.model, "model_name", None)
        if not model_id:
            model_id = model_name_fallback("config.model.model_name")
            logger.warning(f"Model name missing on config.model; using fallback '{model_id}'.")
        # Compute basic capability flags for frontend gating (best-effort)
        try:
            import re
            is_omni = bool(re.search(r"qwen\s*/?qwen(2\\.5|3).*omni", str(model_id), re.IGNORECASE))
        except Exception:
            is_omni = False

        self.model_info = {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "oumi",
            "config_metadata": {
                "is_omni_capable": is_omni,
            },
        }

    async def handle_health(self, request: web.Request) -> Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def handle_models(self, request: web.Request) -> Response:
        """List available models endpoint."""
        return web.json_response({"object": "list", "data": [self.model_info]})

    async def handle_chat_completions(self, request: web.Request) -> Response:
        """Handle chat completions requests in OpenAI format."""
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {
                    "error": {
                        "message": f"Invalid JSON: {str(e)}",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )

        # Extract required fields
        messages = data.get("messages", [])
        if not messages:
            return web.json_response(
                {
                    "error": {
                        "message": "messages field is required",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )

        # Extract optional fields
        model = data.get("model", self.model_info["id"])
        temperature = data.get("temperature", 1.0)
        max_tokens = data.get("max_tokens", 100)
        stream = data.get("stream", False)

        try:
            # Convert OpenAI format messages to Oumi conversation format
            oumi_messages = []

            # Add system prompt if provided
            if self.system_prompt:
                oumi_messages.append(
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                )

            # Convert messages
            for msg in messages:
                role_mapping = {
                    "system": Role.SYSTEM,
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                }
                role = role_mapping.get(msg.get("role"), Role.USER)
                content = msg.get("content", "")
                oumi_messages.append(Message(role=role, content=content))

            # Get the latest user message for inference
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                return web.json_response(
                    {
                        "error": {
                            "message": "No user message found",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )

            latest_user_content = user_messages[-1].get("content", "")

            # Run inference. If the latest user content is multimodal (list of parts),
            # build a full conversation and invoke the engine directly.
            if isinstance(latest_user_content, list):
                conversation = Conversation(messages=oumi_messages)
                results = self.inference_engine.infer(
                    input=[conversation], inference_config=self.config
                )
            else:
                results = infer(
                    config=self.config,
                    inputs=[latest_user_content],
                    system_prompt=self.system_prompt,
                    inference_engine=self.inference_engine,
                )

            if not results:
                return web.json_response(
                    {
                        "error": {
                            "message": "No response generated",
                            "type": "server_error",
                        }
                    },
                    status=500,
                )

            # Extract response content
            response_content = ""
            conversation = results[0]  # Take first result
            for message in conversation.messages:
                # Skip user messages and system messages, only get assistant responses
                if message.role not in [Role.USER, Role.SYSTEM]:
                    if isinstance(message.content, str):
                        response_content = message.content
                        break
                    elif isinstance(message.content, list):
                        for item in message.content:
                            if hasattr(item, "content") and item.content:
                                response_content = str(item.content)
                                break

            if not response_content:
                response_content = str(conversation)

            # Format response in OpenAI format
            response_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(latest_user_content.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(latest_user_content.split())
                    + len(response_content.split()),
                },
            }

            # Handle streaming vs non-streaming
            if stream:
                # For now, just return non-streaming response
                # TODO: Implement proper streaming
                return web.json_response(response_data)
            else:
                return web.json_response(response_data)

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return web.json_response(
                {
                    "error": {
                        "message": f"Inference failed: {str(e)}",
                        "type": "server_error",
                    }
                },
                status=500,
            )

    def create_app(self) -> web.Application:
        """Create and configure the aiohttp application."""
        app = web.Application()

        # Add routes
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)

        # Add CORS headers for web clients
        async def add_cors_headers(request, handler):
            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            return response

        app.middlewares.append(add_cors_headers)

        return app


def run_server(
    config: InferenceConfig,
    host: str = "0.0.0.0",
    port: int = 9000,
    system_prompt: Optional[str] = None,
) -> None:
    """Run the OpenAI-compatible HTTP server."""
    server = OpenAICompatibleServer(config, system_prompt)
    app = server.create_app()

    logger.info("🚀 Starting Oumi inference server")
    logger.info(f"📍 Server URL: http://{host}:{port}")
    logger.info("🔗 OpenAI-compatible endpoints:")
    logger.info(f"   • Models: http://{host}:{port}/v1/models")
    logger.info(f"   • Chat: http://{host}:{port}/v1/chat/completions")
    logger.info(
        f"💡 Use with any OpenAI-compatible client by setting base_url to http://{host}:{port}/v1"
    )
    logger.info("🛑 Press Ctrl+C to stop")

    # Run the server
    web.run_app(app, host=host, port=port, access_log=logger)
