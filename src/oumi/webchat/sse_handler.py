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

"""Server-Sent Events (SSE) handler for real-time streaming responses."""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Optional

from aiohttp import web
from aiohttp.web_response import StreamResponse

from oumi.utils.logging import logger


class SSEHandler:
    """Handles Server-Sent Events for streaming responses."""

    def __init__(self):
        """Initialize SSE handler."""
        self.active_streams: Dict[str, StreamResponse] = {}

    async def create_sse_response(self, request: web.Request) -> StreamResponse:
        """Create a Server-Sent Events response."""
        response = StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )
        await response.prepare(request)
        return response

    async def send_sse_event(
        self,
        response: StreamResponse,
        event_type: str,
        data: Dict,
        event_id: Optional[str] = None,
    ) -> bool:
        """Send an SSE event to the client.
        
        Args:
            response: The SSE response stream
            event_type: Type of event (e.g., 'message', 'error', 'done')
            data: Event data as dictionary
            event_id: Optional event ID
            
        Returns:
            True if event was sent successfully, False if connection was closed
        """
        try:
            # Format SSE message
            sse_data = f"event: {event_type}\n"
            if event_id:
                sse_data += f"id: {event_id}\n"
            sse_data += f"data: {json.dumps(data)}\n\n"

            # Send to client
            await response.write(sse_data.encode("utf-8"))
            return True
        except ConnectionResetError:
            logger.debug("SSE connection closed by client")
            return False
        except Exception as e:
            logger.error(f"Error sending SSE event: {e}")
            return False

    async def stream_chat_response(
        self,
        response: StreamResponse,
        conversation,
        inference_engine,
        session_id: str,
    ) -> AsyncGenerator[Dict, None]:
        """Stream chat response chunks via SSE.
        
        Args:
            response: SSE response stream
            conversation: Conversation object for inference
            inference_engine: Engine to generate response
            session_id: Session identifier
            
        Yields:
            Dictionary with response chunks
        """
        try:
            # Send start event
            await self.send_sse_event(
                response,
                "start",
                {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "message": "Starting generation...",
                },
            )

            # Generate response (this would need to be adapted for streaming)
            result = inference_engine.generate_response(conversation)

            if result and len(result.messages) > 0:
                last_message = result.messages[-1]
                content = ""
                
                if isinstance(last_message.content, str):
                    content = last_message.content
                elif isinstance(last_message.content, list):
                    text_parts = []
                    for item in last_message.content:
                        if hasattr(item, "content") and item.content:
                            text_parts.append(str(item.content))
                    content = " ".join(text_parts)

                # For now, send the complete response
                # TODO: Implement proper streaming from inference engine
                await self.send_sse_event(
                    response,
                    "chunk",
                    {
                        "content": content,
                        "timestamp": time.time(),
                        "is_final": True,
                    },
                )

                yield {
                    "content": content,
                    "role": "assistant",
                    "timestamp": time.time(),
                }

            # Send completion event
            await self.send_sse_event(
                response,
                "done",
                {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "message": "Generation complete",
                },
            )

        except Exception as e:
            logger.error(f"Error in SSE streaming: {e}")
            await self.send_sse_event(
                response,
                "error",
                {
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )
            raise

    async def handle_sse_chat(self, request: web.Request) -> StreamResponse:
        """Handle SSE chat completion requests."""
        try:
            # Parse request data (from query params or body)
            session_id = request.query.get("session_id", "default")
            message = request.query.get("message", "")

            if not message:
                # Try to get from POST body if not in query
                try:
                    body = await request.json()
                    message = body.get("message", "")
                    session_id = body.get("session_id", session_id)
                except:
                    pass

            if not message:
                raise ValueError("Message is required")

            response = await self.create_sse_response(request)
            
            # Store active stream
            stream_id = f"{session_id}_{time.time()}"
            self.active_streams[stream_id] = response

            try:
                # Send keep-alive events periodically
                async def keep_alive():
                    while stream_id in self.active_streams:
                        await asyncio.sleep(30)  # Send every 30 seconds
                        if stream_id in self.active_streams:
                            await self.send_sse_event(
                                response,
                                "ping",
                                {"timestamp": time.time()}
                            )

                keep_alive_task = asyncio.create_task(keep_alive())

                # For now, just send a simple response
                # TODO: Integrate with actual session and inference engine
                await self.send_sse_event(
                    response,
                    "message",
                    {
                        "content": f"Echo: {message}",
                        "role": "assistant",
                        "timestamp": time.time(),
                    },
                )

                # Keep connection alive for a bit
                await asyncio.sleep(1)

            finally:
                # Cleanup
                keep_alive_task.cancel()
                self.active_streams.pop(stream_id, None)

            return response

        except Exception as e:
            logger.error(f"SSE chat error: {e}")
            response = await self.create_sse_response(request)
            await self.send_sse_event(
                response,
                "error",
                {"error": str(e), "timestamp": time.time()},
            )
            return response

    async def handle_sse_events(self, request: web.Request) -> StreamResponse:
        """Handle general SSE events endpoint."""
        session_id = request.query.get("session_id", "default")
        response = await self.create_sse_response(request)

        # Store active stream
        stream_id = f"events_{session_id}_{time.time()}"
        self.active_streams[stream_id] = response

        try:
            # Send initial connection event
            await self.send_sse_event(
                response,
                "connected",
                {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "message": "SSE connection established",
                },
            )

            # Keep connection alive
            while stream_id in self.active_streams:
                await asyncio.sleep(30)
                if stream_id in self.active_streams:
                    await self.send_sse_event(
                        response,
                        "heartbeat",
                        {"timestamp": time.time()}
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE events error: {e}")
        finally:
            self.active_streams.pop(stream_id, None)

        return response

    def close_stream(self, stream_id: str):
        """Close a specific SSE stream."""
        if stream_id in self.active_streams:
            stream = self.active_streams.pop(stream_id)
            # The stream will be closed when the coroutine ends

    def close_all_streams(self):
        """Close all active SSE streams."""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)