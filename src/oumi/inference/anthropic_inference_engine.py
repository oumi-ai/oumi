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

import copy
import itertools
import json
import re
from typing import Any

import pydantic
from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    FinishReason,
    Message,
    Role,
    ToolCall,
    Type,
)
from oumi.core.types.tool_call import ToolDefinition
from oumi.inference.remote_inference_engine import (
    BatchInfo,
    BatchListResponse,
    BatchResult,
    BatchStatus,
    RemoteInferenceEngine,
)
from oumi.utils.conversation_utils import (
    base64encode_content_item_image_bytes,
    load_image_bytes_to_content_item,
)
from oumi.utils.logging import logger

_CONTENT_KEY: str = "content"

# We are only constraining to ToolChoiceOptions in OpenAI
# https://developers.openai.com/api/reference/resources/responses/methods/create#(resource)%20responses%20%3E%20(method)%20create%20%3E%20(params)%200.non_streaming%20%3E%20(param)%20tool_choice%20%3E%20(schema)
_OPENAI_TO_ANTHROPIC_TOOL_CHOICE: dict[str, dict[str, str]] = {
    "auto": {"type": "auto"},
    "required": {"type": "any"},
    "none": {"type": "none"},
}

# Anthropic accepts image/jpeg, image/png, image/gif, image/webp for `image`
# content blocks. https://docs.anthropic.com/en/docs/build-with-claude/vision
_IMAGE_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _detect_image_media_type(data: bytes) -> str:
    """Sniffs an Anthropic-supported image media type from magic bytes.

    Falls back to ``image/png`` for unrecognized formats; Anthropic will reject
    a mismatched media_type, surfacing the issue rather than silently corrupting
    the image.
    """
    for magic, mime in _IMAGE_MAGIC_PREFIXES:
        if data.startswith(magic):
            return mime
    # WebP: "RIFF" + 4-byte size + "WEBP".
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    logger.warning(
        "Unrecognized image format; defaulting media_type to image/png. "
        "Anthropic accepts image/jpeg, image/png, image/gif, image/webp."
    )
    return "image/png"


class AnthropicInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Anthropic API.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with Anthropic's language models via their API. It handles
    the conversion of Oumi's Conversation objects to Anthropic's expected input
    format, as well as parsing the API responses back into Conversation objects.
    """

    anthropic_version = "2023-06-01"
    """The version of the Anthropic API to use.

    For more information on Anthropic API versioning, see:
    https://docs.anthropic.com/claude/reference/versioning
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the Anthropic API."""
        return "https://api.anthropic.com/v1/messages"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the Anthropic API key."""
        return "ANTHROPIC_API_KEY"

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an Anthropic API input.

        This method transforms an Oumi Conversation object into a format
        suitable for the Anthropic API. It handles system messages separately
        and structures the conversation history as required by Anthropic.

        See https://docs.anthropic.com/claude/reference/messages_post for details.

        Args:
            conversation: The Oumi Conversation object to convert.
            generation_params: Parameters for text generation.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for the
            Anthropic API, including the model, messages, and generation parameters.
        """
        # Anthropic API expects a top level `system` message,
        # Extract and exclude system message from the list of messages
        # in the conversation
        system_messages = [
            message for message in conversation.messages if message.role == Role.SYSTEM
        ]

        if len(system_messages) > 0:
            system_message = system_messages[0].content

            if len(system_messages) > 1:
                logger.warning(
                    "Multiple system messages found in conversation. "
                    "Only using the first one."
                )
        else:
            system_message = None

        messages = [
            message for message in conversation.messages if message.role != Role.SYSTEM
        ]

        # Build request body
        # See https://docs.anthropic.com/claude/reference/messages_post
        body: dict[str, Any] = {
            "model": model_params.model_name,
            "messages": self._messages_to_anthropic_blocks(messages),
            "max_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
        }

        # Only include top_p if it's explicitly set (Sonnet 4.5 requires only one of
        # temperature or top_p to be set, not both)
        if generation_params.top_p is not None:
            body["top_p"] = generation_params.top_p

        if system_message:
            body["system"] = system_message

        if generation_params.stop_strings is not None:
            body["stop_sequences"] = generation_params.stop_strings

        if generation_params.guided_decoding:
            if _model_supports_output_config(model_params.model_name):
                body.update(
                    _convert_guided_decoding_config_to_api_input(
                        generation_params.guided_decoding
                    )
                )
            else:
                logger.warning(
                    f"{model_params.model_name!r} does not support structured outputs"
                )

        # Enable prompt caching. Anthropic automatically caches content up to
        # the last cacheable block. This reduces latency and cost for repeated
        # prefixes (system prompts, long context, multi-turn conversations).
        # See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        body["cache_control"] = {"type": "ephemeral"}

        tool_choice = generation_params.tool_choice
        if conversation.tools:
            body["tools"] = self._openai_tools_to_anthropic(conversation.tools)
            if tool_choice is not None:
                body["tool_choice"] = self._translate_tool_choice(tool_choice)

        return body

    @staticmethod
    def _openai_tools_to_anthropic(
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Translates OpenAI-format tool definitions to Anthropic shape.

        Anthropic uses ``input_schema`` instead of ``parameters`` and lifts
        ``name``, ``description`` and ``input_schema`` to the top level (no
        ``function`` wrapper).
        """
        anthropic_tools: list[dict[str, Any]] = [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                # ``parameters`` is a typed JSONSchema; dump to the wire-format
                # dict so Anthropic's ``input_schema`` (and downstream JSON
                # serialization) gets a plain dict.
                "input_schema": (
                    tool.function.parameters.model_dump(mode="json", exclude_none=True)
                    if tool.function.parameters is not None
                    else {"type": "object", "properties": {}}
                ),
            }
            for tool in tools
        ]
        return anthropic_tools

    @staticmethod
    def _translate_tool_choice(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
        """Translates an OpenAI-format ``tool_choice`` value to Anthropic shape."""
        if isinstance(tool_choice, str):
            mapped = _OPENAI_TO_ANTHROPIC_TOOL_CHOICE.get(tool_choice)
            if mapped is None:
                raise ValueError(
                    f"Unsupported tool_choice value '{tool_choice}'. "
                    "Expected one of: 'auto', 'required', 'none', "
                    "or a {'type': 'function', 'function': {'name': ...}} dict."
                )
            return mapped
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function") or {}
            name = function.get("name")
            if not name:
                raise ValueError(
                    f"tool_choice dict missing function.name: {tool_choice}"
                )
            return {"type": "tool", "name": name}

    @staticmethod
    def _messages_to_anthropic_blocks(
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Converts non-system messages to Anthropic's role-and-blocks shape.

        Translates each message into a list of content blocks (text, image,
        tool_use, tool_result) and merges adjacent same-role messages into a
        single turn (Anthropic requires alternating user/assistant turns).
        ``Role.TOOL`` collapses to user.

        For pure-text turns originating from a single ``content: str`` message,
        the wire shape is collapsed back to ``content: "string"`` to keep the
        payload tidy; both forms are valid.
        """
        result: list[dict[str, Any]] = []
        for role, group_iter in itertools.groupby(
            messages, key=AnthropicInferenceEngine._effective_anthropic_role
        ):
            group = list(group_iter)
            blocks: list[dict[str, Any]] = []
            for msg in group:
                blocks.extend(
                    AnthropicInferenceEngine._message_to_anthropic_blocks(msg)
                )
            # Collapse a solo single-text-block turn back to bare-string content.
            # The type check matters — a Role.TOOL message with str content
            # produces a tool_result block, which must stay in the list form.
            if (
                len(group) == 1
                and len(blocks) == 1
                and blocks[0].get("type") == "text"
                and isinstance(group[0].content, str)
            ):
                result.append({"role": role, "content": group[0].content})
            else:
                result.append({"role": role, "content": blocks})
        return result

    @staticmethod
    def _effective_anthropic_role(message: Message) -> str:
        """Anthropic merges ``Role.TOOL`` messages into ``user`` turns."""
        return Role.USER.value if message.role == Role.TOOL else message.role.value

    @staticmethod
    def _message_to_anthropic_blocks(message: Message) -> list[dict[str, Any]]:
        """Translates one Oumi ``Message`` into a list of Anthropic content blocks."""
        if message.role == Role.TOOL:
            return [AnthropicInferenceEngine._tool_result_block(message)]
        blocks = AnthropicInferenceEngine._content_to_anthropic_blocks(message)
        if message.role == Role.ASSISTANT and message.tool_calls:
            blocks.extend(
                AnthropicInferenceEngine._tool_use_block(tc)
                for tc in message.tool_calls
            )
        return blocks

    @staticmethod
    def _tool_result_block(message: Message) -> dict[str, Any]:
        """Builds an Anthropic ``tool_result`` block from a ``Role.TOOL`` message.

        Anthropic accepts either a plain string or a list of content blocks for
        ``content``; the string form is preserved when the source was a plain
        ``str`` to keep the wire payload tidy.
        """
        if not message.tool_call_id:
            raise ValueError("Role.TOOL message is missing tool_call_id")
        if message.content is None:
            content: str | list[dict[str, Any]] = ""
        elif isinstance(message.content, str):
            content = message.content
        else:
            content = AnthropicInferenceEngine._content_to_anthropic_blocks(message)
        return {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": content,
        }

    @staticmethod
    def _tool_use_block(tool_call: ToolCall) -> dict[str, Any]:
        """Builds an Anthropic ``tool_use`` block from an OpenAI-shaped ``ToolCall``.

        Anthropic expects a parsed ``input`` dict; OpenAI keeps ``arguments`` as
        a JSON string, so it is decoded here.
        """
        args = tool_call.function.arguments
        return {
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "input": json.loads(args) if args else {},
        }

    @staticmethod
    def _content_to_anthropic_blocks(message: Message) -> list[dict[str, Any]]:
        """Converts a message's content into a list of Anthropic content blocks."""
        if message.content is None:
            return []
        if isinstance(message.content, str):
            return [{"type": "text", "text": message.content}]
        return [
            AnthropicInferenceEngine._content_item_to_anthropic_block(item)
            for item in message.content
        ]

    @staticmethod
    def _content_item_to_anthropic_block(item: ContentItem) -> dict[str, Any]:
        """Translates a single ``ContentItem`` into an Anthropic content block."""
        if item.type == Type.TEXT:
            return {"type": "text", "text": item.content or ""}
        if item.type == Type.IMAGE_URL and not item.binary:
            return {
                "type": "image",
                "source": {"type": "url", "url": item.content or ""},
            }
        # IMAGE_PATH carries a filesystem path; load the bytes once. IMAGE_BINARY
        # already has them, and IMAGE_URL with prefetched bytes also skips load.
        if item.type == Type.IMAGE_PATH and not item.binary:
            item = load_image_bytes_to_content_item(item)
        b64_data = base64encode_content_item_image_bytes(item, add_mime_prefix=False)
        media_type = _detect_image_media_type(item.binary or b"")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        }

    @staticmethod
    @override
    def _extract_usage_from_response(
        response: dict[str, Any],
    ) -> dict[str, int] | None:
        """Extract normalized token usage from an Anthropic API response."""
        usage = response.get("usage")
        if not usage:
            return None
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        result = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        # Extract cached tokens from Anthropic's flat format
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        if cached_tokens:
            result["cached_tokens"] = cached_tokens
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        if cache_creation_tokens:
            result["cache_creation_tokens"] = cache_creation_tokens
        return result

    @staticmethod
    @override
    def _extract_finish_reason_from_response(
        response: dict[str, Any],
    ) -> FinishReason | None:
        """Extract normalized finish_reason from an Anthropic API response."""
        raw_reason = response.get("stop_reason")
        if raw_reason is None:
            return None
        mapping = {
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "stop_sequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
        }
        return mapping.get(raw_reason, FinishReason.UNKNOWN)

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an Anthropic API response to a conversation."""
        content_blocks = response.get(_CONTENT_KEY, [])
        if not content_blocks:
            raise RuntimeError(
                f"Anthropic API returned empty content. "
                f"stop_reason={response.get('stop_reason')}, "
                f"type={response.get('type')}, "
                f"model={response.get('model')}, "
                f"usage={response.get('usage')}"
            )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "tool_use":
                # Translate to OpenAI wire format. Anthropic returns a parsed
                # ``input`` dict; OpenAI keeps ``arguments`` as a JSON string.
                tool_calls.append(
                    ToolCall.model_validate(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                )
            elif block_type == "text" or "text" in block:
                # Real Anthropic responses always set type="text"; tolerate
                # fixtures that omit type when only `text` is present.
                text_parts.append(block.get("text", ""))
        text_content = "".join(text_parts)
        new_message = Message(
            content=text_content if text_content else None,
            role=Role.ASSISTANT,
            tool_calls=tool_calls or None,
        )
        metadata = dict(original_conversation.metadata)
        usage = self._extract_usage_from_response(response)
        if usage is not None:
            metadata["usage"] = usage
        finish_reason = self._extract_finish_reason_from_response(response)
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason.value
        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=metadata,
            conversation_id=original_conversation.conversation_id,
            tools=original_conversation.tools,
        )

    @override
    def _get_request_headers(self, remote_params: RemoteParams) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "X-API-Key": self._get_api_key(remote_params) or "",
        }

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "guided_decoding",
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "tool_choice",
            "top_p",
        }

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=5, politeness_policy=60.0)

    #
    # Batch API methods
    #

    def _get_batch_api_url(self) -> str:
        """Returns the URL for the Anthropic batch API."""
        return "https://api.anthropic.com/v1/messages/batches"

    def _convert_anthropic_batch_to_batch_info(
        self, response: dict[str, Any]
    ) -> BatchInfo:
        """Convert Anthropic batch response to BatchInfo.

        Anthropic uses different field names and status values than the OpenAI format:
        - `processing_status` instead of `status`
        - Status values: "in_progress", "canceling", "ended"
        - RFC 3339 timestamps instead of Unix timestamps
        - `results_url` instead of `output_file_id`

        Args:
            response: Raw API response dictionary from Anthropic

        Returns:
            BatchInfo: Parsed batch information
        """
        # Map Anthropic processing_status to BatchStatus
        processing_status = response.get("processing_status", "")
        request_counts = response.get("request_counts", {})

        if processing_status == "in_progress":
            status = BatchStatus.IN_PROGRESS
        elif processing_status == "canceling":
            status = BatchStatus.CANCELLING
        elif processing_status == "ended":
            # Determine final status based on request_counts
            if request_counts.get("canceled", 0) > 0:
                status = BatchStatus.CANCELLED
            elif request_counts.get("errored", 0) > 0:
                status = BatchStatus.FAILED
            elif request_counts.get("expired", 0) > 0:
                status = BatchStatus.EXPIRED
            else:
                status = BatchStatus.COMPLETED
        else:
            # Default to in_progress for unknown statuses
            status = BatchStatus.IN_PROGRESS

        # Calculate total requests from request_counts
        total = (
            request_counts.get("processing", 0)
            + request_counts.get("succeeded", 0)
            + request_counts.get("errored", 0)
            + request_counts.get("canceled", 0)
            + request_counts.get("expired", 0)
        )

        return BatchInfo(
            id=response["id"],
            status=status,
            total_requests=total,
            completed_requests=request_counts.get("succeeded", 0),
            failed_requests=request_counts.get("errored", 0),
            endpoint="/v1/messages",
            created_at=self._parse_iso_timestamp(response.get("created_at")),
            expires_at=self._parse_iso_timestamp(response.get("expires_at")),
            completed_at=self._parse_iso_timestamp(response.get("ended_at")),
            canceling_at=self._parse_iso_timestamp(response.get("cancel_initiated_at")),
            # Store results_url in metadata for later retrieval
            metadata={
                "results_url": response.get("results_url"),
                "archived_at": response.get("archived_at"),
                "processing_status": processing_status,
            },
        )

    @override
    def infer_batch(
        self,
        conversations: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> str:
        """Creates a new batch inference job using the Anthropic Message Batches API.

        The Anthropic batch API processes requests asynchronously and can take up to
        24 hours to complete. Unlike the OpenAI batch API, Anthropic does not require
        uploading a file first - requests are sent directly in the API call.

        Args:
            conversations: List of conversations to process in batch
            inference_config: Parameters for inference

        Returns:
            str: The batch job ID
        """
        if inference_config:
            generation_params = inference_config.generation or self._generation_params
            model_params = inference_config.model or self._model_params
        else:
            generation_params = self._generation_params
            model_params = self._model_params

        return safe_asyncio_run(
            self._create_anthropic_batch(conversations, generation_params, model_params)
        )

    async def _create_anthropic_batch(
        self,
        conversations: list[Conversation],
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> str:
        """Creates a new batch job with the Anthropic API.

        Args:
            conversations: List of conversations to process in batch
            generation_params: Generation parameters
            model_params: Model parameters

        Returns:
            str: The batch job ID
        """
        # Prepare batch requests in Anthropic format
        requests = []
        for i, conv in enumerate(conversations):
            api_input = self._convert_conversation_to_api_input(
                conv, generation_params, model_params
            )
            requests.append(
                {
                    "custom_id": f"request-{i}",
                    "params": api_input,
                }
            )

        # Create batch
        async with self._create_session() as (session, headers):
            async with session.post(
                self._get_batch_api_url(),
                json={"requests": requests},
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to create batch: {error_text}")
                data = await response.json()
                return data["id"]

    @override
    def get_batch_status(self, batch_id: str) -> BatchInfo:
        """Gets the status of a batch inference job.

        Args:
            batch_id: The batch job ID

        Returns:
            BatchInfo: Current status of the batch job
        """
        return safe_asyncio_run(self._get_anthropic_batch_status(batch_id))

    async def _get_anthropic_batch_status(self, batch_id: str) -> BatchInfo:
        """Gets the status of a batch job from the Anthropic API.

        Args:
            batch_id: ID of the batch job

        Returns:
            BatchInfo: Current status of the batch job
        """
        async with self._create_session() as (session, headers):
            async with session.get(
                f"{self._get_batch_api_url()}/{batch_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to get batch status: {error_text}")
                data = await response.json()
                return self._convert_anthropic_batch_to_batch_info(data)

    @override
    def list_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
    ) -> BatchListResponse:
        """Lists batch jobs.

        Args:
            after: Cursor for pagination (batch ID to start after)
            limit: Maximum number of batches to return (1-1000)

        Returns:
            BatchListResponse: List of batch jobs
        """
        return safe_asyncio_run(self._list_anthropic_batches(after=after, limit=limit))

    async def _list_anthropic_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
    ) -> BatchListResponse:
        """Lists batch jobs from the Anthropic API.

        Args:
            after: Cursor for pagination (batch ID to start after)
            limit: Maximum number of batches to return (1-1000)

        Returns:
            BatchListResponse: List of batch jobs
        """
        async with self._create_session() as (session, headers):
            params: dict[str, str] = {}
            if after:
                params["after_id"] = after
            if limit:
                params["limit"] = str(limit)

            async with session.get(
                self._get_batch_api_url(),
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to list batches: {error_text}")
                data = await response.json()

                batches = [
                    self._convert_anthropic_batch_to_batch_info(batch_data)
                    for batch_data in data.get("data", [])
                ]

                return BatchListResponse(
                    batches=batches,
                    first_id=data.get("first_id"),
                    last_id=data.get("last_id"),
                    has_more=data.get("has_more", False),
                )

    @override
    def get_batch_results(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """Gets the results of a completed batch job.

        Args:
            batch_id: The batch job ID
            conversations: Original conversations used to create the batch

        Returns:
            List[Conversation]: The processed conversations with responses

        Raises:
            RuntimeError: If the batch failed, has not completed, or any items failed
        """
        batch_result = self.get_batch_results_partial(batch_id, conversations)
        if batch_result.has_failures:
            first_idx = batch_result.failed_indices[0]
            raise RuntimeError(
                f"Batch {batch_id} failed for "
                f"{len(batch_result.failed_indices)} items. "
                f"First error (index {first_idx}): "
                f"{batch_result.error_messages.get(first_idx, 'unknown')}"
            )
        return [conv for _, conv in sorted(batch_result.successful)]

    @override
    def get_batch_results_partial(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> BatchResult:
        """Gets partial results of a completed Anthropic batch job."""
        return safe_asyncio_run(
            self._get_anthropic_batch_results_partial(batch_id, conversations)
        )

    async def _get_anthropic_batch_results_partial(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> BatchResult:
        """Gets partial results of a completed Anthropic batch job.

        Args:
            batch_id: ID of the batch job
            conversations: Original conversations used to create the batch

        Returns:
            BatchResult with successful and failed items

        Raises:
            RuntimeError: If the batch is not terminal or is unrecoverably failed
        """
        batch_info = await self._get_anthropic_batch_status(batch_id)

        if not batch_info.is_terminal:
            raise RuntimeError(
                f"Batch is not in terminal state. Status: {batch_info.status}"
            )

        if batch_info.status in (
            BatchStatus.EXPIRED,
            BatchStatus.CANCELLED,
        ):
            raise RuntimeError(
                f"Batch is unrecoverably {batch_info.status.value}: "
                f"error={batch_info.error}"
            )

        # FAILED batches may still have partial results (some rows succeeded).
        if batch_info.status == BatchStatus.FAILED:
            logger.warning(
                f"Batch {batch_id} has FAILED status but attempting to "
                f"retrieve partial results "
                f"(completed={batch_info.completed_requests}, "
                f"failed={batch_info.failed_requests})"
            )

        # Get results URL from metadata
        results_url = (
            batch_info.metadata.get("results_url") if batch_info.metadata else None
        )
        if not results_url:
            raise RuntimeError("No results URL available")

        # Download results
        async with self._create_session() as (session, headers):
            async with session.get(results_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to download batch results: {error_text}"
                    )
                results_content = await response.text()

        logger.info(
            f"Batch {batch_id}: retrieving partial results "
            f"(status={batch_info.status.value}, "
            f"total={len(conversations)} requests)"
        )

        # Parse results — Anthropic puts both successes and errors in one file,
        successful: list[tuple[int, Conversation]] = []
        failed_indices: list[int] = []
        error_messages: dict[int, str] = {}
        all_indices = set(range(len(conversations)))
        seen_indices: set[int] = set()

        for line in results_content.strip().splitlines():
            if not line:
                continue
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            try:
                idx = int(custom_id.split("-", 1)[1])
            except (IndexError, ValueError):
                continue

            seen_indices.add(idx)
            result_type = result.get("result", {}).get("type")

            if result_type in ("error", "errored"):
                error_info = result.get("result", {}).get("error", {})
                # Anthropic nests the detail under error.error
                inner_error = error_info.get("error", {})
                if isinstance(inner_error, dict) and inner_error.get("message"):
                    error_type = inner_error.get("type", error_info.get("type"))
                    error_msg = inner_error["message"]
                else:
                    error_type = error_info.get("type")
                    error_msg = error_info.get("message")
                failed_indices.append(idx)
                error_messages[idx] = f"{error_type}: {error_msg}"
            elif result_type == "succeeded":
                try:
                    message_response = result.get("result", {}).get("message", {})
                    conv = self._convert_api_output_to_conversation(
                        message_response, conversations[idx]
                    )
                    successful.append((idx, conv))
                except Exception as e:
                    failed_indices.append(idx)
                    error_messages[idx] = f"Failed to parse response: {e}"
            else:
                failed_indices.append(idx)
                error_messages[idx] = f"Unexpected result type: {result_type}"

        # Any index missing from results is also a failure
        for idx in sorted(all_indices - seen_indices):
            failed_indices.append(idx)
            error_messages[idx] = "Request missing from batch output"

        logger.info(
            f"Batch {batch_id}: {len(successful)} succeeded, "
            f"{len(failed_indices)} failed out of {len(conversations)} total"
        )
        if error_messages:
            for idx, msg in error_messages.items():
                logger.warning(f"Batch {batch_id} request {idx} failed: {msg}")

        return BatchResult(
            successful=successful,
            failed_indices=sorted(failed_indices),
            error_messages=error_messages,
        )

    def cancel_batch(self, batch_id: str) -> BatchInfo:
        """Cancels a batch inference job.

        Batches may be canceled any time before processing ends. Once cancellation
        is initiated, the batch enters a "canceling" state.

        Args:
            batch_id: The batch job ID to cancel

        Returns:
            BatchInfo: Updated status of the batch job
        """
        return safe_asyncio_run(self._cancel_anthropic_batch(batch_id))

    async def _cancel_anthropic_batch(self, batch_id: str) -> BatchInfo:
        """Cancels a batch job via the Anthropic API.

        Args:
            batch_id: ID of the batch job to cancel

        Returns:
            BatchInfo: Updated status of the batch job
        """
        async with self._create_session() as (session, headers):
            async with session.post(
                f"{self._get_batch_api_url()}/{batch_id}/cancel",
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to cancel batch: {error_text}")
                data = await response.json()
                return self._convert_anthropic_batch_to_batch_info(data)


def _convert_guided_decoding_config_to_api_input(
    guided_config: GuidedDecodingParams,
) -> dict:
    """Converts a guided decoding configuration to an Anthropic API input."""
    if guided_config.json is None:
        raise ValueError(
            "Only JSON schema guided decoding is supported, got '%s'",
            guided_config,
        )

    json_schema = guided_config.json

    if isinstance(json_schema, type) and issubclass(json_schema, pydantic.BaseModel):
        schema_value = json_schema.model_json_schema()
    elif isinstance(json_schema, dict):
        schema_value = copy.deepcopy(json_schema)
    elif isinstance(json_schema, str):
        schema_value = json.loads(json_schema)
    else:
        raise ValueError(
            f"Got unsupported JSON schema type: {type(json_schema)}"
            "Please provide a Pydantic model or a JSON schema as a "
            "string or dict."
        )

    # Anthropic's output_config requires `additionalProperties: false` on every
    # object schema; inject it where missing.
    RemoteInferenceEngine._enforce_additional_properties_false(schema_value)

    return {
        "output_config": {
            "format": {"type": "json_schema", "schema": schema_value},
        },
    }


def _model_supports_output_config(model_name: str) -> bool:
    """Returns True if the model accepts the output_config field, False otherwise."""
    # Anthropic's `output_config` (structured outputs) is GA on Claude 4.5+ (Opus,
    # Sonnet, Haiku) and Mythos. Older models (Claude 3.x, 3.5) reject the field.
    if model_name.startswith("claude-mythos"):
        return True

    version_re = re.compile(r"^claude-(?:opus|sonnet|haiku)-(\d+)-(\d+)")
    match = version_re.match(model_name)
    return (int(match[1]), int(match[2])) >= (4, 5) if match else False
