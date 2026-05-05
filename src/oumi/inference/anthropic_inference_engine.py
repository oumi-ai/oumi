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

import json
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
from oumi.core.types.conversation import Conversation, FinishReason, Message, Role
from oumi.inference.remote_inference_engine import (
    BatchInfo,
    BatchListResponse,
    BatchResult,
    BatchStatus,
    RemoteInferenceEngine,
)
from oumi.utils.logging import logger

_CONTENT_KEY: str = "content"


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
        body = {
            "model": model_params.model_name,
            "messages": self._get_list_of_message_json_dicts(
                messages, group_adjacent_same_role_turns=True
            ),
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
            body.update(
                _convert_guided_decoding_config_to_api_input(
                    generation_params.guided_decoding
                )
            )

        # Enable prompt caching. Anthropic automatically caches content up to
        # the last cacheable block. This reduces latency and cost for repeated
        # prefixes (system prompts, long context, multi-turn conversations).
        # See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        body["cache_control"] = {"type": "ephemeral"}

        return body

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
        # Forced tool use returns the structured payload in a tool_use block.
        message_content: str | None = None
        for block in content_blocks:
            if block.get("type") == "tool_use":
                message_content = json.dumps(block.get("input", {}))
                break
        if message_content is None:
            message_content = content_blocks[0]["text"]
        new_message = Message(
            content=message_content,
            role=Role.ASSISTANT,
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
        schema_name = json_schema.__name__
        schema_value = json_schema.model_json_schema()
    elif isinstance(json_schema, dict):
        schema_name = "Response"  # Generic name if no schema name is provided.
        schema_value = json_schema
    elif isinstance(json_schema, str):
        schema_name = "Response"  # Generic name if no schema name is provided.
        schema_value = json.loads(json_schema)
    else:
        raise ValueError(
            f"Got unsupported JSON schema type: {type(json_schema)}"
            "Please provide a Pydantic model or a JSON schema as a "
            "string or dict."
        )

    return {
        "tools": [{"name": schema_name, "input_schema": schema_value}],
        "tool_choice": {"type": "tool", "name": schema_name},
    }
