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
from datetime import datetime
from typing import Any

import aiohttp
from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import (
    BatchInfo,
    BatchListResponse,
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

        return body

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an Anthropic API response to a conversation."""
        new_message = Message(
            content=response[_CONTENT_KEY][0]["text"],
            role=Role.ASSISTANT,
        )
        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
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

    @staticmethod
    def _parse_rfc3339_timestamp(timestamp: str | None) -> datetime | None:
        """Parse RFC 3339 timestamp string to datetime.

        Args:
            timestamp: RFC 3339 formatted timestamp string
                (e.g., "2024-01-01T00:00:00Z")

        Returns:
            datetime or None if timestamp is None or empty
        """
        if not timestamp:
            return None
        # Handle both "Z" suffix and "+00:00" format
        timestamp = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(timestamp)

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
            status = BatchStatus.CANCELLED
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
            created_at=self._parse_rfc3339_timestamp(response.get("created_at")),
            expires_at=self._parse_rfc3339_timestamp(response.get("expires_at")),
            completed_at=self._parse_rfc3339_timestamp(response.get("ended_at")),
            canceling_at=self._parse_rfc3339_timestamp(
                response.get("cancel_initiated_at")
            ),
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
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

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
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

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
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

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
            RuntimeError: If the batch failed or has not completed
        """
        return safe_asyncio_run(
            self._get_anthropic_batch_results(batch_id, conversations)
        )

    async def _get_anthropic_batch_results(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """Gets the results of a completed batch job from the Anthropic API.

        Args:
            batch_id: ID of the batch job
            conversations: Original conversations used to create the batch

        Returns:
            List[Conversation]: The processed conversations with responses

        Raises:
            RuntimeError: If batch status is not completed or if there are errors
        """
        # Get batch status first
        batch_info = await self._get_anthropic_batch_status(batch_id)

        if not batch_info.is_terminal:
            raise RuntimeError(
                f"Batch is not in terminal state. Status: {batch_info.status}"
            )

        if batch_info.status == BatchStatus.FAILED:
            raise RuntimeError(f"Batch failed: {batch_info.error}")

        if batch_info.status == BatchStatus.CANCELLED:
            raise RuntimeError("Batch was cancelled")

        if batch_info.status == BatchStatus.EXPIRED:
            raise RuntimeError("Batch expired before completion")

        # Get results URL from metadata
        results_url = (
            batch_info.metadata.get("results_url") if batch_info.metadata else None
        )
        if not results_url:
            raise RuntimeError("No results URL available")

        # Download results from the URL
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

            async with session.get(results_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to download batch results: {error_text}"
                    )
                results_content = await response.text()

        # Parse results and map back to conversations by custom_id
        results_by_id: dict[str, dict[str, Any]] = {}
        for line in results_content.strip().splitlines():
            if not line:
                continue
            result = json.loads(line)
            custom_id = result.get("custom_id")
            if custom_id:
                results_by_id[custom_id] = result

        # Build output conversations in order
        processed_conversations = []
        for i, conv in enumerate(conversations):
            custom_id = f"request-{i}"
            result = results_by_id.get(custom_id)

            if not result:
                raise RuntimeError(f"Missing result for {custom_id}")

            result_type = result.get("result", {}).get("type")
            if result_type == "error":
                error_info = result.get("result", {}).get("error", {})
                raise RuntimeError(
                    f"Batch request {custom_id} failed: "
                    f"{error_info.get('type')}: {error_info.get('message')}"
                )

            if result_type != "succeeded":
                raise RuntimeError(
                    f"Batch request {custom_id} has unexpected result type: "
                    f"{result_type}"
                )

            # Extract the message response
            message_response = result.get("result", {}).get("message", {})
            processed_conv = self._convert_api_output_to_conversation(
                message_response, conv
            )
            processed_conversations.append(processed_conv)

        return processed_conversations

    @override
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
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

            async with session.post(
                f"{self._get_batch_api_url()}/{batch_id}/cancel",
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to cancel batch: {error_text}")
                data = await response.json()
                return self._convert_anthropic_batch_to_batch_info(data)
