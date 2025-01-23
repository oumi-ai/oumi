import asyncio
import copy
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiohttp
import jsonlines
import pydantic
from tqdm.asyncio import tqdm
from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import (
    Conversation,
    Message,
    Role,
)
from oumi.utils.conversation_utils import (
    convert_message_to_json_content_list,
    create_list_of_message_json_dicts,
)

_AUTHORIZATION_KEY: str = "Authorization"
_BATCH_PURPOSE = "batch"
_BATCH_ENDPOINT = "/v1/chat/completions"


class BatchStatus(Enum):
    """Status of a batch inference job."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class BatchInfo:
    """Information about a batch job."""

    id: str
    status: BatchStatus
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    endpoint: Optional[str] = None
    input_file_id: Optional[str] = None
    batch_completion_window: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    in_progress_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    finalizing_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    canceling_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None

    @staticmethod
    def _convert_timestamp(timestamp: Optional[int]) -> Optional[datetime]:
        """Convert Unix timestamp to datetime.

        Args:
            timestamp: Unix timestamp in seconds

        Returns:
            datetime: Converted datetime or None if timestamp is None
        """
        return datetime.fromtimestamp(timestamp) if timestamp is not None else None

    @classmethod
    def from_api_response(cls, response: dict[str, Any]) -> "BatchInfo":
        """Create BatchInfo from API response dictionary.

        Args:
            response: Raw API response dictionary

        Returns:
            BatchInfo: Parsed batch information
        """
        return cls(
            id=response["id"],
            status=BatchStatus(response["status"]),
            endpoint=response.get("endpoint"),
            input_file_id=response.get("input_file_id"),
            batch_completion_window=response.get("batch_completion_window"),
            output_file_id=response.get("output_file_id"),
            error_file_id=response.get("error_file_id"),
            error=response.get("error"),
            created_at=cls._convert_timestamp(response.get("created_at")),
            in_progress_at=cls._convert_timestamp(response.get("in_progress_at")),
            expires_at=cls._convert_timestamp(response.get("expires_at")),
            finalizing_at=cls._convert_timestamp(response.get("finalizing_at")),
            completed_at=cls._convert_timestamp(response.get("completed_at")),
            failed_at=cls._convert_timestamp(response.get("failed_at")),
            expired_at=cls._convert_timestamp(response.get("expired_at")),
            canceling_at=cls._convert_timestamp(response.get("cancelling_at")),
            canceled_at=cls._convert_timestamp(response.get("cancelled_at")),
            total_requests=response.get("request_counts", {}).get("total", 0),
            completed_requests=response.get("request_counts", {}).get("completed", 0),
            failed_requests=response.get("request_counts", {}).get("failed", 0),
            metadata=response.get("metadata"),
        )

    @property
    def is_terminal(self) -> bool:
        """Return True if the batch is in a terminal state."""
        return self.status in (
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.EXPIRED,
            BatchStatus.CANCELLED,
        )

    @property
    def completion_percentage(self) -> float:
        """Return the percentage of completed requests."""
        return (
            (100 * self.completed_requests / self.total_requests)
            if self.total_requests > 0
            else 0.0
        )

    @property
    def has_errors(self) -> bool:
        """Return True if the batch has any errors."""
        return bool(self.error) or self.failed_requests > 0


@dataclass
class BatchListResponse:
    """Response from listing batch jobs."""

    batches: list[BatchInfo]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


@dataclass
class FileInfo:
    """Information about a file."""

    id: str
    filename: str
    bytes: int
    created_at: int
    purpose: str


@dataclass
class FileListResponse:
    """Response from listing files."""

    files: list[FileInfo]
    has_more: bool = False


class RemoteInferenceEngine(BaseInferenceEngine):
    """Engine for running inference against a server implementing the OpenAI API."""

    def __init__(self, model_params: ModelParams, remote_params: RemoteParams):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote server params.
        """
        self._model = model_params.model_name
        self._adapter_model = model_params.adapter_model
        self._remote_params = copy.deepcopy(remote_params)

    @staticmethod
    def _get_list_of_message_json_dicts(
        messages: list[Message],
        *,
        group_adjacent_same_role_turns: bool,
    ) -> list[dict[str, Any]]:
        return create_list_of_message_json_dicts(
            messages, group_adjacent_same_role_turns=group_adjacent_same_role_turns
        )

    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Converts a conversation to an OpenAI input.

        Documentation: https://platform.openai.com/docs/api-reference/chat/create

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the OpenAI input.
        """
        api_input = {
            "model": self._model,
            "messages": [
                {
                    "content": convert_message_to_json_content_list(message),
                    "role": message.role.value,
                }
                for message in conversation.messages
            ],
            "max_completion_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "frequency_penalty": generation_params.frequency_penalty,
            "presence_penalty": generation_params.presence_penalty,
            "n": 1,  # Number of completions to generate for each prompt.
            "seed": generation_params.seed,
            "logit_bias": generation_params.logit_bias,
        }

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings

        if generation_params.guided_decoding:
            json_schema = generation_params.guided_decoding.json

            if json_schema is not None:
                if isinstance(json_schema, type) and issubclass(
                    json_schema, pydantic.BaseModel
                ):
                    schema_name = json_schema.__name__
                    schema_value = json_schema.model_json_schema()
                elif isinstance(json_schema, dict):
                    # Use a generic name if no schema is provided.
                    schema_name = "Response"
                    schema_value = json_schema
                elif isinstance(json_schema, str):
                    # Use a generic name if no schema is provided.
                    schema_name = "Response"
                    # Try to parse as JSON string
                    schema_value = json.loads(json_schema)
                else:
                    raise ValueError(
                        f"Got unsupported JSON schema type: {type(json_schema)}"
                        "Please provide a Pydantic model or a JSON schema as a "
                        "string or dict."
                    )

                api_input["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema_value,
                    },
                }
            else:
                raise ValueError(
                    "Only JSON schema guided decoding is supported, got '%s'",
                    generation_params.guided_decoding,
                )

        return api_input

    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an API response to a conversation.

        Args:
            response: The API response to convert.
            original_conversation: The original conversation.

        Returns:
            Conversation: The conversation including the generated response.
        """
        message = response["choices"][0]["message"]
        return Conversation(
            messages=[
                *original_conversation.messages,
                Message(
                    content=message["content"],
                    role=Role(message["role"]),
                ),
            ],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    def _get_api_key(self, remote_params: RemoteParams) -> Optional[str]:
        if not remote_params:
            return None

        if remote_params.api_key:
            return remote_params.api_key

        if remote_params.api_key_env_varname:
            return os.environ.get(remote_params.api_key_env_varname)

        return None

    def _get_request_headers(
        self, remote_params: Optional[RemoteParams]
    ) -> dict[str, str]:
        headers = {}

        if not remote_params:
            return headers

        headers[_AUTHORIZATION_KEY] = f"Bearer {self._get_api_key(remote_params)}"
        return headers

    async def _query_api(
        self,
        conversation: Conversation,
        inference_config: InferenceConfig,
        remote_params: RemoteParams,
        semaphore: asyncio.Semaphore,
        session: aiohttp.ClientSession,
    ) -> Conversation:
        """Queries the API with the provided input.

        Args:
            conversation: The conversations to run inference on.
            inference_config: Parameters for inference.
            remote_params: Parameters for running inference against a remote API.
            semaphore: Semaphore to limit concurrent requests.
            session: The aiohttp session to use for the request.

        Returns:
            Conversation: Inference output.
        """
        assert remote_params.api_url
        async with semaphore:
            api_input = self._convert_conversation_to_api_input(
                conversation, inference_config.generation
            )
            headers = self._get_request_headers(inference_config.remote_params)
            retries = 0
            failure_reason = None
            # Retry the request if it fails.
            for _ in range(remote_params.max_retries + 1):
                async with session.post(
                    remote_params.api_url,
                    json=api_input,
                    headers=headers,
                    timeout=remote_params.connection_timeout,
                ) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        result = self._convert_api_output_to_conversation(
                            response_json, conversation
                        )
                        if inference_config.output_path:
                            # Write what we have so far to our scratch directory.
                            self._save_conversation(
                                result,
                                self._get_scratch_filepath(
                                    inference_config.output_path
                                ),
                            )
                        await asyncio.sleep(remote_params.politeness_policy)
                        return result
                    else:
                        failure_reason = (
                            response_json.get("error").get("message")
                            if response_json and response_json.get("error")
                            else None
                        )
                        retries += 1
                        await asyncio.sleep(remote_params.politeness_policy)
            raise RuntimeError(
                f"Failed to query API after {remote_params.max_retries} retries. "
                + (f"Reason: {failure_reason}" if failure_reason else "")
            )

    async def _infer(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig,
        remote_params: RemoteParams,
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.
            remote_params: Parameters for running inference against a remote API.

        Returns:
            List[Conversation]: Inference output.
        """
        # Limit number of HTTP connections to the number of workers.
        connector = aiohttp.TCPConnector(limit=remote_params.num_workers)
        # Control the number of concurrent tasks via a semaphore.
        semaphore = asyncio.BoundedSemaphore(remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._query_api(
                    conversation,
                    inference_config,
                    remote_params,
                    semaphore,
                    session,
                )
                for conversation in input
            ]

            disable_tqdm = len(tasks) < 2
            return await tqdm.gather(*tasks, disable=disable_tqdm)

    @override
    def infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig,
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        if not inference_config.remote_params:
            raise ValueError("Remote params must be provided in inference config.")
        conversations = safe_asyncio_run(
            self._infer(input, inference_config, inference_config.remote_params)
        )
        if inference_config.output_path:
            self._save_conversations(conversations, inference_config.output_path)
        return conversations

    @override
    def infer_from_file(
        self, input_filepath: str, inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        if not inference_config.remote_params:
            raise ValueError("Remote params must be provided in inference config.")
        input = self._read_conversations(input_filepath)
        conversations = safe_asyncio_run(
            self._infer(input, inference_config, inference_config.remote_params)
        )
        if inference_config.output_path:
            self._save_conversations(conversations, inference_config.output_path)
        return conversations

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "logit_bias",
            "max_new_tokens",
            "presence_penalty",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }

    #
    # Batch inference
    #
    def infer_batch(
        self,
        conversations: list[Conversation],
        inference_config: InferenceConfig,
    ) -> str:
        """Creates a new batch inference job.

        Args:
            conversations: List of conversations to process in batch
            inference_config: Parameters for inference

        Returns:
            str: The batch job ID

        Raises:
            ValueError: If remote_params is not provided in inference_config
        """
        if not inference_config.remote_params:
            raise ValueError("Remote params must be provided in inference_config.")

        return safe_asyncio_run(
            self._create_batch(conversations, inference_config.generation)
        )

    def get_batch_status(
        self,
        batch_id: str,
    ) -> BatchInfo:
        """Gets the status of a batch inference job.

        Args:
            batch_id: The batch job ID

        Returns:
            BatchInfo: Current status of the batch job
        """
        return safe_asyncio_run(self._get_batch_status(batch_id))

    def list_batches(
        self,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> BatchListResponse:
        """Lists batch jobs.

        Args:
            after: Cursor for pagination (batch ID to start after)
            limit: Maximum number of batches to return (1-100)

        Returns:
            BatchListResponse: List of batch jobs
        """
        return safe_asyncio_run(
            self._list_batches(
                after=after,
                limit=limit,
            )
        )

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
            self._get_batch_results_with_mapping(batch_id, conversations)
        )

    async def _upload_batch_file(
        self,
        batch_requests: list[dict],
    ) -> str:
        """Uploads a JSONL file containing batch requests.

        Args:
            batch_requests: List of request objects to include in the batch

        Returns:
            str: The uploaded file ID
        """
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            with jsonlines.Writer(tmp) as writer:
                for request in batch_requests:
                    writer.write(request)
            tmp_path = tmp.name

        try:
            # Upload the file
            connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = self._get_request_headers(self._remote_params)

                # Create form data with file
                form = aiohttp.FormData()
                async with aiofiles.open(tmp_path, "rb") as f:
                    file_data = await f.read()
                    form.add_field("file", file_data, filename="batch_requests.jsonl")
                form.add_field("purpose", _BATCH_PURPOSE)

                async with session.post(
                    f"{self._remote_params.api_url}/files",
                    data=form,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to upload batch file: {await response.text()}"
                        )
                    data = await response.json()
                    return data["id"]
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink()

    async def _create_batch(
        self,
        conversations: list[Conversation],
        generation_params: GenerationParams,
    ) -> str:
        """Creates a new batch job.

        Args:
            conversations: List of conversations to process in batch
            generation_params: Generation parameters

        Returns:
            str: The batch job ID
        """
        # Prepare batch requests
        batch_requests = []
        for i, conv in enumerate(conversations):
            api_input = self._convert_conversation_to_api_input(conv, generation_params)
            batch_requests.append(
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": _BATCH_ENDPOINT,
                    "body": api_input,
                }
            )

        # Upload batch file
        file_id = await self._upload_batch_file(batch_requests)

        # Create batch
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.post(
                f"{self._remote_params.api_url}/batches",
                json={
                    "input_file_id": file_id,
                    "endpoint": _BATCH_ENDPOINT,
                    "batch_completion_window": (
                        self._remote_params.batch_completion_window
                    ),
                },
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to create batch: {await response.text()}"
                    )
                data = await response.json()
                return data["id"]

    async def _get_batch_status(
        self,
        batch_id: str,
    ) -> BatchInfo:
        """Gets the status of a batch job.

        Args:
            batch_id: ID of the batch job

        Returns:
            BatchInfo: Current status of the batch job
        """
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self._remote_params.api_url}/batches/{batch_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to get batch status: {await response.text()}"
                    )
                data = await response.json()
                return BatchInfo.from_api_response(data)

    async def _list_batches(
        self,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> BatchListResponse:
        """Lists batch jobs.

        Args:
            after: Cursor for pagination (batch ID to start after)
            limit: Maximum number of batches to return (1-100)

        Returns:
            BatchListResponse: List of batch jobs
        """
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

            params = {}
            if after:
                params["after"] = after
            if limit:
                params["limit"] = str(limit)

            async with session.get(
                f"{self._remote_params.api_url}/batches",
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to list batches: {await response.text()}"
                    )
                data = await response.json()

                batches = [
                    BatchInfo.from_api_response(batch_data)
                    for batch_data in data["data"]
                ]

                return BatchListResponse(
                    batches=batches,
                    first_id=data.get("first_id"),
                    last_id=data.get("last_id"),
                    has_more=data.get("has_more", False),
                )

    async def _get_batch_results_with_mapping(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """Gets the results of a completed batch job and maps them to conversations.

        Args:
            batch_id: ID of the batch job
            conversations: Original conversations used to create the batch

        Returns:
            List[Conversation]: The processed conversations with responses

        Raises:
            RuntimeError: If batch status is not completed or if there are errors
        """
        # Get batch status first
        batch_info = await self._get_batch_status(batch_id)

        if not batch_info.is_terminal:
            raise RuntimeError(
                f"Batch is not in terminal state. Status: {batch_info.status}"
            )

        if batch_info.has_errors:
            # Download error file if there are failed requests
            if batch_info.error_file_id:
                error_content = await self._download_file(batch_info.error_file_id)
                raise RuntimeError(f"Batch has failed requests: {error_content}")
            raise RuntimeError(f"Batch failed with error: {batch_info.error}")

        # Download results file
        if not batch_info.output_file_id:
            raise RuntimeError("No output file available")

        results_content = await self._download_file(batch_info.output_file_id)

        # Parse results
        processed_conversations = []
        for line, conv in zip(results_content.splitlines(), conversations):
            result = json.loads(line)
            if result.get("error"):
                raise RuntimeError(f"Batch request failed: {result['error']}")
            processed_conv = self._convert_api_output_to_conversation(
                result["response"]["body"], conv
            )
            processed_conversations.append(processed_conv)
        return processed_conversations

    #
    # File operations
    #
    def list_files(
        self,
        purpose: Optional[str] = None,
        limit: Optional[int] = None,
        order: str = "desc",
        after: Optional[str] = None,
    ) -> FileListResponse:
        """Lists files."""
        return safe_asyncio_run(
            self._list_files(
                purpose=purpose,
                limit=limit,
                order=order,
                after=after,
            )
        )

    def get_file(
        self,
        file_id: str,
    ) -> FileInfo:
        """Gets information about a file."""
        return safe_asyncio_run(self._get_file(file_id))

    def delete_file(
        self,
        file_id: str,
    ) -> bool:
        """Deletes a file."""
        return safe_asyncio_run(self._delete_file(file_id))

    def get_file_content(
        self,
        file_id: str,
    ) -> str:
        """Gets a file's content."""
        return safe_asyncio_run(self._download_file(file_id))

    async def _list_files(
        self,
        purpose: Optional[str] = None,
        limit: Optional[int] = None,
        order: str = "desc",
        after: Optional[str] = None,
    ) -> FileListResponse:
        """Lists files.

        Args:
            purpose: Only return files with this purpose
            limit: Maximum number of files to return (1-10000)
            order: Sort order (asc or desc)
            after: Cursor for pagination

        Returns:
            FileListResponse: List of files
        """
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

            params = {"order": order}
            if purpose:
                params["purpose"] = purpose
            if limit:
                params["limit"] = str(limit)
            if after:
                params["after"] = after

            async with session.get(
                f"{self._remote_params.api_url}/files",
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to list files: {await response.text()}")
                data = await response.json()

                files = [
                    FileInfo(
                        id=file["id"],
                        filename=file["filename"],
                        bytes=file["bytes"],
                        created_at=file["created_at"],
                        purpose=file["purpose"],
                    )
                    for file in data["data"]
                ]

                return FileListResponse(
                    files=files, has_more=len(files) == limit if limit else False
                )

    async def _get_file(
        self,
        file_id: str,
    ) -> FileInfo:
        """Gets information about a file.

        Args:
            file_id: ID of the file
            remote_params: Remote API parameters

        Returns:
            FileInfo: File information
        """
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self._remote_params.api_url}/files/{file_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get file: {await response.text()}")
                data = await response.json()
                return FileInfo(
                    id=data["id"],
                    filename=data["filename"],
                    bytes=data["bytes"],
                    created_at=data["created_at"],
                    purpose=data["purpose"],
                )

    async def _delete_file(
        self,
        file_id: str,
    ) -> bool:
        """Deletes a file.

        Args:
            file_id: ID of the file to delete
            remote_params: Remote API parameters

        Returns:
            bool: True if deletion was successful
        """
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.delete(
                f"{self._remote_params.api_url}/files/{file_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to delete file: {await response.text()}"
                    )
                data = await response.json()
                return data.get("deleted", False)

    async def _download_file(
        self,
        file_id: str,
    ) -> str:
        """Downloads a file's content.

        Args:
            file_id: ID of the file to download
            remote_params: Remote API parameters

        Returns:
            str: The file content
        """
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self._remote_params.api_url}/files/{file_id}/content",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to download file: {await response.text()}"
                    )
                return await response.text()
