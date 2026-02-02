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
import os
import uuid
from datetime import datetime
from typing import Any

import aiohttp
from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import (
    BatchInfo,
    BatchListResponse,
    BatchStatus,
    RemoteInferenceEngine,
)


class FireworksInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Fireworks AI API.

    For batch inference, this engine requires the FIREWORKS_ACCOUNT_ID environment
    variable to be set in addition to FIREWORKS_API_KEY.
    """

    account_id_env_varname: str = "FIREWORKS_ACCOUNT_ID"
    """Environment variable name for the Fireworks account ID."""

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the Fireworks API."""
        return "https://api.fireworks.ai/inference/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the Fireworks API key."""
        return "FIREWORKS_API_KEY"

    def _get_account_id(self) -> str:
        """Get the Fireworks account ID from environment variable.

        Returns:
            str: The account ID

        Raises:
            ValueError: If the account ID is not set
        """
        account_id = os.environ.get(self.account_id_env_varname)
        if not account_id:
            raise ValueError(
                f"Fireworks batch API requires the {self.account_id_env_varname} "
                "environment variable to be set."
            )
        return account_id

    def _get_batch_api_base_url(self) -> str:
        """Returns the base URL for the Fireworks batch API."""
        account_id = self._get_account_id()
        return f"https://api.fireworks.ai/v1/accounts/{account_id}"

    def _get_fireworks_request_headers(self) -> dict[str, str]:
        """Get request headers for Fireworks API calls."""
        api_key = self._get_api_key(self._remote_params)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_fireworks_timestamp(timestamp: str | None) -> datetime | None:
        """Parse Fireworks timestamp string to datetime.

        Args:
            timestamp: ISO 8601 formatted timestamp string

        Returns:
            datetime or None if timestamp is None or empty
        """
        if not timestamp:
            return None
        # Handle "Z" suffix
        timestamp = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(timestamp)

    def _convert_fireworks_job_to_batch_info(
        self, response: dict[str, Any]
    ) -> BatchInfo:
        """Convert Fireworks batch job response to BatchInfo.

        Fireworks uses different field names and status values:
        - `state` field with values: CREATING, QUEUED, PENDING, RUNNING, COMPLETED, etc.
        - Different timestamp field names
        - Progress tracked via `jobProgress` object

        Args:
            response: Raw API response dictionary from Fireworks

        Returns:
            BatchInfo: Parsed batch information
        """
        # Map Fireworks state to BatchStatus
        # Fireworks uses JOB_STATE_* prefix (e.g., JOB_STATE_COMPLETED)
        state = response.get("state", "").upper()
        # Remove JOB_STATE_ prefix if present
        if state.startswith("JOB_STATE_"):
            state = state[len("JOB_STATE_") :]
        state_mapping = {
            "UNSPECIFIED": BatchStatus.IN_PROGRESS,
            "CREATING": BatchStatus.VALIDATING,
            "QUEUED": BatchStatus.IN_PROGRESS,
            "PENDING": BatchStatus.IN_PROGRESS,
            "RUNNING": BatchStatus.IN_PROGRESS,
            "COMPLETED": BatchStatus.COMPLETED,
            "FAILED": BatchStatus.FAILED,
            "CANCELLING": BatchStatus.CANCELLED,
            "CANCELLED": BatchStatus.CANCELLED,
            "DELETING": BatchStatus.CANCELLED,
        }
        status = state_mapping.get(state, BatchStatus.IN_PROGRESS)

        # Extract progress information (jobProgress can be None)
        job_progress = response.get("jobProgress") or {}
        total_requests = job_progress.get("totalRequests", 0)
        processed_requests = job_progress.get("processedRequests", 0)
        failed_requests = job_progress.get("failedRequests", 0)

        # Extract job name/id - Fireworks uses "name" field with full path
        job_name = response.get("name", "")
        # Extract just the job ID from the full resource name
        # Format: accounts/{account_id}/batchInferenceJobs/{job_id}
        job_id = job_name.split("/")[-1] if "/" in job_name else job_name

        return BatchInfo(
            id=job_id,
            status=status,
            total_requests=total_requests,
            completed_requests=processed_requests - failed_requests,
            failed_requests=failed_requests,
            endpoint="/v1/chat/completions",
            created_at=self._parse_fireworks_timestamp(response.get("createTime")),
            in_progress_at=self._parse_fireworks_timestamp(response.get("startTime")),
            completed_at=self._parse_fireworks_timestamp(response.get("endTime")),
            metadata={
                "fireworks_state": state,
                "input_dataset_id": response.get("inputDatasetId"),
                "output_dataset_id": response.get("outputDatasetId"),
                "model": response.get("model"),
                "display_name": response.get("displayName"),
                "percent_complete": job_progress.get("percentComplete", 0),
            },
        )

    async def _create_fireworks_dataset(
        self, dataset_id: str, session: aiohttp.ClientSession
    ) -> None:
        """Create a dataset entry in Fireworks.

        Args:
            dataset_id: Unique identifier for the dataset
            session: aiohttp session to use
        """
        base_url = self._get_batch_api_base_url()
        headers = self._get_fireworks_request_headers()

        async with session.post(
            f"{base_url}/datasets",
            json={
                "datasetId": dataset_id,
                "dataset": {"userUploaded": {}},
            },
            headers=headers,
        ) as response:
            if response.status not in (200, 201):
                error_text = await response.text()
                raise RuntimeError(f"Failed to create dataset: {error_text}")

    async def _upload_to_fireworks_dataset(
        self,
        dataset_id: str,
        content: bytes,
        session: aiohttp.ClientSession,
    ) -> None:
        """Upload content to a Fireworks dataset.

        Args:
            dataset_id: The dataset ID to upload to
            content: The file content as bytes
            session: aiohttp session to use
        """
        base_url = self._get_batch_api_base_url()
        headers = self._get_fireworks_request_headers()
        # Remove Content-Type for multipart upload
        upload_headers = {"Authorization": headers["Authorization"]}

        # Use multipart form data for file upload
        form = aiohttp.FormData()
        form.add_field(
            "file",
            content,
            filename="batch_input.jsonl",
            content_type="application/jsonl",
        )

        async with session.post(
            f"{base_url}/datasets/{dataset_id}:upload",
            data=form,
            headers=upload_headers,
        ) as response:
            if response.status not in (200, 201):
                error_text = await response.text()
                raise RuntimeError(f"Failed to upload to dataset: {error_text}")

    async def _delete_fireworks_dataset(
        self, dataset_id: str, session: aiohttp.ClientSession
    ) -> None:
        """Delete a Fireworks dataset.

        Args:
            dataset_id: The dataset ID to delete
            session: aiohttp session to use
        """
        base_url = self._get_batch_api_base_url()
        headers = self._get_fireworks_request_headers()

        async with session.delete(
            f"{base_url}/datasets/{dataset_id}",
            headers=headers,
        ):
            # Ignore errors on cleanup
            pass

    async def _download_fireworks_dataset(
        self, dataset_id: str, session: aiohttp.ClientSession
    ) -> str:
        """Download content from a Fireworks dataset.

        Args:
            dataset_id: The dataset ID to download from
            session: aiohttp session to use

        Returns:
            str: The dataset content
        """
        base_url = self._get_batch_api_base_url()
        headers = self._get_fireworks_request_headers()

        # First get the download endpoint (uses GET, not POST)
        async with session.get(
            f"{base_url}/datasets/{dataset_id}:getDownloadEndpoint",
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to get download endpoint: {error_text}")
            data = await response.json()
            # Response contains filenameToSignedUrls mapping
            signed_urls = data.get("filenameToSignedUrls", {})
            # Get the results file URL (BIJOutputSet.jsonl, not error-data)
            download_url = None
            for filename, url in signed_urls.items():
                if "error" not in filename.lower() and filename.endswith(".jsonl"):
                    download_url = url
                    break
            if not download_url and signed_urls:
                # Fallback to first available URL
                download_url = next(iter(signed_urls.values()))

        if not download_url:
            raise RuntimeError("No download URL returned from Fireworks")

        # Download the actual content
        async with session.get(download_url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to download dataset: {error_text}")
            return await response.text()

    #
    # Batch API public methods
    #

    @override
    def infer_batch(
        self,
        conversations: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> str:
        """Creates a new batch inference job using the Fireworks Batch API.

        The Fireworks batch API processes requests asynchronously at 50% lower cost.
        Results can be retrieved within 24 hours.

        Requires FIREWORKS_ACCOUNT_ID environment variable to be set.

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
            self._create_fireworks_batch(conversations, generation_params, model_params)
        )

    async def _create_fireworks_batch(
        self,
        conversations: list[Conversation],
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> str:
        """Creates a new batch job with the Fireworks API.

        Args:
            conversations: List of conversations to process in batch
            generation_params: Generation parameters
            model_params: Model parameters

        Returns:
            str: The batch job ID
        """
        # Generate unique dataset IDs
        batch_uuid = str(uuid.uuid4())[:8]
        input_dataset_id = f"oumi-batch-input-{batch_uuid}"
        output_dataset_id = f"oumi-batch-output-{batch_uuid}"

        # Prepare batch requests in Fireworks JSONL format
        lines = []
        for i, conv in enumerate(conversations):
            api_input = self._convert_conversation_to_api_input(
                conv, generation_params, model_params
            )
            # Remove model from body as it's specified at job level
            api_input.pop("model", None)
            request = {
                "custom_id": f"request-{i}",
                "body": api_input,
            }
            lines.append(json.dumps(request))
        content = "\n".join(lines).encode("utf-8")

        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create input dataset (output dataset is created by the batch job)
            await self._create_fireworks_dataset(input_dataset_id, session)

            # Upload input data
            await self._upload_to_fireworks_dataset(input_dataset_id, content, session)

            # Create batch inference job
            base_url = self._get_batch_api_base_url()
            headers = self._get_fireworks_request_headers()
            account_id = self._get_account_id()

            # Fireworks expects full resource paths for dataset IDs
            input_dataset_path = f"accounts/{account_id}/datasets/{input_dataset_id}"
            output_dataset_path = f"accounts/{account_id}/datasets/{output_dataset_id}"

            # Note: Don't add inferenceParameters here - they're already in each
            # request body from _convert_conversation_to_api_input. Adding them
            # at job level would cause "cannot specify both max_tokens and
            # max_completion_tokens" errors.
            job_request: dict[str, Any] = {
                "model": model_params.model_name,
                "inputDatasetId": input_dataset_path,
                "outputDatasetId": output_dataset_path,
                "displayName": f"oumi-batch-{batch_uuid}",
            }

            async with session.post(
                f"{base_url}/batchInferenceJobs",
                json=job_request,
                headers=headers,
            ) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to create batch job: {error_text}")
                data = await response.json()
                # Extract job ID from the full resource name
                job_name = data.get("name", "")
                job_id = job_name.split("/")[-1] if "/" in job_name else job_name
                return job_id

    @override
    def get_batch_status(self, batch_id: str) -> BatchInfo:
        """Gets the status of a batch inference job.

        Args:
            batch_id: The batch job ID

        Returns:
            BatchInfo: Current status of the batch job
        """
        return safe_asyncio_run(self._get_fireworks_batch_status(batch_id))

    async def _get_fireworks_batch_status(self, batch_id: str) -> BatchInfo:
        """Gets the status of a batch job from the Fireworks API.

        Args:
            batch_id: ID of the batch job

        Returns:
            BatchInfo: Current status of the batch job
        """
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            base_url = self._get_batch_api_base_url()
            headers = self._get_fireworks_request_headers()

            async with session.get(
                f"{base_url}/batchInferenceJobs/{batch_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to get batch status: {error_text}")
                data = await response.json()
                return self._convert_fireworks_job_to_batch_info(data)

    @override
    def list_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
    ) -> BatchListResponse:
        """Lists batch jobs.

        Args:
            after: Cursor for pagination (page token)
            limit: Maximum number of batches to return (1-200)

        Returns:
            BatchListResponse: List of batch jobs
        """
        return safe_asyncio_run(self._list_fireworks_batches(after=after, limit=limit))

    async def _list_fireworks_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
    ) -> BatchListResponse:
        """Lists batch jobs from the Fireworks API.

        Args:
            after: Cursor for pagination (page token)
            limit: Maximum number of batches to return (1-200)

        Returns:
            BatchListResponse: List of batch jobs
        """
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            base_url = self._get_batch_api_base_url()
            headers = self._get_fireworks_request_headers()

            params: dict[str, str] = {}
            if after:
                params["pageToken"] = after
            if limit:
                params["pageSize"] = str(min(limit, 200))

            async with session.get(
                f"{base_url}/batchInferenceJobs",
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to list batches: {error_text}")
                data = await response.json()

                batches = [
                    self._convert_fireworks_job_to_batch_info(job_data)
                    for job_data in data.get("batchInferenceJobs", [])
                ]

                return BatchListResponse(
                    batches=batches,
                    first_id=batches[0].id if batches else None,
                    last_id=batches[-1].id if batches else None,
                    has_more=bool(data.get("nextPageToken")),
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
            self._get_fireworks_batch_results(batch_id, conversations)
        )

    async def _get_fireworks_batch_results(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """Gets the results of a completed batch job from the Fireworks API.

        Args:
            batch_id: ID of the batch job
            conversations: Original conversations used to create the batch

        Returns:
            List[Conversation]: The processed conversations with responses

        Raises:
            RuntimeError: If batch status is not completed or if there are errors
        """
        # Get batch status first
        batch_info = await self._get_fireworks_batch_status(batch_id)

        if not batch_info.is_terminal:
            raise RuntimeError(
                f"Batch is not in terminal state. Status: {batch_info.status}"
            )

        if batch_info.status == BatchStatus.FAILED:
            raise RuntimeError(f"Batch failed: {batch_info.error}")

        if batch_info.status == BatchStatus.CANCELLED:
            raise RuntimeError("Batch was cancelled")

        # Get output dataset ID from metadata (may be full path or just ID)
        output_dataset_path = (
            batch_info.metadata.get("output_dataset_id")
            if batch_info.metadata
            else None
        )
        if not output_dataset_path:
            raise RuntimeError("No output dataset ID available")

        # Extract just the dataset ID if it's a full path
        # Path format: accounts/{account_id}/datasets/{dataset_id}
        if "/" in output_dataset_path:
            output_dataset_id = output_dataset_path.split("/")[-1]
        else:
            output_dataset_id = output_dataset_path

        # Download results from output dataset
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            results_content = await self._download_fireworks_dataset(
                output_dataset_id, session
            )

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

            # Check for errors in the result
            if result.get("error"):
                raise RuntimeError(
                    f"Batch request {custom_id} failed: {result['error']}"
                )

            # Extract the response - Fireworks puts the response directly
            # under "response" (unlike OpenAI batch which uses response.body)
            response_body = result.get("response", {})
            processed_conv = self._convert_api_output_to_conversation(
                response_body, conv
            )
            processed_conversations.append(processed_conv)

        return processed_conversations

    @override
    def cancel_batch(self, batch_id: str) -> BatchInfo:
        """Cancels a batch inference job.

        Batches may be canceled if they are queued, pending, or running.

        Args:
            batch_id: The batch job ID to cancel

        Returns:
            BatchInfo: Updated status of the batch job
        """
        return safe_asyncio_run(self._cancel_fireworks_batch(batch_id))

    async def _cancel_fireworks_batch(self, batch_id: str) -> BatchInfo:
        """Cancels a batch job via the Fireworks API.

        Args:
            batch_id: ID of the batch job to cancel

        Returns:
            BatchInfo: Updated status of the batch job
        """
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            base_url = self._get_batch_api_base_url()
            headers = self._get_fireworks_request_headers()

            async with session.post(
                f"{base_url}/batchInferenceJobs/{batch_id}:cancel",
                json={},
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to cancel batch: {error_text}")

            # Get updated status
            return await self._get_fireworks_batch_status(batch_id)
