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
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import (
    BatchInfo,
    BatchListResponse,
    BatchResult,
    BatchStatus,
    RemoteInferenceEngine,
)
from oumi.utils.logging import logger


class FireworksInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Fireworks AI API.

    For batch inference, this engine requires the FIREWORKS_ACCOUNT_ID environment
    variable to be set in addition to FIREWORKS_API_KEY.
    """

    account_id_env_varname: str = "FIREWORKS_ACCOUNT_ID"
    """Environment variable name for the Fireworks account ID."""

    _FIREWORKS_STATE_MAPPING: dict[str, BatchStatus] = {
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
    """Mapping from Fireworks job states to BatchStatus."""

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

    @override
    def _get_request_headers(
        self, remote_params: RemoteParams | None
    ) -> dict[str, str]:
        """Get request headers for Fireworks API calls."""
        api_key = self._get_api_key(remote_params or self._remote_params)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_resource_id(resource_path: str) -> str:
        """Extract the ID from a Fireworks resource path.

        Args:
            resource_path: Full path like 'accounts/x/datasets/y' or just 'y'

        Returns:
            str: The extracted resource ID (last segment of the path)
        """
        return resource_path.split("/")[-1] if "/" in resource_path else resource_path

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
        if state.startswith("JOB_STATE_"):
            state = state[len("JOB_STATE_") :]
        status = self._FIREWORKS_STATE_MAPPING.get(state, BatchStatus.IN_PROGRESS)

        # Extract progress information (jobProgress can be None)
        job_progress = response.get("jobProgress") or {}
        total_requests = job_progress.get("totalRequests", 0)
        processed_requests = job_progress.get("processedRequests", 0)
        failed_requests = job_progress.get("failedRequests", 0)

        # Extract job ID from full resource name (accounts/{id}/batchInferenceJobs/{id})
        job_id = self._extract_resource_id(response.get("name", ""))

        return BatchInfo(
            id=job_id,
            status=status,
            total_requests=total_requests,
            completed_requests=processed_requests - failed_requests,
            failed_requests=failed_requests,
            endpoint="/v1/chat/completions",
            created_at=self._parse_iso_timestamp(response.get("createTime")),
            in_progress_at=self._parse_iso_timestamp(response.get("startTime")),
            completed_at=self._parse_iso_timestamp(response.get("endTime")),
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
        self, dataset_id: str, example_count: int, session: aiohttp.ClientSession
    ) -> None:
        """Create a dataset entry in Fireworks.

        Args:
            dataset_id: Unique identifier for the dataset
            example_count: Number of examples in the dataset
            session: aiohttp session to use
        """
        base_url = self._get_batch_api_base_url()
        headers = self._get_request_headers(self._remote_params)

        async with session.post(
            f"{base_url}/datasets",
            json={
                "datasetId": dataset_id,
                "dataset": {
                    "userUploaded": {},
                    "example_count": example_count,
                },
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
        headers = self._get_request_headers(self._remote_params)
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
        headers = self._get_request_headers(self._remote_params)

        async with session.delete(
            f"{base_url}/datasets/{dataset_id}",
            headers=headers,
        ) as response:
            if response.status not in (200, 204):
                error_text = await response.text()
                logger.warning(
                    f"Failed to delete Fireworks dataset {dataset_id}: {error_text}"
                )

    async def _get_fireworks_dataset_urls(
        self, dataset_id: str, session: aiohttp.ClientSession
    ) -> dict[str, str]:
        """Get signed download URLs for all files in a Fireworks dataset.

        Args:
            dataset_id: The dataset ID
            session: aiohttp session to use

        Returns:
            Dict mapping filename to signed URL.
        """
        base_url = self._get_batch_api_base_url()
        headers = self._get_request_headers(self._remote_params)

        async with session.get(
            f"{base_url}/datasets/{dataset_id}:getDownloadEndpoint",
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to get download endpoint: {error_text}")
            data = await response.json()
            return data.get("filenameToSignedUrls", {})

    async def _download_fireworks_file(
        self, url: str, session: aiohttp.ClientSession
    ) -> str:
        """Download content from a signed URL.

        Args:
            url: The signed URL to download from
            session: aiohttp session to use

        Returns:
            str: The file content
        """
        async with session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to download file: {error_text}")
            return await response.text()

    async def _download_fireworks_dataset(
        self, dataset_id: str, session: aiohttp.ClientSession
    ) -> str:
        """Download the results file from a Fireworks dataset.

        Args:
            dataset_id: The dataset ID to download from
            session: aiohttp session to use

        Returns:
            str: The dataset content (results file only, not errors)
        """
        signed_urls = await self._get_fireworks_dataset_urls(dataset_id, session)

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

        return await self._download_fireworks_file(download_url, session)

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
            await self._create_fireworks_dataset(
                input_dataset_id, len(conversations), session
            )

            try:
                # Upload input data
                await self._upload_to_fireworks_dataset(
                    input_dataset_id, content, session
                )

                # Create batch inference job
                base_url = self._get_batch_api_base_url()
                headers = self._get_request_headers(self._remote_params)
                account_id = self._get_account_id()

                # Fireworks expects full resource paths for dataset IDs
                input_dataset_path = (
                    f"accounts/{account_id}/datasets/{input_dataset_id}"
                )
                output_dataset_path = (
                    f"accounts/{account_id}/datasets/{output_dataset_id}"
                )

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
                    return self._extract_resource_id(data.get("name", ""))
            except Exception:
                # Clean up the input dataset if batch creation fails
                await self._delete_fireworks_dataset(input_dataset_id, session)
                raise

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
        base_url = self._get_batch_api_base_url()
        async with self._create_session() as (session, headers):
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
        base_url = self._get_batch_api_base_url()
        async with self._create_session() as (session, headers):
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
        """Gets partial results of a completed Fireworks batch job."""
        return safe_asyncio_run(
            self._get_fireworks_batch_results_partial(batch_id, conversations)
        )

    async def _get_fireworks_batch_results_partial(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> BatchResult:
        """Gets partial results of a completed Fireworks batch job."""
        batch_info = await self._get_fireworks_batch_status(batch_id)

        if not batch_info.is_terminal:
            raise RuntimeError(
                f"Batch is not in terminal state. Status: {batch_info.status}"
            )

        if batch_info.status in (
            BatchStatus.FAILED,
            BatchStatus.EXPIRED,
            BatchStatus.CANCELLED,
        ):
            raise RuntimeError(
                f"Batch is unrecoverably {batch_info.status.value}: "
                f"error={batch_info.error}"
            )

        output_dataset_path = (
            batch_info.metadata.get("output_dataset_id")
            if batch_info.metadata
            else None
        )
        if not output_dataset_path:
            raise RuntimeError("No output dataset ID available")

        output_dataset_id = self._extract_resource_id(output_dataset_path)

        logger.info(
            f"Batch {batch_id}: retrieving partial results "
            f"(status={batch_info.status.value}, "
            f"total={len(conversations)} requests, "
            f"dataset={output_dataset_id})"
        )

        # Fireworks output dataset contains two files: results and errors.
        # Download both from the dataset's signed URLs.
        successful: list[tuple[int, Conversation]] = []
        failed_indices: list[int] = []
        error_messages: dict[int, str] = {}

        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            signed_urls = await self._get_fireworks_dataset_urls(
                output_dataset_id, session
            )
            logger.info(
                f"Batch {batch_id}: output dataset has "
                f"{len(signed_urls)} files: {list(signed_urls.keys())}"
            )

            results_url = None
            error_url = None
            for filename, url in signed_urls.items():
                if "error" in filename.lower():
                    error_url = url
                elif filename.endswith(".jsonl"):
                    results_url = url

            # Parse results file (successful responses)
            if results_url:
                results_content = await self._download_fireworks_file(
                    results_url, session
                )
                for line in results_content.strip().splitlines():
                    if not line:
                        continue
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    try:
                        idx = int(custom_id.split("-", 1)[1])
                    except (IndexError, ValueError):
                        continue

                    try:
                        response_body = result.get("response", {})
                        conv = self._convert_api_output_to_conversation(
                            response_body, conversations[idx]
                        )
                        successful.append((idx, conv))
                    except Exception as e:
                        failed_indices.append(idx)
                        error_messages[idx] = f"Failed to parse response: {e}"

            # Parse error file (failed requests)
            if error_url:
                error_content = await self._download_fireworks_file(error_url, session)
                for line in error_content.strip().splitlines():
                    if not line:
                        continue
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    try:
                        idx = int(custom_id.split("-", 1)[1])
                    except (IndexError, ValueError):
                        continue

                    failed_indices.append(idx)
                    error_msg = result.get("error", {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                    error_messages[idx] = str(error_msg)

        logger.info(
            f"Batch {batch_id}: {len(successful)} succeeded, "
            f"{len(failed_indices)} failed out of {len(conversations)} total"
        )
        if error_messages:
            for idx, msg in error_messages.items():
                logger.warning(f"Batch {batch_id} request {idx} failed: {msg}")

        return BatchResult(
            successful=successful,
            failed_indices=failed_indices,
            error_messages=error_messages,
        )

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
        base_url = self._get_batch_api_base_url()
        async with self._create_session() as (session, headers):
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
