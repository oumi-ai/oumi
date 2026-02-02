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

"""Together AI inference engine implementation."""

import tempfile
from pathlib import Path
from typing import Any

import aiohttp
import jsonlines
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import (
    _BATCH_ENDPOINT,
    BatchInfo,
    RemoteInferenceEngine,
)


class TogetherInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Together AI API.

    Together AI supports batch inference via their batch API. Note that Together
    uses a redirect-based file upload flow that differs from OpenAI's direct
    multipart upload.

    See: https://docs.together.ai/docs/batch-inference
    """

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the Together API."""
        return "https://api.together.xyz/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the Together API key."""
        return "TOGETHER_API_KEY"

    @property
    @override
    def _batch_purpose(self) -> str:
        """Return the purpose value for batch file uploads.

        Together AI uses "batch-api" instead of OpenAI's "batch".
        """
        return "batch-api"

    @override
    async def _upload_batch_file(
        self,
        batch_requests: list[dict],
    ) -> str:
        """Uploads a JSONL file for batch processing using Together's redirect flow.

        Together AI uses a different upload mechanism than OpenAI:
        1. POST to /files with JSON body to get a signed upload URL (302 redirect)
        2. PUT the file content to the signed URL
        3. POST to /files/{file_id}/preprocess to finalize

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
            tmp_path = Path(tmp.name)

        try:
            connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = self._get_request_headers(self._remote_params)

                # Step 1: Request signed upload URL
                # Together expects form-encoded data (not JSON or multipart)
                request_data = {
                    "purpose": self._batch_purpose,
                    "file_name": tmp_path.name,
                    "file_type": "jsonl",
                }

                async with session.post(
                    self.get_file_api_url(),
                    data=request_data,  # Form-encoded, not json=
                    headers=headers,
                    allow_redirects=False,  # We need to handle the redirect manually
                ) as response:
                    if response.status == 302:
                        # Get the signed URL and file ID from headers
                        redirect_url = response.headers.get("Location")
                        file_id = response.headers.get("X-Together-File-Id")

                        if not redirect_url or not file_id:
                            raise RuntimeError(
                                "Together API did not return redirect URL or file ID. "
                                f"Headers: {dict(response.headers)}"
                            )
                    elif response.status == 200:
                        # Some endpoints might return the file directly
                        data = await response.json()
                        return data["id"]
                    else:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Failed to get upload URL from Together: {error_text}"
                        )

                # Step 2: Upload file content to signed URL
                file_content = tmp_path.read_bytes()

                async with session.put(
                    redirect_url,
                    data=file_content,
                ) as upload_response:
                    if upload_response.status not in (200, 201):
                        error_text = await upload_response.text()
                        raise RuntimeError(
                            f"Failed to upload file to Together: {error_text}"
                        )

                # Step 3: Finalize upload by calling preprocess endpoint
                preprocess_url = f"{self.get_file_api_url()}/{file_id}/preprocess"
                async with session.post(
                    preprocess_url,
                    headers=self._get_request_headers(self._remote_params),
                ) as preprocess_response:
                    if preprocess_response.status != 200:
                        error_text = await preprocess_response.text()
                        raise RuntimeError(
                            f"Failed to preprocess file on Together: {error_text}"
                        )

                return file_id

        finally:
            # Clean up temporary file
            tmp_path.unlink()

    def _normalize_together_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize Together's response format to match OpenAI's format.

        Together uses uppercase status values
        (e.g., "COMPLETED" instead of "completed").
        """
        if "status" in data:
            data["status"] = data["status"].lower()
        return data

    def _parse_batch_create_response(
        self, response: aiohttp.ClientResponse, data: dict[str, Any]
    ) -> str:
        """Parse batch creation response.

        Together returns 201 and {"job": {"id": "..."}} instead of OpenAI's
        200 and {"id": "..."}.
        """
        if response.status not in (200, 201):
            raise RuntimeError(f"Unexpected status code: {response.status}")
        if "job" in data:
            return data["job"]["id"]
        return data["id"]

    @override
    async def _create_batch(
        self,
        conversations: list[Conversation],
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> str:
        """Creates a batch job, handling Together's response format."""
        # Prepare batch requests
        batch_requests = []
        for i, conv in enumerate(conversations):
            api_input = self._convert_conversation_to_api_input(
                conv, generation_params, model_params
            )
            batch_requests.append(
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": _BATCH_ENDPOINT,
                    "body": api_input,
                }
            )

        # Upload batch file (uses Together's redirect-based flow)
        file_id = await self._upload_batch_file(batch_requests)

        # Create batch
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.post(
                self.get_batch_api_url(),
                json={
                    "input_file_id": file_id,
                    "endpoint": _BATCH_ENDPOINT,
                    "completion_window": self._remote_params.batch_completion_window,
                },
                headers=headers,
            ) as response:
                if response.status not in (200, 201):
                    raise RuntimeError(
                        f"Failed to create batch: {await response.text()}"
                    )
                data = await response.json()
                return self._parse_batch_create_response(response, data)

    @override
    async def _get_batch_status(self, batch_id: str) -> BatchInfo:
        """Gets the status of a batch job, normalizing Together's response format."""
        connector = aiohttp.TCPConnector(limit=self._get_connection_limit())
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self.get_batch_api_url()}/{batch_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to get batch status: {await response.text()}"
                    )
                data = await response.json()
                return BatchInfo.from_api_response(
                    self._normalize_together_response(data)
                )
