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

"""Fireworks.ai deployment client."""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import requests as _requests

from oumi.deploy.base_client import (
    AutoscalingConfig,
    BaseDeploymentClient,
    DeploymentProvider,
    Endpoint,
    EndpointState,
    HardwareConfig,
    Model,
    ModelType,
    UploadedModel,
)

logger = logging.getLogger(__name__)

# Mapping from Fireworks accelerator names to our standard format
FIREWORKS_ACCELERATORS = {
    "nvidia_a100_80gb": "NVIDIA_A100_80GB",
    "nvidia_h100_80gb": "NVIDIA_H100_80GB",
    "nvidia_h200_141gb": "NVIDIA_H200_141GB",
    "amd_mi300x_192gb": "AMD_MI300X_192GB",
}

# Reverse mapping
FIREWORKS_ACCELERATORS_REVERSE = {v: k for k, v in FIREWORKS_ACCELERATORS.items()}

# Hardcoded hardware list version
# (Fireworks has no public hardware discovery API as of 2026-01)
FIREWORKS_HARDWARE_LIST_VERSION = "2026-01"

# Mapping from Fireworks deployment states to our EndpointState enum
FIREWORKS_STATE_MAP = {
    "PENDING": EndpointState.PENDING,
    "CREATING": EndpointState.STARTING,
    "READY": EndpointState.RUNNING,
    "RUNNING": EndpointState.RUNNING,
    "DELETING": EndpointState.STOPPING,
    "DELETED": EndpointState.STOPPED,
    "FAILED": EndpointState.ERROR,
    "ERROR": EndpointState.ERROR,
}


def _raise_api_error(response: httpx.Response, context: str) -> None:
    """Extract the human-readable message from a Fireworks error response and raise.

    Fireworks returns JSON bodies on errors.  Common shapes::

        {"error": {"message": "...", "code": "INVALID_ARGUMENT", ...}}
        {"message": "...", "code": 400}

    Args:
        response: The failed HTTP response.
        context: Short description of the operation that failed (used in the message).

    Raises:
        ValueError: Always, with a message that includes the API error detail,
            HTTP status code, the original request method + URL, and the
            request body (for payload inspection during debugging).
    """
    detail: str
    try:
        body = response.json()
        if isinstance(body, dict):
            if "error" in body:
                err = body["error"]
                detail = (
                    err.get("message", str(err)) if isinstance(err, dict) else str(err)
                )
            else:
                detail = body.get("message", str(body))
        else:
            detail = str(body)
    except Exception:
        detail = response.text or "(no details)"

    req = response.request
    try:
        req_body = req.content.decode("utf-8", errors="replace") or "(empty)"
    except Exception:
        req_body = "(unreadable)"
    raise ValueError(
        f"Failed to {context}: {detail} "
        f"(HTTP {response.status_code}, {req.method} {req.url} — "
        f"request body: {req_body})"
    )


class FireworksDeploymentClient(BaseDeploymentClient):
    """Fireworks.ai deployment client (async).

    API Reference: https://docs.fireworks.ai/api-reference

    Authentication: Bearer token via FIREWORKS_API_KEY env var.
    Account ID: Via FIREWORKS_ACCOUNT_ID env var or constructor.

    Accelerator types: NVIDIA_A100_80GB, NVIDIA_H100_80GB, NVIDIA_H200_141GB, AMD_MI300X

    Upload flow: create model -> get signed URL -> PUT tar.gz -> validate -> prepare
    """

    BASE_URL = "https://api.fireworks.ai"
    provider = DeploymentProvider.FIREWORKS

    def __init__(self, api_key: str | None = None, account_id: str | None = None):
        """Initializes the Fireworks.ai deployment client.

        Args:
            api_key: Fireworks API key. If not provided, reads from
                     FIREWORKS_API_KEY environment variable.
            account_id: Fireworks account ID. If not provided, reads from
                        FIREWORKS_ACCOUNT_ID environment variable.
        """
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks API key must be provided "
                "or set via FIREWORKS_API_KEY env var"
            )

        self.account_id = account_id or os.environ.get("FIREWORKS_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError(
                "Fireworks account ID must be provided or set via "
                "FIREWORKS_ACCOUNT_ID env var"
            )

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    async def __aenter__(self) -> "FireworksDeploymentClient":  # type: ignore[override]
        """Enters the async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the async context manager and closes the HTTP client."""
        await self._client.aclose()

    def _get_inference_auth_headers(self) -> dict[str, str]:
        """Returns auth headers for inference (test_endpoint)."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _to_fireworks_accelerator(self, hw: HardwareConfig) -> str:
        """Converts HardwareConfig accelerator to Fireworks format.

        Args:
            hw: HardwareConfig with accelerator name

        Returns:
            Fireworks accelerator type string
        """
        result = FIREWORKS_ACCELERATORS.get(hw.accelerator)
        return result if result is not None else hw.accelerator.upper()

    def _from_fireworks_accelerator(self, accelerator: str) -> str:
        """Converts Fireworks accelerator to our standard format.

        Args:
            accelerator: Fireworks accelerator type string

        Returns:
            Standard accelerator name
        """
        result = FIREWORKS_ACCELERATORS_REVERSE.get(accelerator)
        return result if result is not None else accelerator.lower()

    def _parse_deployment(self, data: dict[str, Any]) -> Endpoint:
        """Parses Fireworks deployment response into Endpoint dataclass."""
        state_str = data.get("state", "PENDING").upper()
        state = FIREWORKS_STATE_MAP.get(state_str, EndpointState.PENDING)

        # Parse hardware config — these are top-level fields on gatewayDeployment
        accelerator = data.get("acceleratorType", "")
        count = data.get("acceleratorCount", 1)
        hardware = HardwareConfig(
            accelerator=self._from_fireworks_accelerator(accelerator),
            count=count,
        )

        # Parse autoscaling — spec field names are minReplicaCount / maxReplicaCount
        min_replicas = data.get("minReplicaCount", 0)
        max_replicas = data.get("maxReplicaCount", 1)
        autoscaling = AutoscalingConfig(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )

        # Parse created time
        created_at = None
        if data.get("createTime"):
            try:
                created_at = datetime.fromisoformat(
                    data["createTime"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        # Extract deployment ID from name (accounts/{account}/deployments/{id})
        name = data.get("name", "")
        endpoint_id = name.split("/")[-1] if "/" in name else name

        return Endpoint(
            endpoint_id=endpoint_id,
            provider=DeploymentProvider.FIREWORKS,
            model_id=data.get("baseModel", ""),
            endpoint_url=data.get("endpointUrl"),
            state=state,
            hardware=hardware,
            autoscaling=autoscaling,
            created_at=created_at,
            display_name=data.get("displayName"),
        )

    # Validation retry settings.
    # The Fireworks REST API docs call validateUpload immediately after
    # uploading files (no propagation delay).  We wait briefly before the
    # first attempt so GCS can propagate (avoids "config.json not found");
    # then retry with back-off for transient errors.
    VALIDATION_INITIAL_DELAY_S: float = 15.0
    VALIDATION_MAX_RETRIES: int = 3
    VALIDATION_RETRY_DELAY_S: int = 10

    # Per-file upload settings.
    UPLOAD_MAX_RETRIES: int = 5
    UPLOAD_INITIAL_BACKOFF_S: float = 2.0
    UPLOAD_BACKOFF_FACTOR: float = 2.0
    UPLOAD_MAX_BACKOFF_S: float = 60.0
    UPLOAD_TIMEOUT_S: float = 600.0  # per-request timeout

    # ------------------------------------------------------------------
    # upload_model — orchestrator
    # ------------------------------------------------------------------

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        progress_callback: Any | None = None,
    ) -> UploadedModel:
        """Uploads a model to Fireworks.ai using multi-step flow.

        API Flow (from https://docs.fireworks.ai/models/uploading-custom-models-api):
        1. Create model object with modelId and model structure
        2. Get signed upload URLs (one per file) with file sizes
        3. Upload each file to its signed URL
        4. Wait for GCS propagation then validate the upload

        Args:
            model_source: Path to model directory or cloud storage URL (e.g., s3://bucket/path)
            model_name: Model ID to use (e.g., "my-custom-model")
            model_type: Type of model (FULL or ADAPTER)
            base_model: Base model for LoRA adapters
            progress_callback: Optional async callback for progress updates.
                Signature: async def callback(stage: str, message: str, details: dict)

        Returns:
            UploadedModel with provider-specific model ID
        """
        # Sanitize model name to create a valid model ID
        base_id = model_name.lower().replace(" ", "-")
        base_id = "".join(c for c in base_id if c.isalnum() or c in "-_")
        model_id = f"{base_id}-{int(time.time())}"

        # Step 1: Create model resource on Fireworks
        create_payload = await self._create_model_resource(
            model_id, model_type, base_model, progress_callback
        )

        # Steps 2–4: Resolve source, upload files, validate
        temp_dir = None
        try:
            model_dir, temp_dir = await self._resolve_model_source(
                model_source, progress_callback
            )
            await self._upload_model_files(model_dir, model_id, progress_callback)
            await self._wait_and_validate(model_id, progress_callback)
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return UploadedModel(
            provider_model_id=f"accounts/{self.account_id}/models/{model_id}",
            status="validating",
            request_payload=create_payload,
        )

    # ------------------------------------------------------------------
    # upload_model — private helpers
    # ------------------------------------------------------------------

    async def _create_model_resource(
        self,
        model_id: str,
        model_type: ModelType,
        base_model: str | None,
        progress_callback: Any | None,
    ) -> dict[str, Any]:
        """Creates a model resource on Fireworks and returns the create payload."""
        logger.info(
            "Creating model on Fireworks: model_id=%s, model_type=%s, base_model=%s",
            model_id,
            model_type,
            base_model,
        )

        if model_type == ModelType.ADAPTER and base_model:
            # HF_PEFT_ADDON is the correct kind for LoRA adapters per the
            # gatewayModelKind enum in the OpenAPI spec
            # (fireworks.openapi.yaml).  peftDetails (baseModel, r,
            # targetModules) is required for this kind.
            create_payload: dict[str, Any] = {
                "modelId": model_id,
                "model": {
                    "kind": "HF_PEFT_ADDON",
                    "peftDetails": {
                        "baseModel": base_model,
                        "r": 8,
                        "targetModules": [],
                    },
                },
            }
        else:
            # CUSTOM_MODEL is the only kind that supports the client-side
            # presigned-URL upload flow used by this client.  HF_BASE_MODEL
            # exists in the gatewayModelKind enum (fireworks.openapi.yaml) but
            # is intended for models that Fireworks imports server-side; the
            # :getUploadEndpoint API rejects file manifests for HF_BASE_MODEL
            # resources with INVALID_ARGUMENT ("unexpected file").
            #
            # Reference: https://docs.fireworks.ai/models/uploading-custom-models-api
            # The official REST upload example uses kind=CUSTOM_MODEL with
            # checkpointFormat=HUGGINGFACE.
            create_payload = {
                "modelId": model_id,
                "model": {
                    "kind": "CUSTOM_MODEL",
                    "baseModelDetails": {
                        "checkpointFormat": "HUGGINGFACE",
                        "worldSize": 1,
                    },
                },
            }

        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/models",
            json=create_payload,
        )

        if response.is_error:
            logger.error(
                "Fireworks API error response (HTTP %d): %s",
                response.status_code,
                response.text,
            )
            if response.status_code == 409:
                logger.error(
                    "Model ID '%s' already exists. "
                    "Delete it manually or use a different name.",
                    model_id,
                )
            _raise_api_error(response, context=f"create model resource '{model_id}'")

        created_model_name = response.json().get("name", "")
        logger.info("Model created: %s", created_model_name)

        if progress_callback:
            await progress_callback(
                "creating",
                f"Model resource created on Fireworks: {model_id}",
                {"provider_model_id": created_model_name},
            )
        return create_payload

    @staticmethod
    def _validate_local_model_path(model_source: str) -> Path:
        """Validates that *model_source* is an existing directory.

        Returns:
            A ``Path`` pointing to the model directory.

        Raises:
            ValueError: If the path does not exist or is not a directory.
        """
        model_dir = Path(model_source)
        if not model_dir.exists():
            raise ValueError(f"Model source does not exist: {model_source}")
        if not model_dir.is_dir():
            raise ValueError(f"Model source must be a directory: {model_source}")
        return model_dir

    async def _resolve_model_source(
        self,
        model_source: str,
        progress_callback: Any | None,
    ) -> tuple[Path, Path | None]:
        """Resolves model_source to a local directory.

        Supports:
        - Local directory paths
        - HTTP/HTTPS URLs (presigned URLs)
        - HuggingFace repository IDs (e.g., "Qwen/Qwen3-4B")

        Returns:
            (model_dir, temp_dir) — temp_dir is non-None when a temporary
            directory was created and must be cleaned up by the caller.
        """
        if model_source.startswith(("http://", "https://")):
            return await self._download_and_extract(model_source, progress_callback)
        if model_source.startswith(("s3://", "gs://")):
            raise NotImplementedError(
                "Direct cloud storage paths not supported. "
                "Convert to a presigned URL first. "
                f"Model source: {model_source}"
            )

        # Check if it's a local path first — use executor to avoid blocking the
        # event loop with synchronous filesystem calls (ruff ASYNC240).
        model_path = Path(model_source)
        loop = asyncio.get_event_loop()
        path_exists = await loop.run_in_executor(None, model_path.exists)
        path_is_dir = await loop.run_in_executor(None, model_path.is_dir)
        if path_exists and path_is_dir:
            return self._validate_local_model_path(model_source), None

        # If not a local path and contains "/" (but not absolute path),
        # treat as HuggingFace repo ID
        if "/" in model_source and not model_source.startswith("/"):
            logger.info(
                "Model source appears to be a HuggingFace repo ID: %s", model_source
            )
            return await self._download_from_huggingface(
                model_source, progress_callback
            )

        # If we get here, it's neither a valid local path nor a HF repo
        raise ValueError(
            f"Model source does not exist and is not a valid HuggingFace repo ID: "
            f"{model_source}"
        )

    async def _download_from_huggingface(
        self,
        repo_id: str,
        progress_callback: Any | None,
    ) -> tuple[Path, None]:
        """Downloads a model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "Qwen/Qwen3-4B")
            progress_callback: Optional callback for progress updates

        Returns:
            (model_dir, None) — model is cached by HF Hub, no temp dir to clean up.

        Raises:
            ImportError: If huggingface_hub is not installed
            Exception: If download fails
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download models from HuggingFace. "
                "Install it with: pip install huggingface_hub"
            ) from e

        logger.info("Downloading model from HuggingFace Hub: %s", repo_id)
        if progress_callback:
            await progress_callback(
                "downloading",
                f"Downloading model from HuggingFace: {repo_id}",
                {},
            )

        # Download to HF cache (runs in executor since it's blocking)
        loop = asyncio.get_event_loop()
        try:
            cache_path = await loop.run_in_executor(
                None,
                lambda: snapshot_download(repo_id=repo_id, repo_type="model"),
            )
        except Exception as e:
            logger.error("Failed to download model from HuggingFace: %s", e)
            raise ValueError(
                f"Failed to download model from HuggingFace Hub: {repo_id}. Error: {e}"
            ) from e

        model_dir = Path(cache_path)
        logger.info("Model downloaded to: %s", model_dir)

        if progress_callback:
            await progress_callback(
                "downloaded",
                f"Model downloaded from HuggingFace: {repo_id}",
                {"local_path": str(model_dir)},
            )

        # Return None for temp_dir since HF Hub manages the cache
        return model_dir, None

    async def _download_and_extract(
        self,
        url: str,
        progress_callback: Any | None,
    ) -> tuple[Path, Path]:
        """Downloads an archive from *url* and extracts it.

        Returns:
            (extracted_dir, temp_dir) — caller must clean up temp_dir.
        """
        logger.info("Downloading model archive from presigned URL")
        if progress_callback:
            await progress_callback(
                "downloading",
                "Downloading model archive from cloud storage...",
                {},
            )

        temp_dir = Path(tempfile.mkdtemp())
        archive_path = temp_dir / "model_archive"

        async with httpx.AsyncClient(timeout=300.0) as download_client:
            response = await download_client.get(url)
            if response.is_error:
                _raise_api_error(response, context="download model archive")
            async with aiofiles.open(archive_path, "wb") as f:
                await f.write(response.content)

        size_mb = len(response.content) / 1024 / 1024
        logger.info("Downloaded %.1f MB", size_mb)
        if progress_callback:
            await progress_callback(
                "downloading",
                f"Download complete ({size_mb:.1f} MB)",
                {"bytes_downloaded": len(response.content)},
            )

        # Extract
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        if progress_callback:
            await progress_callback("extracting", "Extracting model archive...", {})

        if zipfile.is_zipfile(archive_path):
            logger.info("Extracting ZIP archive")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            raise ValueError("Model source is not a recognized archive format (zip)")

        logger.info("Extracted model to %s", extract_dir)
        return extract_dir, temp_dir

    async def _upload_model_files(
        self,
        model_dir: Path,
        model_id: str,
        progress_callback: Any | None,
    ) -> None:
        """Collects files, obtains signed URLs, and uploads each file.

        Follows the Fireworks REST API upload flow documented at
        https://docs.fireworks.ai/models/uploading-custom-models-api:

        1. Build a ``filenameToSize`` inventory of the model directory.
        2. Call ``getUploadEndpoint`` to obtain per-file signed URLs.
        3. PUT each file to its signed URL (streamed from disk, with
           retry and exponential back-off on transient failures).

        Each upload attempt uses a **fresh** ``httpx.AsyncClient`` to
        avoid stale TLS connections, and file contents are streamed in
        small read-chunks so that large model shards are never held
        entirely in memory.

        Files are filtered to exclude HuggingFace cache artifacts (e.g.
        ``*.lock``, ``*.metadata``) so the backend sees only model files,
        and critical files (e.g. ``config.json``) are uploaded first to
        improve validation success (GCS propagation).
        """
        # Collect file inventory, excluding cache artifacts (match firectl behavior)
        _IGNORED_SUFFIXES = (".lock", ".metadata")
        file_sizes: dict[str, int] = {}
        for root, _, files in os.walk(model_dir):
            for fname in files:
                if fname.endswith(_IGNORED_SUFFIXES):
                    logger.debug("Skipping cache artifact: %s", fname)
                    continue
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(model_dir))
                if ".cache" in rel or "huggingface" in rel.lower():
                    logger.debug("Skipping path under cache: %s", rel)
                    continue

                # Prepend "hf/" to match firectl behavior and backend expectations
                remote_name = f"hf/{rel}"
                file_sizes[remote_name] = fpath.stat().st_size

        total_bytes = sum(file_sizes.values())
        logger.info(
            "Found %d files to upload (%.1f MB)",
            len(file_sizes),
            total_bytes / 1024 / 1024,
        )
        if "hf/config.json" in file_sizes:
            logger.info("config.json found (%d bytes)", file_sizes["hf/config.json"])
        else:
            logger.error(
                "config.json NOT found in model files: %s", list(file_sizes.keys())
            )

        if progress_callback:
            await progress_callback(
                "extracting",
                f"Found {len(file_sizes)} files "
                f"({total_bytes / 1024 / 1024:.1f} MB total)",
                {"file_count": len(file_sizes), "files": list(file_sizes.keys())},
            )

        # Obtain signed upload URLs
        # https://docs.fireworks.ai/api-reference/get-model-upload-endpoint
        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/models/{model_id}:getUploadEndpoint",
            json={"filenameToSize": file_sizes, "enableResumableUpload": False},
        )
        if response.is_error:
            _raise_api_error(
                response,
                context=f"get upload endpoint for model '{model_id}'",
            )
        file_upload_urls = response.json().get("filenameToSignedUrls", {})
        if not file_upload_urls:
            raise ValueError("No upload URLs received from Fireworks API.")

        # Upload in deterministic order: config.json and small config/tokenizer
        # files first so they are visible to validation sooner (GCS propagation).
        def _upload_order(item: tuple[str, str]) -> tuple[int, str]:
            filename = item[0]
            if filename == "hf/config.json":
                return (0, filename)
            if filename == "hf/generation_config.json":
                return (1, filename)
            if filename in (
                "hf/tokenizer_config.json",
                "hf/tokenizer.json",
                "hf/tokenizer.model",
            ):
                return (2, filename)
            if filename.endswith(".index.json") or "tokenizer" in filename.lower():
                return (3, filename)
            return (4, filename)

        upload_items = sorted(file_upload_urls.items(), key=_upload_order)

        # Upload each file
        total_files = len(upload_items)
        uploaded_bytes = 0

        if progress_callback:
            await progress_callback(
                "uploading",
                f"Starting upload of {total_files} files "
                f"({total_bytes / 1024 / 1024:.1f} MB)",
                {"total_files": total_files, "total_bytes": total_bytes},
            )

        for idx, (filename, signed_url) in enumerate(upload_items, 1):
            # Map remote filename (hf/foo.json) back to local path (foo.json)
            # We know we prepended "hf/" in the inventory step.
            local_rel_path = filename
            if filename.startswith("hf/"):
                local_rel_path = filename[3:]

            file_path = model_dir / local_rel_path
            file_size = file_sizes[filename]

            logger.info(
                "[%d/%d] Uploading %s (%.2f MB)",
                idx,
                total_files,
                filename,
                file_size / 1024 / 1024,
            )

            await self._upload_single_file(
                file_path, file_size, signed_url, filename, idx, total_files
            )

            uploaded_bytes += file_size
            logger.info(
                "[%d/%d] Uploaded %s (%.1f / %.1f MB)",
                idx,
                total_files,
                filename,
                uploaded_bytes / 1024 / 1024,
                total_bytes / 1024 / 1024,
            )

            if progress_callback:
                await progress_callback(
                    "uploading",
                    f"Uploaded {filename} ({idx}/{total_files}, "
                    f"{uploaded_bytes / 1024 / 1024:.1f} / "
                    f"{total_bytes / 1024 / 1024:.1f} MB)",
                    {
                        "current_file": filename,
                        "uploaded_count": idx,
                        "total_files": total_files,
                        "uploaded_bytes": uploaded_bytes,
                        "total_bytes": total_bytes,
                    },
                )

        logger.info(
            "All %d files uploaded (%.1f MB total)",
            total_files,
            total_bytes / 1024 / 1024,
        )
        if progress_callback:
            await progress_callback(
                "uploading",
                f"All {total_files} files uploaded "
                f"({total_bytes / 1024 / 1024:.1f} MB total)",
                {"status": "complete", "total_files": total_files},
            )

    # ------------------------------------------------------------------
    # Per-file upload with retry
    # ------------------------------------------------------------------

    async def _upload_single_file(
        self,
        file_path: Path,
        file_size: int,
        signed_url: str,
        filename: str,
        idx: int,
        total_files: int,
    ) -> None:
        """PUTs a single file to its signed URL with retry.

        Uses the ``requests`` library (sync, run in an executor) to
        match the exact upload pattern from the Fireworks REST API docs
        (https://docs.fireworks.ai/models/uploading-custom-models-api).
        This is necessary because:

        * ``httpx`` async sends ``Transfer-Encoding: chunked`` when
          given an async generator, even if ``Content-Length`` is set
          explicitly.  GCS signed-URL PUTs silently discard files
          uploaded with chunked encoding, causing Fireworks validation
          to fail with "config.json not found".
        * ``requests.put(url, data=file_handle)`` auto-detects file
          size via ``seek``/``tell``, sets ``Content-Length`` correctly,
          and streams the file from disk without loading it into memory.

        Raises:
            requests.HTTPError: If all retry attempts are exhausted.
            OSError: If all retry attempts are exhausted due to I/O errors.
        """
        headers = {
            "Content-Type": "application/octet-stream",
            "x-goog-content-length-range": f"{file_size},{file_size}",
        }

        backoff = self.UPLOAD_INITIAL_BACKOFF_S
        loop = asyncio.get_event_loop()

        for attempt in range(1, self.UPLOAD_MAX_RETRIES + 1):
            try:
                await loop.run_in_executor(
                    None,
                    self._sync_put_file,
                    file_path,
                    signed_url,
                    headers,
                )
                return

            except (OSError, _requests.RequestException) as exc:
                if attempt < self.UPLOAD_MAX_RETRIES:
                    logger.warning(
                        "[%d/%d] Upload attempt %d/%d for %s failed (%s: %s). "
                        "Retrying in %.0fs...",
                        idx,
                        total_files,
                        attempt,
                        self.UPLOAD_MAX_RETRIES,
                        filename,
                        type(exc).__name__,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(
                        backoff * self.UPLOAD_BACKOFF_FACTOR,
                        self.UPLOAD_MAX_BACKOFF_S,
                    )
                else:
                    logger.error(
                        "[%d/%d] All %d upload attempts for %s exhausted.",
                        idx,
                        total_files,
                        self.UPLOAD_MAX_RETRIES,
                        filename,
                    )
                    raise

    @staticmethod
    def _sync_put_file(
        file_path: Path,
        signed_url: str,
        headers: dict[str, str],
    ) -> None:
        """Synchronous PUT of a file to a signed URL.

        Opens the file and passes the handle to ``requests.put(data=f)``,
        which streams the content from disk and sets ``Content-Length``
        automatically via ``seek``/``tell``.
        """
        with open(file_path, "rb") as f:
            response = _requests.put(signed_url, data=f, headers=headers, timeout=600)
        if not response.ok:
            logger.error(
                "GCS upload failed (HTTP %d, %s %s): %s",
                response.status_code,
                response.request.method,
                response.request.url,
                response.text or "(no details)",
            )
        response.raise_for_status()

    async def _wait_and_validate(
        self,
        model_id: str,
        progress_callback: Any | None,
    ) -> None:
        """Validates the upload, following the Fireworks REST API flow.

        Per https://docs.fireworks.ai/models/uploading-custom-models-api the
        ``validateUpload`` endpoint is called immediately after uploading all
        files (no propagation delay).  We add a small number of retries with
        short back-off as a safety-net for transient errors.

        Raises:
            ValueError: If validation fails after all retries.
        """
        logger.info("Triggering upload validation...")
        if progress_callback:
            await progress_callback("validating", "Validating uploaded model...", {})

        max_retries = self.VALIDATION_MAX_RETRIES
        retry_delay = self.VALIDATION_RETRY_DELAY_S
        initial_delay = self.VALIDATION_INITIAL_DELAY_S

        for attempt in range(max_retries):
            if attempt == 0 and initial_delay > 0:
                logger.info(
                    "Waiting %.0fs for GCS propagation before first validation...",
                    initial_delay,
                )
                await asyncio.sleep(initial_delay)
            elif attempt > 0:
                logger.info(
                    "Validation retry %d/%d: waiting %ds...",
                    attempt + 1,
                    max_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

            logger.info("Validation attempt %d/%d", attempt + 1, max_retries)
            if progress_callback:
                await progress_callback(
                    "validating",
                    f"Validation attempt {attempt + 1}/{max_retries}...",
                    {"attempt": attempt + 1, "max_retries": max_retries},
                )

            response = await self._client.get(
                f"/v1/accounts/{self.account_id}/models/{model_id}:validateUpload"
            )

            if response.status_code < 400:
                logger.info("Validation succeeded on attempt %d", attempt + 1)
                if progress_callback:
                    await progress_callback(
                        "validating",
                        f"Validation successful on attempt {attempt + 1}",
                        {"status": "success", "attempt": attempt + 1},
                    )
                return

            # Failure handling
            error_body = response.text
            logger.error(
                "Validation attempt %d/%d failed (HTTP %d): %s",
                attempt + 1,
                max_retries,
                response.status_code,
                error_body,
            )

            if attempt == max_retries - 1:
                logger.error(
                    "All %d validation attempts failed. Last error: %s",
                    max_retries,
                    error_body,
                )
                _raise_api_error(
                    response,
                    context=f"validate upload for model '{model_id}'",
                )
            else:
                if progress_callback:
                    await progress_callback(
                        "validating",
                        f"Validation failed, retrying in {retry_delay}s "
                        f"({max_retries - attempt - 1} retries remaining)...",
                        {
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error": error_body,
                        },
                    )

    async def get_model_status(self, model_id: str) -> str:
        """Gets the status of an uploaded model.

        Args:
            model_id: Fireworks model ID (short ID or full path)

        Returns:
            Status string
        """
        # Handle both short ID and full path
        if "/" not in model_id:
            model_path = f"/v1/accounts/{self.account_id}/models/{model_id}"
        else:
            model_path = f"/v1/{model_id}"

        response = await self._client.get(model_path)
        if response.is_error:
            _raise_api_error(response, context=f"get model status for '{model_id}'")
        data = response.json()
        return data.get("state", "unknown")

    async def prepare_model(
        self, model_id: str, precision: str | None = None
    ) -> dict[str, Any]:
        """Prepares a model for deployment (optional precision conversion).

        Args:
            model_id: Fireworks model ID
            precision: Target precision (e.g., "fp16", "int8")

        Returns:
            Preparation result
        """
        if "/" not in model_id:
            model_path = f"/v1/accounts/{self.account_id}/models/{model_id}:prepare"
        else:
            model_path = f"/v1/{model_id}:prepare"

        payload: dict[str, Any] = {}
        if precision:
            payload["precision"] = precision

        response = await self._client.post(model_path, json=payload)
        if response.is_error:
            _raise_api_error(response, context=f"prepare model '{model_id}'")
        return response.json()

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
    ) -> Endpoint:
        """Creates an inference endpoint (deployment) for a model.

        Args:
            model_id: Fireworks model ID
            hardware: Hardware configuration
            autoscaling: Autoscaling configuration
            display_name: Optional display name

        Returns:
            Created Endpoint
        """
        payload: dict[str, Any] = {
            "baseModel": model_id,
            "acceleratorType": self._to_fireworks_accelerator(hardware),
            "acceleratorCount": hardware.count,
            "minReplicaCount": autoscaling.min_replicas,
            "maxReplicaCount": autoscaling.max_replicas,
        }

        if display_name:
            payload["displayName"] = display_name

        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/deployments",
            json=payload,
        )
        if not response.is_success:
            _raise_api_error(
                response,
                context=f"create endpoint for model '{model_id}'",
            )
        data = response.json()

        return self._parse_deployment(data)

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Gets details of a deployment.

        Args:
            endpoint_id: Fireworks deployment ID

        Returns:
            Endpoint details
        """
        response = await self._client.get(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}"
        )
        if response.is_error:
            _raise_api_error(response, context=f"get endpoint '{endpoint_id}'")
        data = response.json()

        return self._parse_deployment(data)

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Updates a deployment's configuration (autoscaling and/or hardware).

        Uses PATCH /v1/accounts/{account}/deployments/{deployment_id} per the spec
        (Gateway_UpdateDeployment).  The spec requires ``baseModel`` in the request
        body, so the current deployment is fetched first to retrieve it.

        Args:
            endpoint_id: Fireworks deployment ID
            autoscaling: New autoscaling configuration
            hardware: New hardware configuration

        Returns:
            Updated Endpoint
        """
        # Fetch the current deployment to retrieve the required baseModel field.
        current = await self.get_endpoint(endpoint_id)

        # Build update payload with top-level field names per gatewayDeployment schema.
        payload: dict[str, Any] = {"baseModel": current.model_id}

        if autoscaling:
            payload["minReplicaCount"] = autoscaling.min_replicas
            payload["maxReplicaCount"] = autoscaling.max_replicas

        if hardware:
            payload["acceleratorType"] = self._to_fireworks_accelerator(hardware)
            payload["acceleratorCount"] = hardware.count

        response = await self._client.patch(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}",
            json=payload,
        )
        if response.is_error:
            _raise_api_error(response, context=f"update endpoint '{endpoint_id}'")
        data = response.json()

        return self._parse_deployment(data)

    async def delete_endpoint(self, endpoint_id: str) -> None:
        """Deletes a deployment.

        Args:
            endpoint_id: Fireworks deployment ID
        """
        response = await self._client.delete(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}"
        )
        if response.is_error:
            _raise_api_error(response, context=f"delete endpoint '{endpoint_id}'")

    async def list_endpoints(self) -> list[Endpoint]:
        """Lists all deployments owned by this account.

        Returns:
            List of Endpoints
        """
        endpoints: list[Endpoint] = []
        page_token = None

        while True:
            params: dict[str, Any] = {}
            if page_token:
                params["pageToken"] = page_token

            response = await self._client.get(
                f"/v1/accounts/{self.account_id}/deployments",
                params=params,
            )
            if response.is_error:
                _raise_api_error(response, context="list endpoints")
            data = response.json()

            for item in data.get("deployments", []):
                endpoints.append(self._parse_deployment(item))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return endpoints

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """Lists available hardware configurations.

        Note: Fireworks does not expose a hardware discovery API; this list is
        hardcoded (version FIREWORKS_HARDWARE_LIST_VERSION). Update when new
        accelerators are added by the provider.

        Args:
            model_id: Optional model ID (ignored for Fireworks)

        Returns:
            List of available HardwareConfigs
        """
        _ = model_id  # Not used by Fireworks API
        return [
            HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
            HardwareConfig(accelerator="nvidia_h100_80gb", count=1),
            HardwareConfig(accelerator="nvidia_h200_141gb", count=1),
            HardwareConfig(accelerator="amd_mi300x_192gb", count=1),
        ]

    async def list_models(
        self, include_public: bool = False, organization: str | None = None
    ) -> list[Model]:
        """Lists models uploaded to Fireworks.ai.

        Args:
            include_public: If True, include public/platform models.
                If False (default), only return user-uploaded
                custom models.
            organization: Not used for Fireworks.ai (included for
                interface compatibility).

        Returns:
            List of Model objects with status information
        """
        models = []

        # Get user's account models
        response = await self._client.get(f"/v1/accounts/{self.account_id}/models")
        if response.is_error:
            _raise_api_error(response, context="list models")
        data = response.json()

        items = data if isinstance(data, list) else data.get("models", [])

        # If include_public, also fetch public models from 'fireworks'
        if include_public:
            try:
                public_response = await self._client.get(
                    "/v1/accounts/fireworks/models"
                )
                if public_response.is_error:
                    _raise_api_error(public_response, context="list public models")
                public_data = public_response.json()
                public_items = (
                    public_data
                    if isinstance(public_data, list)
                    else public_data.get("models", [])
                )
                items.extend(public_items)
            except Exception:
                # If fetching public models fails, just continue with user models
                pass

        for item in items:
            # Parse model information from Fireworks response
            model_id = item.get("name", item.get("id", ""))

            # Use displayName if present and non-empty, otherwise extract from model_id
            display_name = item.get("displayName", "")
            if display_name:
                model_name = display_name
            else:
                # Extract the last part of the model path as the name
                model_name = model_id.split("/")[-1] if "/" in model_id else model_id

            status = item.get("state", "unknown").lower()

            # Determine model type from the 'kind' field
            model_type = None
            kind = item.get("kind", "").upper()
            if "LORA" in kind or "PEFT" in kind:
                model_type = ModelType.ADAPTER
            elif kind == "CUSTOM_MODEL":
                model_type = ModelType.FULL
            else:
                # Fallback to checking modelFormat
                model_format = item.get("modelFormat", "").lower()
                if "lora" in model_format or "adapter" in model_format:
                    model_type = ModelType.ADAPTER
                elif model_format == "huggingface":
                    model_type = ModelType.FULL

            # Parse created_at - Fireworks uses 'createTime' not 'createdAt'
            created_at = None
            create_time = item.get("createTime") or item.get("createdAt")
            if create_time:
                try:
                    created_at = datetime.fromisoformat(
                        create_time.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            models.append(
                Model(
                    model_id=model_id,
                    model_name=model_name,
                    status=status,
                    provider=DeploymentProvider.FIREWORKS,
                    model_type=model_type,
                    created_at=created_at,
                    base_model=item.get("baseModel"),
                )
            )

        return models

    async def delete_model(self, model_id: str) -> None:
        """Deletes a model on Fireworks.ai.

        Args:
            model_id: Fireworks model ID (e.g., "my-model" or
                     "accounts/{account_id}/models/my-model")
        """
        # Handle both short ID and full path
        if "/" not in model_id:
            model_path = f"/v1/accounts/{self.account_id}/models/{model_id}"
        else:
            # Extract model ID from full path
            # accounts/{account_id}/models/{model_id} -> model_id
            model_id_only = model_id.split("/")[-1]
            model_path = f"/v1/accounts/{self.account_id}/models/{model_id_only}"

        response = await self._client.delete(model_path)
        if response.is_error:
            _raise_api_error(response, context=f"delete model '{model_id}'")
