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

_MB = 1024 * 1024


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parses an ISO-8601 timestamp (with optional trailing ``Z``) into a datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


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


class FireworksInvalidModelIdError(ValueError):
    """Raised when a model ID violates Fireworks resource-ID naming rules.

    Fireworks resource IDs must satisfy all of the following constraints
    (https://docs.fireworks.ai/getting-started/concepts#resource-names-and-ids):

    * Between 1 and 63 characters (inclusive)
    * Consists only of lowercase letters (a-z), digits (0-9), and hyphens (-)
    * Does not begin or end with a hyphen (-)
    * Does not begin with a digit
    """


def _validate_fireworks_model_id(model_id: str) -> None:
    """Validate that *model_id* conforms to Fireworks resource-ID naming rules.

    Rules (https://docs.fireworks.ai/getting-started/concepts#resource-names-and-ids):

    * Between 1 and 63 characters (inclusive)
    * Consists only of lowercase letters (a-z), digits (0-9), and hyphens (-)
    * Does not begin or end with a hyphen (-)
    * Does not begin with a digit

    Args:
        model_id: The candidate model ID string to validate.

    Raises:
        FireworksInvalidModelIdError: If *model_id* violates any naming rule.
    """
    if not model_id:
        raise FireworksInvalidModelIdError(
            f"Model ID must be between 1 and 63 characters; got empty string."
        )
    if len(model_id) > 63:
        raise FireworksInvalidModelIdError(
            f"Model ID must be at most 63 characters; "
            f"'{model_id}' has {len(model_id)} characters."
        )
    invalid_chars = {c for c in model_id if not (c.islower() or c.isdigit() or c == "-")}
    if invalid_chars:
        raise FireworksInvalidModelIdError(
            f"Model ID must consist only of lowercase letters (a-z), digits (0-9), "
            f"and hyphens (-); '{model_id}' contains invalid characters: "
            f"{sorted(invalid_chars)}."
        )
    if model_id[0] == "-":
        raise FireworksInvalidModelIdError(
            f"Model ID must not begin with a hyphen; got '{model_id}'."
        )
    if model_id[-1] == "-":
        raise FireworksInvalidModelIdError(
            f"Model ID must not end with a hyphen; got '{model_id}'."
        )
    if model_id[0].isdigit():
        raise FireworksInvalidModelIdError(
            f"Model ID must not begin with a digit; got '{model_id}'."
        )


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

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the async context manager and closes the HTTP client."""
        await self._client.aclose()

    def _get_inference_auth_headers(self) -> dict[str, str]:
        """Returns auth headers for inference (test_endpoint)."""
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _check_response(response: httpx.Response, context: str) -> None:
        """Raises if the response indicates an error."""
        if not response.is_success:
            _raise_api_error(response, context=context)

    @staticmethod
    async def _notify(
        callback: Any | None,
        stage: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Invokes the progress callback if one was provided."""
        if callback:
            await callback(stage, message, details or {})

    @staticmethod
    def _to_fireworks_accelerator(hw: HardwareConfig) -> str:
        """Converts HardwareConfig accelerator to Fireworks format."""
        return FIREWORKS_ACCELERATORS.get(hw.accelerator, hw.accelerator.upper())

    @staticmethod
    def _from_fireworks_accelerator(accelerator: str) -> str:
        """Converts Fireworks accelerator to our standard format."""
        return FIREWORKS_ACCELERATORS_REVERSE.get(accelerator, accelerator.lower())

    def _model_api_path(self, model_id: str, suffix: str = "") -> str:
        """Returns the API path for a model.

        Accepts short ID or full path (e.g. accounts/.../models/id).

        Args:
            model_id: Fireworks model ID (short) or full path
            suffix: Optional suffix (e.g. ':prepare' for prepare endpoint)

        Returns:
            Path segment for use with the API client.
        """
        if "/" not in model_id:
            return f"/v1/accounts/{self.account_id}/models/{model_id}{suffix}"
        return f"/v1/{model_id}{suffix}"

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

        created_at = _parse_timestamp(data.get("createTime"))

        # Extract deployment ID from name (accounts/{account}/deployments/{id})
        name = data.get("name", "")
        endpoint_id = name.split("/")[-1] if "/" in name else name

        # Fireworks Get Deployment API does not return endpointUrl (see
        # https://docs.fireworks.ai/api-reference/get-deployment). For on-demand
        # deployments, inference uses the standard chat completions endpoint with
        # the deployment resource name as the model. See on-demand quickstart:
        # https://docs.fireworks.ai/getting-started/ondemand-quickstart
        endpoint_url = data.get("endpointUrl")
        if not endpoint_url:
            endpoint_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        inference_model_name = name if name else None

        return Endpoint(
            endpoint_id=endpoint_id,
            provider=DeploymentProvider.FIREWORKS,
            model_id=data.get("baseModel", ""),
            endpoint_url=endpoint_url,
            state=state,
            hardware=hardware,
            autoscaling=autoscaling,
            created_at=created_at,
            display_name=data.get("displayName"),
            inference_model_name=inference_model_name,
        )

    @staticmethod
    def _detect_model_type(item: dict[str, Any]) -> ModelType | None:
        """Infers the ModelType from Fireworks ``kind`` / ``modelFormat`` fields."""
        kind = item.get("kind", "").upper()
        if "LORA" in kind or "PEFT" in kind:
            return ModelType.ADAPTER
        if kind in ("CUSTOM_MODEL", "HF_BASE_MODEL"):
            return ModelType.FULL
        model_format = item.get("modelFormat", "").lower()
        if "lora" in model_format or "adapter" in model_format:
            return ModelType.ADAPTER
        if model_format == "huggingface":
            return ModelType.FULL
        return None

    @classmethod
    def _parse_model(cls, item: dict[str, Any]) -> Model:
        """Parses a Fireworks model response dict into a Model dataclass."""
        model_id = item.get("name", item.get("id", ""))
        display_name = item.get("displayName", "")
        model_name = (
            display_name
            if display_name
            else (model_id.split("/")[-1] if "/" in model_id else model_id)
        )
        return Model(
            model_id=model_id,
            model_name=model_name,
            status=item.get("state", "unknown").lower(),
            provider=DeploymentProvider.FIREWORKS,
            model_type=cls._detect_model_type(item),
            created_at=_parse_timestamp(
                item.get("createTime") or item.get("createdAt")
            ),
            base_model=item.get("baseModel"),
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
        _validate_fireworks_model_id(model_name)
        model_id = model_name

        # Validate model_source before touching the Fireworks API so that
        # unsupported formats are rejected immediately without leaving orphaned
        # model resources.
        self._check_model_source_supported(model_source)

        # Step 1: Resolve model source to a local directory so we can
        # enumerate files before creating the model resource.
        temp_dir = None
        try:
            model_dir, temp_dir = await self._resolve_model_source(
                model_source, progress_callback
            )

            # Step 2: Collect the file manifest.  For HF_BASE_MODEL uploads the
            # create payload must include huggingfaceFiles so the backend knows
            # which files to expect.
            file_inventory = self._collect_file_inventory(model_dir)
            hf_files = sorted(file_inventory.keys())

            # Step 3: Create model resource on Fireworks
            create_payload = await self._create_model_resource(
                model_id,
                model_type,
                base_model,
                progress_callback,
                huggingface_files=hf_files,
            )

            # Steps 4–5: Upload files, validate
            await self._upload_model_files(
                model_dir, model_id, progress_callback, file_inventory
            )
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
        huggingface_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Creates a model resource on Fireworks and returns the create payload.

        Args:
            model_id: Short model identifier.
            model_type: FULL or ADAPTER.
            base_model: Required base model for LoRA adapters.
            progress_callback: Optional async progress callback.
            huggingface_files: File names to declare in the create payload
                when uploading a HuggingFace checkpoint (HF_BASE_MODEL kind).
                Must be provided for FULL model uploads so the server knows
                which files to expect.
        """
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
            # HF_BASE_MODEL with huggingfaceFiles tells the Fireworks backend
            # exactly which files the model checkpoint consists of.  This is
            # required for the presigned-URL upload flow to work correctly.
            base_model_details: dict[str, Any] = {
                "checkpointFormat": "HUGGINGFACE",
                "worldSize": 1,
            }
            if huggingface_files:
                base_model_details["huggingfaceFiles"] = huggingface_files

            create_payload = {
                "modelId": model_id,
                "model": {
                    "kind": "HF_BASE_MODEL",
                    "baseModelDetails": base_model_details,
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
            self._check_response(response, f"create model resource '{model_id}'")

        created_model_name = response.json().get("name", "")
        logger.info("Model created: %s", created_model_name)

        await self._notify(
            progress_callback,
            "creating",
            f"Model resource created on Fireworks: {model_id}",
            {"provider_model_id": created_model_name},
        )
        return create_payload

    @staticmethod
    def _check_model_source_supported(model_source: str) -> None:
        """Raises ValueError early for model source formats that are not supported.

        Called before any Fireworks API requests to prevent orphaned model
        resources being created when the source cannot be resolved.
        """
        from urllib.parse import urlparse  # noqa: PLC0415 (lazy import — rarely needed)

        if not model_source.startswith(("http://", "https://")):
            return
        parsed = urlparse(model_source)
        hostname = (parsed.hostname or "").lower().removeprefix("www.")
        if hostname == "huggingface.co":
            repo_id = parsed.path.lstrip("/")
            raise ValueError(
                f"HuggingFace URLs are not supported as a model source for Fireworks. "
                f"Pass the bare repository ID instead.\n"
                f"  Got:  {model_source}\n"
                f"  Use:  {repo_id}"
            )

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
        - HTTP/HTTPS presigned URLs (e.g., GCS/S3 signed archive URLs)
        - HuggingFace repository IDs (e.g., "Qwen/Qwen3-4B")

        Not supported:
        - HuggingFace HTTPS URLs (e.g., "https://huggingface.co/Qwen/Qwen3-4B").
          Use the bare repo ID instead. `_check_model_source_supported` rejects
          these before this method is called.

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
        path_exists = await asyncio.to_thread(model_path.exists)
        path_is_dir = path_exists and await asyncio.to_thread(model_path.is_dir)
        if path_is_dir:
            return model_path, None

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
        await self._notify(
            progress_callback,
            "downloading",
            f"Downloading model from HuggingFace: {repo_id}",
        )

        try:
            cache_path = await asyncio.to_thread(
                snapshot_download, repo_id=repo_id, repo_type="model"
            )
        except Exception as e:
            logger.error("Failed to download model from HuggingFace: %s", e)
            raise ValueError(
                f"Failed to download model from HuggingFace Hub: {repo_id}. Error: {e}"
            ) from e

        model_dir = Path(cache_path)
        logger.info("Model downloaded to: %s", model_dir)

        await self._notify(
            progress_callback,
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
        await self._notify(
            progress_callback,
            "downloading",
            "Downloading model archive from cloud storage...",
        )

        temp_dir = Path(tempfile.mkdtemp())
        archive_path = temp_dir / "model_archive"

        async with httpx.AsyncClient(timeout=300.0) as download_client:
            response = await download_client.get(url)
            self._check_response(response, "download model archive")
            async with aiofiles.open(archive_path, "wb") as f:
                await f.write(response.content)

        size_mb = len(response.content) / _MB
        logger.info("Downloaded %.1f MB", size_mb)
        await self._notify(
            progress_callback,
            "downloading",
            f"Download complete ({size_mb:.1f} MB)",
            {"bytes_downloaded": len(response.content)},
        )

        # Extract
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        await self._notify(progress_callback, "extracting", "Extracting model archive...")

        if zipfile.is_zipfile(archive_path):
            logger.info("Extracting ZIP archive")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            raise ValueError("Model source is not a recognized archive format (zip)")

        logger.info("Extracted model to %s", extract_dir)
        return extract_dir, temp_dir

    def _collect_file_inventory(self, model_dir: Path) -> dict[str, int]:
        """Builds filenameToSize inventory for the model directory.

        Excludes HuggingFace cache artifacts (e.g. *.lock, *.metadata), paths
        under .cache or huggingface, and training-state files that are not needed
        for inference (optimizer, scheduler, RNG state, trainer bookkeeping).

        Returns:
            Dict mapping relative filename (e.g. "config.json") to file size
            in bytes.  These names are used both in the ``huggingfaceFiles``
            list of the create payload and in the ``filenameToSize`` map sent
            to ``getUploadEndpoint``.
        """
        _IGNORED_SUFFIXES = (".lock", ".metadata")
        # Training-state files produced by intermediate checkpoints (e.g.
        # checkpoint-100/).  These are irrelevant for inference and can be
        # very large (optimizer.pt is typically 2× the model weights).
        _TRAINING_STATE_NAMES = frozenset({
            "optimizer.pt",
            "optimizer.bin",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
            "scaler.pt",
        })
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
                if fname in _TRAINING_STATE_NAMES:
                    logger.debug("Skipping training-state file: %s", rel)
                    continue
                # rng_state*.pth covers rank-suffixed variants (e.g. rng_state_0.pth)
                if fname.startswith("rng_state") and fname.endswith(".pth"):
                    logger.debug("Skipping rng state file: %s", rel)
                    continue

                file_sizes[rel] = fpath.stat().st_size

        return file_sizes

    def _upload_order_key(self, item: tuple[str, str]) -> tuple[int, str]:
        """Sort key: config/tokenizer files first for GCS propagation."""
        filename = item[0]
        if filename == "config.json":
            return (0, filename)
        if filename == "generation_config.json":
            return (1, filename)
        if filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        ):
            return (2, filename)
        if filename.endswith(".index.json") or "tokenizer" in filename.lower():
            return (3, filename)
        return (4, filename)

    async def _get_signed_urls_ordered(
        self, model_id: str, file_sizes: dict[str, int]
    ) -> list[tuple[str, str]]:
        """Return (filename, signed_url) list from getUploadEndpoint in upload order."""
        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/models/{model_id}:getUploadEndpoint",
            json={"filenameToSize": file_sizes, "enableResumableUpload": False},
        )
        self._check_response(
            response, f"get upload endpoint for model '{model_id}'"
        )
        file_upload_urls = response.json().get("filenameToSignedUrls", {})
        if not file_upload_urls:
            raise ValueError("No upload URLs received from Fireworks API.")

        return sorted(file_upload_urls.items(), key=self._upload_order_key)

    async def _upload_model_files(
        self,
        model_dir: Path,
        model_id: str,
        progress_callback: Any | None,
        file_sizes: dict[str, int] | None = None,
    ) -> None:
        """Obtains signed URLs and uploads each file.

        Follows the Fireworks REST API upload flow documented at
        https://docs.fireworks.ai/models/uploading-custom-models-api:

        1. Call ``getUploadEndpoint`` to obtain per-file signed URLs.
        2. PUT each file to its signed URL (streamed from disk, with
           retry and exponential back-off on transient failures).

        Files are uploaded in deterministic order (config/tokenizer first) to
        improve validation success (GCS propagation).

        Args:
            file_sizes: Pre-computed file inventory. When ``None`` the
                inventory is collected from *model_dir* (backward compat).
        """
        if file_sizes is None:
            file_sizes = self._collect_file_inventory(model_dir)
        total_bytes = sum(file_sizes.values())
        logger.info(
            "Found %d files to upload (%.1f MB)",
            len(file_sizes),
            total_bytes / _MB,
        )
        if "config.json" in file_sizes:
            logger.info("config.json found (%d bytes)", file_sizes["config.json"])
        else:
            logger.error(
                "config.json NOT found in model files: %s", list(file_sizes.keys())
            )

        await self._notify(
            progress_callback,
            "extracting",
            f"Found {len(file_sizes)} files ({total_bytes / _MB:.1f} MB total)",
            {"file_count": len(file_sizes), "files": list(file_sizes.keys())},
        )

        upload_items = await self._get_signed_urls_ordered(model_id, file_sizes)
        total_files = len(upload_items)
        uploaded_bytes = 0

        await self._notify(
            progress_callback,
            "uploading",
            f"Starting upload of {total_files} files ({total_bytes / _MB:.1f} MB)",
            {"total_files": total_files, "total_bytes": total_bytes},
        )

        for idx, (filename, signed_url) in enumerate(upload_items, 1):
            file_path = model_dir / filename
            file_size = file_sizes[filename]

            logger.info(
                "[%d/%d] Uploading %s (%.2f MB)",
                idx,
                total_files,
                filename,
                file_size / _MB,
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
                uploaded_bytes / _MB,
                total_bytes / _MB,
            )

            await self._notify(
                progress_callback,
                "uploading",
                f"Uploaded {filename} ({idx}/{total_files}, "
                f"{uploaded_bytes / _MB:.1f} / {total_bytes / _MB:.1f} MB)",
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
            total_bytes / _MB,
        )
        await self._notify(
            progress_callback,
            "uploading",
            f"All {total_files} files uploaded ({total_bytes / _MB:.1f} MB total)",
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

        for attempt in range(1, self.UPLOAD_MAX_RETRIES + 1):
            try:
                await asyncio.to_thread(
                    self._sync_put_file, file_path, signed_url, headers
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
        await self._notify(progress_callback, "validating", "Validating uploaded model...")

        max_retries = self.VALIDATION_MAX_RETRIES
        retry_delay = self.VALIDATION_RETRY_DELAY_S

        if self.VALIDATION_INITIAL_DELAY_S > 0:
            logger.info(
                "Waiting %.0fs for GCS propagation before first validation...",
                self.VALIDATION_INITIAL_DELAY_S,
            )
            await asyncio.sleep(self.VALIDATION_INITIAL_DELAY_S)

        for attempt in range(max_retries):
            if attempt > 0:
                logger.info(
                    "Validation retry %d/%d: waiting %ds...",
                    attempt + 1,
                    max_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

            logger.info("Validation attempt %d/%d", attempt + 1, max_retries)
            await self._notify(
                progress_callback,
                "validating",
                f"Validation attempt {attempt + 1}/{max_retries}...",
                {"attempt": attempt + 1, "max_retries": max_retries},
            )

            response = await self._client.get(
                f"/v1/accounts/{self.account_id}/models/{model_id}:validateUpload"
            )

            if not response.is_error:
                logger.info("Validation succeeded on attempt %d", attempt + 1)
                await self._notify(
                    progress_callback,
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
                self._check_response(
                    response, f"validate upload for model '{model_id}'"
                )
            else:
                await self._notify(
                    progress_callback,
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
        model_path = self._model_api_path(model_id)
        response = await self._client.get(model_path)
        self._check_response(response, f"get model status for '{model_id}'")
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
        model_path = self._model_api_path(model_id, ":prepare")
        payload: dict[str, Any] = {}
        if precision:
            payload["precision"] = precision

        response = await self._client.post(model_path, json=payload)
        self._check_response(response, f"prepare model '{model_id}'")
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
        self._check_response(response, f"create endpoint for model '{model_id}'")
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
        self._check_response(response, f"get endpoint '{endpoint_id}'")
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
        self._check_response(response, f"update endpoint '{endpoint_id}'")
        data = response.json()

        return self._parse_deployment(data)

    async def delete_endpoint(self, endpoint_id: str, *, force: bool = False) -> None:
        """Deletes a deployment.

        Args:
            endpoint_id: Fireworks deployment ID
            force: If True, pass ignoreChecks and hard query params to bypass
                Fireworks safety checks (e.g. deployments with recent inference
                requests) and perform a hard deletion.
        """
        params: dict[str, Any] = {"ignoreChecks": True, "hard": True} if force else {}
        response = await self._client.delete(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}",
            params=params,
        )
        self._check_response(response, f"delete endpoint '{endpoint_id}'")

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
            self._check_response(response, "list endpoints")
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
        self._check_response(response, "list models")
        data = response.json()

        items = data if isinstance(data, list) else data.get("models", [])

        # If include_public, also fetch public models from 'fireworks'
        if include_public:
            try:
                public_response = await self._client.get(
                    "/v1/accounts/fireworks/models"
                )
                self._check_response(public_response, "list public models")
                public_data = public_response.json()
                public_items = (
                    public_data
                    if isinstance(public_data, list)
                    else public_data.get("models", [])
                )
                items.extend(public_items)
            except Exception:
                logger.warning("Failed to fetch public models", exc_info=True)

        return [self._parse_model(item) for item in items]

    async def delete_model(self, model_id: str) -> None:
        """Deletes a model on Fireworks.ai.

        Retries on HTTP 400 "active deployments" errors to handle the Fireworks
        control-plane propagation lag that can occur immediately after an endpoint
        is deleted (the deployment association may not be cleared instantly).

        Args:
            model_id: Fireworks model ID (e.g., "my-model" or
                     "accounts/{account_id}/models/my-model")
        """
        short_id = model_id.split("/")[-1] if "/" in model_id else model_id
        model_path = self._model_api_path(short_id)

        _MAX_RETRIES = 5
        for attempt in range(_MAX_RETRIES):
            response = await self._client.delete(model_path)
            if not response.is_error:
                return
            if (
                response.status_code == 400
                and "active deployments" in response.text
                and attempt < _MAX_RETRIES - 1
            ):
                wait_s = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s
                logger.info(
                    "Model '%s' still has active deployments (propagation lag), "
                    "retrying in %ds (%d/%d)...",
                    model_id,
                    wait_s,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                await asyncio.sleep(wait_s)
                continue
            self._check_response(response, f"delete model '{model_id}'")
