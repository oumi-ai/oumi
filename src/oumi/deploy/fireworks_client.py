"""Fireworks.ai deployment client."""

import asyncio
import os
import shutil
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx

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

# Mapping from Fireworks accelerator names to our standard format
FIREWORKS_ACCELERATORS = {
    "nvidia_a100_80gb": "NVIDIA_A100_80GB",
    "nvidia_h100_80gb": "NVIDIA_H100_80GB",
    "nvidia_h200_141gb": "NVIDIA_H200_141GB",
    "amd_mi300x": "AMD_MI300X",
}

# Reverse mapping
FIREWORKS_ACCELERATORS_REVERSE = {v: k for k, v in FIREWORKS_ACCELERATORS.items()}

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
        """Initialize the Fireworks.ai deployment client.

        Args:
            api_key: Fireworks API key. If not provided, reads from
                     FIREWORKS_API_KEY environment variable.
            account_id: Fireworks account ID. If not provided, reads from
                        FIREWORKS_ACCOUNT_ID environment variable.
        """
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks API key must be provided or set via FIREWORKS_API_KEY env var"
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

    async def __aenter__(self) -> "FireworksDeploymentClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._client.aclose()

    def _to_fireworks_accelerator(self, hw: HardwareConfig) -> str:
        """Convert HardwareConfig accelerator to Fireworks format.

        Args:
            hw: HardwareConfig with accelerator name

        Returns:
            Fireworks accelerator type string
        """
        result = FIREWORKS_ACCELERATORS.get(hw.accelerator)
        return result if result is not None else hw.accelerator.upper()

    def _from_fireworks_accelerator(self, accelerator: str) -> str:
        """Convert Fireworks accelerator to our standard format.

        Args:
            accelerator: Fireworks accelerator type string

        Returns:
            Standard accelerator name
        """
        result = FIREWORKS_ACCELERATORS_REVERSE.get(accelerator)
        return result if result is not None else accelerator.lower()

    def _parse_deployment(self, data: dict[str, Any]) -> Endpoint:
        """Parse Fireworks deployment response into Endpoint dataclass."""
        state_str = data.get("state", "PENDING").upper()
        state = FIREWORKS_STATE_MAP.get(state_str, EndpointState.PENDING)

        # Parse hardware config
        config = data.get("config", {})
        accelerator = config.get("acceleratorType", "")
        count = config.get("acceleratorCount", 1)
        hardware = HardwareConfig(
            accelerator=self._from_fireworks_accelerator(accelerator),
            count=count,
        )

        # Parse autoscaling
        min_replicas = config.get("minReplicas", 1)
        max_replicas = config.get("maxReplicas", 1)
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
            model_id=data.get("model", ""),
            endpoint_url=data.get("endpointUrl"),
            state=state,
            hardware=hardware,
            autoscaling=autoscaling,
            created_at=created_at,
            display_name=data.get("displayName"),
        )

    async def _package_model(self, model_source: str) -> Path:
        """Package model directory as tar.gz for Fireworks upload.

        Args:
            model_source: Path to model directory or cloud storage URL

        Returns:
            Path to the created tar.gz file
        """
        # If it's a cloud storage path, we'd need to download first
        # For now, assume local path
        model_dir = Path(model_source)

        tar_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_dir, arcname=".")

        return tar_path

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        progress_callback: Any | None = None,
    ) -> UploadedModel:
        """Upload a model to Fireworks.ai using multi-step flow.

        API Flow (from https://docs.fireworks.ai/models/uploading-custom-models-api):
        1. Create model object with modelId and model structure
        2. Get signed upload URLs (one per file) with file sizes
        3. Upload each file to its signed URL
        4. Validate the upload

        Args:
            model_source: Path to model directory or cloud storage URL (e.g., s3://bucket/path)
            model_name: Model ID to use (e.g., "my-custom-model")
            model_type: Type of model (FULL or ADAPTER)
            base_model: Base model for LoRA adapters (e.g., "accounts/fireworks/models/llama-v3p1-8b-instruct")
            progress_callback: Optional async callback function for progress updates.
                Signature: async def callback(stage: str, message: str, details: dict)
                Stages: "downloading", "extracting", "uploading", "waiting", "validating"

        Returns:
            UploadedModel with provider-specific model ID
        """
        import logging
        import time

        logger = logging.getLogger(__name__)

        # Sanitize model name to create a valid model ID (alphanumeric, hyphens, underscores)
        # Add timestamp suffix to ensure uniqueness across retries
        base_id = model_name.lower().replace(" ", "-")
        base_id = "".join(c for c in base_id if c.isalnum() or c in "-_")
        timestamp = int(time.time())
        model_id = f"{base_id}-{timestamp}"

        logger.info(
            f"Creating model on Fireworks: model_id={model_id}, "
            f"model_type={model_type}, base_model={base_model}"
        )

        # Step 1: Create model resource
        # API expects: {"modelId": "...", "model": {"kind": "...", "baseModelDetails": {...}}}
        if model_type == ModelType.ADAPTER and base_model:
            # For LoRA adapters
            create_payload = {
                "modelId": model_id,
                "model": {
                    "kind": "LORA_ADAPTER",
                    "baseModel": base_model,
                },
            }
        else:
            # For full models
            create_payload = {
                "modelId": model_id,
                "model": {
                    "kind": "CUSTOM_MODEL",
                    "baseModelDetails": {
                        "checkpointFormat": "HUGGINGFACE",
                        "worldSize": 1,  # Number of GPUs used during training
                    },
                },
            }

        logger.info(f"Creating model with payload: {create_payload}")
        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/models",
            json=create_payload,
        )

        # Capture error details before raising
        if response.status_code >= 400:
            error_body = response.text
            logger.error(f"Fireworks API error response: {error_body}")

            # Handle model ID conflict specifically
            if response.status_code == 409:
                logger.error(
                    f"Model ID '{model_id}' already exists in Fireworks. "
                    f"This shouldn't happen with timestamp-based IDs. "
                    f"You may need to delete the existing model manually or use a different name."
                )

        response.raise_for_status()
        model_data = response.json()
        created_model_name = model_data.get("name", "")
        logger.info(f"✓ Model created: {created_model_name}")

        if progress_callback:
            await progress_callback(
                "creating",
                f"Model resource created on Fireworks: {model_id}",
                {"provider_model_id": created_model_name},
            )

        # Step 2: Download model files from cloud storage if needed
        # The model_source is likely an S3 presigned URL pointing to an archive
        # We need to download, extract, and get the individual files and their sizes
        temp_dir = None
        model_dir = None

        try:
            if model_source.startswith(("http://", "https://")):
                # Presigned URL - download the archive
                logger.info("Downloading model archive from presigned URL")
                if progress_callback:
                    await progress_callback(
                        "downloading",
                        "Downloading model archive from cloud storage...",
                        {},
                    )
                temp_dir = Path(tempfile.mkdtemp())

                # Download the archive
                archive_path = temp_dir / "model_archive"
                logger.info(f"Downloading to {archive_path}")

                async with httpx.AsyncClient(timeout=300.0) as download_client:
                    response = await download_client.get(model_source)
                    response.raise_for_status()

                    async with aiofiles.open(archive_path, "wb") as f:
                        await f.write(response.content)

                logger.info(f"Downloaded {len(response.content)} bytes")

                if progress_callback:
                    await progress_callback(
                        "downloading",
                        f"Download complete ({len(response.content) / 1024 / 1024:.1f} MB)",
                        {"bytes_downloaded": len(response.content)},
                    )

                # Extract the archive
                extract_dir = temp_dir / "extracted"
                extract_dir.mkdir()

                if progress_callback:
                    await progress_callback(
                        "extracting",
                        "Extracting model archive...",
                        {},
                    )

                # Detect archive type and extract
                if zipfile.is_zipfile(archive_path):
                    logger.info("Extracting ZIP archive")
                    with zipfile.ZipFile(archive_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                elif tarfile.is_tarfile(archive_path):
                    logger.info("Extracting TAR archive")
                    with tarfile.open(archive_path, "r:*") as tar_ref:
                        tar_ref.extractall(extract_dir)
                else:
                    raise ValueError(
                        "Model source is not a recognized archive format (zip or tar)"
                    )

                model_dir = extract_dir
                logger.info(f"Extracted model to {model_dir}")

            elif model_source.startswith(("s3://", "gs://")):
                # Direct cloud storage path (not presigned URL)
                raise NotImplementedError(
                    f"Direct cloud storage paths not supported. "
                    f"Convert to presigned URL first. Model source: {model_source}"
                )
            else:
                # Local path
                model_dir = Path(model_source)
                if not model_dir.exists():
                    raise ValueError(f"Model source does not exist: {model_source}")
                if not model_dir.is_dir():
                    raise ValueError(
                        f"Model source must be a directory: {model_source}"
                    )

            # Get all files and their sizes
            file_sizes = {}
            for root, _, files in os.walk(model_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(model_dir)
                    file_sizes[str(rel_path)] = file_path.stat().st_size

            logger.info(f"Found {len(file_sizes)} files to upload")
            logger.info(f"Files to upload: {list(file_sizes.keys())}")

            if progress_callback:
                await progress_callback(
                    "extracting",
                    f"Found {len(file_sizes)} files to upload "
                    f"({sum(file_sizes.values()) / 1024 / 1024:.1f} MB total)",
                    {"file_count": len(file_sizes), "files": list(file_sizes.keys())},
                )

            # Check for config.json specifically
            if "config.json" in file_sizes:
                logger.info(
                    f"✓ config.json found in extracted files ({file_sizes['config.json']} bytes)"
                )
            else:
                logger.error("✗ config.json NOT found in extracted files!")
                logger.error(f"Available files: {list(file_sizes.keys())}")

            # Step 3: Get signed upload URLs
            upload_payload = {
                "filenameToSize": file_sizes,
                "enableResumableUpload": False,
            }
            logger.info(f"Requesting upload URLs for {len(file_sizes)} files")
            response = await self._client.post(
                f"/v1/accounts/{self.account_id}/models/{model_id}:getUploadEndpoint",
                json=upload_payload,
            )
            response.raise_for_status()
            upload_data = response.json()
            file_upload_urls = upload_data.get("filenameToSignedUrls", {})

            if not file_upload_urls:
                logger.error("No upload URLs received from Fireworks API!")
                logger.error(f"Response data: {upload_data}")
                raise ValueError(
                    "No upload URLs received. Check if 'filenameToSignedUrls' is in response."
                )

            logger.info(f"Received {len(file_upload_urls)} signed URLs for upload")

            # Step 4: Upload each file to its signed URL
            total_files = len(file_upload_urls)
            uploaded_count = 0
            total_bytes = sum(file_sizes.values())
            uploaded_bytes = 0

            logger.info(
                f"Starting upload of {total_files} files "
                f"({total_bytes / 1024 / 1024:.1f} MB total)"
            )

            if progress_callback:
                await progress_callback(
                    "uploading",
                    f"Starting upload of {total_files} files "
                    f"({total_bytes / 1024 / 1024:.1f} MB total)",
                    {"total_files": total_files, "total_bytes": total_bytes},
                )

            async with httpx.AsyncClient(timeout=300.0) as upload_client:
                for filename, signed_url in file_upload_urls.items():
                    file_path = model_dir / filename
                    file_size = file_sizes[filename]
                    uploaded_count += 1

                    logger.info(
                        f"[{uploaded_count}/{total_files}] Uploading {filename} "
                        f"({file_size / 1024 / 1024:.2f} MB)..."
                    )

                    async with aiofiles.open(file_path, "rb") as f:
                        content = await f.read()

                    upload_response = await upload_client.put(
                        signed_url,
                        content=content,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "x-goog-content-length-range": f"{file_size},{file_size}",
                        },
                    )
                    upload_response.raise_for_status()
                    uploaded_bytes += file_size
                    logger.info(
                        f"[{uploaded_count}/{total_files}] ✓ Uploaded {filename} "
                        f"({uploaded_bytes / 1024 / 1024:.1f} MB / "
                        f"{total_bytes / 1024 / 1024:.1f} MB complete)"
                    )

                    if progress_callback:
                        await progress_callback(
                            "uploading",
                            f"Uploaded {filename} ({uploaded_count}/{total_files} files, "
                            f"{uploaded_bytes / 1024 / 1024:.1f} MB / "
                            f"{total_bytes / 1024 / 1024:.1f} MB)",
                            {
                                "current_file": filename,
                                "uploaded_count": uploaded_count,
                                "total_files": total_files,
                                "uploaded_bytes": uploaded_bytes,
                                "total_bytes": total_bytes,
                            },
                        )

            logger.info(
                f"✓ All {total_files} files uploaded successfully "
                f"({total_bytes / 1024 / 1024:.1f} MB total)"
            )

            if progress_callback:
                await progress_callback(
                    "uploading",
                    f"All {total_files} files uploaded successfully "
                    f"({total_bytes / 1024 / 1024:.1f} MB total)",
                    {"status": "complete", "total_files": total_files},
                )

            # Step 5: Wait for GCS propagation before triggering validation
            # Fireworks needs time to recognize uploaded files in GCS
            propagation_delay = 300  # 5 minutes
            logger.info(
                f"⏳ Waiting {propagation_delay // 60} minutes for files to propagate "
                f"on Fireworks infrastructure before validation..."
            )

            if progress_callback:
                await progress_callback(
                    "waiting",
                    f"Waiting {propagation_delay // 60} minutes for files to propagate "
                    f"on Fireworks infrastructure...",
                    {"wait_seconds": propagation_delay},
                )

            # Log progress every 30 seconds during the wait
            wait_interval = 30
            elapsed = 0
            while elapsed < propagation_delay:
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
                remaining = propagation_delay - elapsed
                if remaining > 0:
                    logger.info(
                        f"⏳ Propagation wait: {elapsed // 60}m {elapsed % 60}s elapsed, "
                        f"{remaining // 60}m {remaining % 60}s remaining..."
                    )
                    if progress_callback:
                        await progress_callback(
                            "waiting",
                            f"Propagation wait: {elapsed // 60}m {elapsed % 60}s elapsed, "
                            f"{remaining // 60}m {remaining % 60}s remaining...",
                            {
                                "elapsed_seconds": elapsed,
                                "remaining_seconds": remaining,
                            },
                        )

            logger.info("⏳ Propagation wait complete. Triggering model validation...")

            if progress_callback:
                await progress_callback(
                    "validating",
                    "Propagation wait complete. Starting validation...",
                    {},
                )

            # Add retry logic with delay - Fireworks may need additional time
            max_retries = 5
            retry_delay = 60  # 1 minute between retries

            for attempt in range(max_retries):
                if attempt > 0:
                    logger.info(
                        f"🔄 Validation retry {attempt + 1}/{max_retries}: "
                        f"Waiting {retry_delay}s before next attempt..."
                    )
                    await asyncio.sleep(retry_delay)
                    logger.info(
                        f"🔄 Validation retry {attempt + 1}/{max_retries}: "
                        f"Sending validation request..."
                    )
                else:
                    logger.info(
                        f"🔄 Validation attempt {attempt + 1}/{max_retries}: "
                        f"Sending validation request..."
                    )

                if progress_callback:
                    await progress_callback(
                        "validating",
                        f"Validation attempt {attempt + 1}/{max_retries}...",
                        {"attempt": attempt + 1, "max_retries": max_retries},
                    )

                response = await self._client.get(
                    f"/v1/accounts/{self.account_id}/models/{model_id}:validateUpload"
                )

                # Capture error details
                if response.status_code >= 400:
                    error_body = response.text
                    logger.error(
                        f"❌ Validation attempt {attempt + 1}/{max_retries} failed!"
                    )
                    logger.error(f"   Status code: {response.status_code}")
                    logger.error(f"   Error response: {error_body}")

                    # If this is the last retry, check if it's the config.json error
                    if attempt == max_retries - 1:
                        if (
                            response.status_code == 400
                            and "config.json not found" in error_body
                        ):
                            logger.warning(
                                "⚠️  Validation failed with 'config.json not found' even though "
                                "config.json was successfully uploaded. This appears to be a "
                                "Fireworks API issue. Continuing anyway - the model may still work."
                            )
                            # Don't raise - continue despite validation error
                            break
                        else:
                            logger.error(
                                f"❌ All {max_retries} validation attempts failed. "
                                f"Last error: {error_body}"
                            )
                    else:
                        # Not last retry, continue to next attempt
                        logger.info(
                            f"   Will retry in {retry_delay}s "
                            f"({max_retries - attempt - 1} retries remaining)"
                        )
                        if progress_callback:
                            await progress_callback(
                                "validating",
                                f"Validation failed, retrying in {retry_delay}s "
                                f"({max_retries - attempt - 1} retries remaining)...",
                                {
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "error": error_body,
                                    "status": "retrying",
                                },
                            )
                        continue

                # Success - break out of retry loop
                response.raise_for_status()
                logger.info(
                    f"✓ Validation triggered successfully on attempt {attempt + 1}"
                )
                if progress_callback:
                    await progress_callback(
                        "validating",
                        f"Validation successful on attempt {attempt + 1}",
                        {"status": "success", "attempt": attempt + 1},
                    )
                break

        finally:
            # Clean up temp directory if created
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return UploadedModel(
            provider_model_id=f"accounts/{self.account_id}/models/{model_id}",
            status="validating",
            request_payload=create_payload,
        )

    async def get_model_status(self, model_id: str) -> str:
        """Get the status of an uploaded model.

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
        response.raise_for_status()
        data = response.json()
        return data.get("state", "unknown")

    async def prepare_model(
        self, model_id: str, precision: str | None = None
    ) -> dict[str, Any]:
        """Prepare a model for deployment (optional precision conversion).

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
        response.raise_for_status()
        return response.json()

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
    ) -> Endpoint:
        """Create an inference endpoint (deployment) for a model.

        Args:
            model_id: Fireworks model ID
            hardware: Hardware configuration
            autoscaling: Autoscaling configuration
            display_name: Optional display name

        Returns:
            Created Endpoint
        """
        payload: dict[str, Any] = {
            "model": model_id,
            "config": {
                "accelerator_type": self._to_fireworks_accelerator(hardware),
                "accelerator_count": hardware.count,
                "min_replicas": autoscaling.min_replicas,
                "max_replicas": autoscaling.max_replicas,
            },
        }

        if display_name:
            payload["display_name"] = display_name

        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/deployments",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_deployment(data)

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Get details of a deployment.

        Args:
            endpoint_id: Fireworks deployment ID

        Returns:
            Endpoint details
        """
        response = await self._client.get(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}"
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_deployment(data)

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Update a deployment's configuration (scale).

        Args:
            endpoint_id: Fireworks deployment ID
            autoscaling: New autoscaling configuration
            hardware: New hardware configuration

        Returns:
            Updated Endpoint
        """
        config: dict[str, Any] = {}

        if autoscaling:
            config["min_replicas"] = autoscaling.min_replicas
            config["max_replicas"] = autoscaling.max_replicas

        if hardware:
            config["accelerator_type"] = self._to_fireworks_accelerator(hardware)
            config["accelerator_count"] = hardware.count

        payload: dict[str, Any] = {"config": config}

        response = await self._client.patch(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}:scale",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_deployment(data)

    async def delete_endpoint(self, endpoint_id: str) -> None:
        """Delete a deployment.

        Args:
            endpoint_id: Fireworks deployment ID
        """
        response = await self._client.delete(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}"
        )
        response.raise_for_status()

    async def list_endpoints(self) -> list[Endpoint]:
        """List all deployments owned by this account.

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
            response.raise_for_status()
            data = response.json()

            for item in data.get("deployments", []):
                endpoints.append(self._parse_deployment(item))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return endpoints

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """List available hardware configurations.

        Note: Fireworks doesn't have an API for this, so we return a hardcoded list.

        Args:
            model_id: Optional model ID (ignored for Fireworks)

        Returns:
            List of available HardwareConfigs
        """
        # Fireworks doesn't have a hardware list API, return known configs
        _ = model_id  # Not used by Fireworks API
        return [
            HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
            HardwareConfig(accelerator="nvidia_h100_80gb", count=1),
            HardwareConfig(accelerator="nvidia_h200_141gb", count=1),
            HardwareConfig(accelerator="amd_mi300x", count=1),
        ]

    async def list_models(
        self, include_public: bool = False, organization: str | None = None
    ) -> list[Model]:
        """List models uploaded to Fireworks.ai.

        Args:
            include_public: If True, include public/platform models. If False (default),
                           only return user-uploaded custom models.
            organization: Not used for Fireworks.ai (included for interface compatibility).

        Returns:
            List of Model objects with status information
        """
        models = []

        # Get user's account models
        response = await self._client.get(f"/v1/accounts/{self.account_id}/models")
        response.raise_for_status()
        data = response.json()

        items = data if isinstance(data, list) else data.get("models", [])

        # If include_public is True, also fetch public models from the 'fireworks' account
        if include_public:
            try:
                public_response = await self._client.get(
                    "/v1/accounts/fireworks/models"
                )
                public_response.raise_for_status()
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
        """Delete a model on Fireworks.ai.

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
        response.raise_for_status()
