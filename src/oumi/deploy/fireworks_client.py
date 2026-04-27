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
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import requests as _requests

from oumi.deploy.base_client import (
    AutoscalingConfig,
    BaseDeploymentClient,
    DeploymentProvider,
    Endpoint,
    EndpointState,
    FileResolver,
    HardwareConfig,
    Model,
    ModelType,
    ProgressCallback,
    UploadedModel,
)
from oumi.deploy.fireworks_api import (
    FW_STATE_TO_ENDPOINT,
    BaseModelDetailsCheckpointFormat,
    DeploymentPrecision,
    GatewayAcceleratorType,
    GatewayBaseModelDetails,
    GatewayCreateModelBody,
    GatewayDeployment,
    GatewayGetModelUploadEndpointBody,
    GatewayGetModelUploadEndpointResponse,
    GatewayListDeploymentsResponse,
    GatewayListModelsResponse,
    GatewayModel,
    GatewayModelKind,
    GatewayPEFTDetails,
    GatewayPrepareModelBody,
)
from oumi.deploy.fireworks_errors import classify_fireworks_invalid_request
from oumi.deploy.utils import raise_api_error

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


# Mapping from Oumi-standard accelerator names (lowercase) to Fireworks REST API
# ``acceleratorType`` enum values, derived from GatewayAcceleratorType.
# See: https://docs.fireworks.ai/api-reference/create-deployment#body-accelerator-type
FIREWORKS_ACCELERATORS: dict[str, str] = {
    member.value.lower(): member.value
    for member in GatewayAcceleratorType
    if member != GatewayAcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED
}

# Reverse mapping
FIREWORKS_ACCELERATORS_REVERSE = {v: k for k, v in FIREWORKS_ACCELERATORS.items()}

# Hardcoded hardware list version
# (Fireworks has no public hardware discovery API as of 2026-01)
FIREWORKS_HARDWARE_LIST_VERSION = "2026-01"


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
            "Model ID must be between 1 and 63 characters; got empty string."
        )
    if len(model_id) > 63:
        raise FireworksInvalidModelIdError(
            f"Model ID must be at most 63 characters; "
            f"'{model_id}' has {len(model_id)} characters."
        )
    invalid_chars = {
        c for c in model_id if not (c.islower() or c.isdigit() or c == "-")
    }
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


_raise_api_error = raise_api_error


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
        self._api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self._api_key:
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
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=120.0,
        )

    async def __aenter__(self) -> "FireworksDeploymentClient":
        """Enters the async context manager.

        Narrows the base ``__aenter__`` return type to this subclass so
        callers using ``async with`` get full access to Fireworks-specific
        methods without a cast.
        """
        return self

    async def close(self) -> None:
        """Closes the HTTP client and releases resources."""
        await self._client.aclose()

    def _get_inference_auth_headers(self) -> dict[str, str]:
        """Returns auth headers for inference (test_endpoint)."""
        return {"Authorization": f"Bearer {self._api_key}"}

    @staticmethod
    def _check_response(response: httpx.Response, context: str) -> None:
        """Raises if the response indicates an error.

        Threads the Fireworks 4xx classifier so 400/422 responses are
        refined into Fireworks-specific subclasses when their detail
        strings match.
        """
        if not response.is_success:
            _raise_api_error(
                response,
                context=context,
                classify_4xx=classify_fireworks_invalid_request,
            )

    @staticmethod
    async def _notify(
        callback: ProgressCallback | None,
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

    def _parse_deployment(self, deployment: GatewayDeployment) -> Endpoint:
        """Converts a Fireworks ``GatewayDeployment`` into an ``Endpoint``."""
        state = (
            FW_STATE_TO_ENDPOINT.get(deployment.state, EndpointState.PENDING)
            if deployment.state
            else EndpointState.PENDING
        )

        hardware = HardwareConfig(
            accelerator=self._from_fireworks_accelerator(
                deployment.accelerator_type.value if deployment.accelerator_type else ""
            ),
            count=deployment.accelerator_count or 1,
        )

        autoscaling = AutoscalingConfig(
            min_replicas=deployment.min_replica_count or 0,
            max_replicas=deployment.max_replica_count or 1,
        )

        name = deployment.name or ""
        endpoint_id = name.split("/")[-1] if "/" in name else name

        # Fireworks Get Deployment API does not return endpointUrl (see
        # https://docs.fireworks.ai/api-reference/get-deployment). For on-demand
        # deployments, inference uses the standard chat completions endpoint with
        # the deployment resource name as the model. See on-demand quickstart:
        # https://docs.fireworks.ai/getting-started/ondemand-quickstart
        endpoint_url = (deployment.model_extra or {}).get("endpointUrl")
        if not endpoint_url:
            endpoint_url = "https://api.fireworks.ai/inference/v1/chat/completions"

        status = deployment.status
        status_code = status.code.value if status and status.code else None
        status_message = status.message if status else None

        return Endpoint(
            endpoint_id=endpoint_id,
            provider=DeploymentProvider.FIREWORKS,
            model_id=deployment.base_model,
            endpoint_url=endpoint_url,
            state=state,
            hardware=hardware,
            autoscaling=autoscaling,
            created_at=deployment.create_time,
            display_name=deployment.display_name,
            inference_model_name=name or None,
            status_code=status_code,
            status_message=status_message,
        )

    @staticmethod
    def _detect_model_type(item: GatewayModel) -> ModelType | None:
        """Infers the ``ModelType`` from a ``GatewayModel``'s kind."""
        kind = item.kind
        if kind in (
            GatewayModelKind.HF_PEFT_ADDON,
            GatewayModelKind.HF_TEFT_ADDON,
        ):
            return ModelType.ADAPTER
        if kind in (GatewayModelKind.CUSTOM_MODEL, GatewayModelKind.HF_BASE_MODEL):
            return ModelType.FULL
        # Fall back to extra ``modelFormat`` field (not in the OpenAPI spec)
        model_format = str((item.model_extra or {}).get("modelFormat", "")).lower()
        if "lora" in model_format or "adapter" in model_format:
            return ModelType.ADAPTER
        if model_format == "huggingface":
            return ModelType.FULL
        return None

    @classmethod
    def _parse_model(cls, item: GatewayModel) -> Model:
        """Converts a ``GatewayModel`` into a ``Model`` dataclass."""
        model_id = item.name or (item.model_extra or {}).get("id", "")
        display_name = item.display_name or ""
        model_name = (
            display_name
            if display_name
            else (model_id.split("/")[-1] if "/" in model_id else model_id)
        )
        return Model(
            model_id=model_id,
            model_name=model_name,
            status=(item.state.value if item.state else "unknown").lower(),
            provider=DeploymentProvider.FIREWORKS,
            model_type=cls._detect_model_type(item),
            created_at=item.create_time,
            base_model=item.base_model,
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
        progress_callback: ProgressCallback | None = None,
    ) -> UploadedModel:
        """Uploads a model to Fireworks.ai using multi-step flow.

        API Flow (from https://docs.fireworks.ai/models/uploading-custom-models-api):
        1. Create model object with modelId and model structure
        2. Get signed upload URLs (one per file) with file sizes
        3. Upload each file to its signed URL
        4. Wait for GCS propagation then validate the upload

        Args:
            model_source: Path to local model directory containing model weights.
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
            # which files to expect.  For adapter uploads, only PEFT files are
            # included (Fireworks rejects non-adapter files for HF_PEFT_ADDON).
            file_inventory = self._collect_file_inventory(model_dir, model_type)

            # Step 2b: For adapters, read adapter_config.json so we can
            # populate peftDetails with the real r and target_modules.
            # Remaining adapter validation (base_model present, base model
            # exists on Fireworks) runs inside ``_do_upload`` so both entry
            # points share it.
            adapter_config = None
            if model_type == ModelType.ADAPTER:
                adapter_config = self._read_adapter_config(model_dir)
                if adapter_config is None:
                    raise ValueError(
                        f"Model type is 'adapter' but no adapter_config.json "
                        f"found in '{model_dir}'. Ensure the directory contains "
                        f"a valid PEFT/LoRA adapter checkpoint."
                    )

            @asynccontextmanager
            async def local_resolver(filename: str):
                yield model_dir / filename

            return await self._do_upload(
                model_id,
                file_inventory,
                local_resolver,
                model_type,
                base_model,
                adapter_config,
                progress_callback,
            )
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def upload_model_with_resolver(
        self,
        model_name: str,
        file_inventory: dict[str, int],
        file_resolver: FileResolver,
        progress_callback: ProgressCallback | None = None,
        *,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        adapter_config: dict[str, Any] | None = None,
    ) -> UploadedModel:
        """Uploads a model using a pre-computed file inventory and resolver.

        Unlike ``upload_model()`` which requires all files on local disk, this
        method accepts a ``file_inventory`` (filenames → sizes in bytes) and a
        ``file_resolver`` callback that yields each file on demand.  This enables
        streaming from cloud storage one file at a time, keeping peak disk usage
        to the size of a single shard rather than the full model.

        Both FULL and ADAPTER uploads are supported. For ADAPTER uploads the
        caller must supply the adapter's parsed ``adapter_config`` dict and
        the Fireworks-hosted ``base_model`` resource path — callers resolve
        the HF → Fireworks base mapping before calling this method.

        Args:
            model_name: Fireworks model ID (e.g., ``"my-custom-model"``).
            file_inventory: Mapping of relative filename to file size in bytes.
            file_resolver: Async context manager factory.  For each filename,
                ``file_resolver(filename)`` must yield a local ``Path`` to the
                file and clean up after the ``async with`` block exits.
            progress_callback: Optional async progress callback.
            model_type: FULL or ADAPTER. Defaults to FULL.
            base_model: Fireworks-hosted base model resource path, required
                when ``model_type == ADAPTER`` (e.g.
                ``accounts/fireworks/models/llama-v3p1-8b-instruct``).
            adapter_config: Parsed ``adapter_config.json`` dict, required when
                ``model_type == ADAPTER``. Must contain a non-empty
                ``target_modules`` list; ``r`` defaults to 8 if absent.

        Returns:
            ``UploadedModel`` with the Fireworks provider model ID.

        Raises:
            FireworksInvalidModelIdError: If ``model_name`` violates Fireworks
                naming rules.
            ValueError: When ``model_type == ADAPTER`` and ``base_model`` is
                missing or refers to a model that does not exist / is not
                ready on Fireworks, or when ``adapter_config`` is missing /
                has an empty ``target_modules`` list (raised by ``_do_upload``
                / ``_create_model_resource``).
        """
        _validate_fireworks_model_id(model_name)
        return await self._do_upload(
            model_name,
            file_inventory,
            file_resolver,
            model_type,
            base_model,
            adapter_config,
            progress_callback,
        )

    # ------------------------------------------------------------------
    # upload_model — private helpers
    # ------------------------------------------------------------------

    async def _do_upload(
        self,
        model_name: str,
        file_inventory: dict[str, int],
        file_resolver: FileResolver,
        model_type: ModelType,
        base_model: str | None,
        adapter_config: dict[str, Any] | None,
        progress_callback: ProgressCallback | None,
    ) -> UploadedModel:
        """Shared upload implementation: create model → upload files → validate.

        Both ``upload_model`` and ``upload_model_with_resolver`` delegate here
        after resolving their input to a ``(file_inventory, file_resolver)``
        pair. Adapter-specific pre-flight (``base_model`` present and ready
        on Fireworks) runs here so every entry path gets the same guarantees;
        the ``target_modules`` invariant is enforced downstream in
        ``_create_model_resource``.
        """
        if model_type == ModelType.ADAPTER:
            if not base_model:
                raise ValueError(
                    "Adapter uploads require 'base_model'. Provide the "
                    "Fireworks model path of the base model this adapter "
                    "was trained on (e.g. 'accounts/fireworks/models/"
                    "llama-v3p1-8b-instruct')."
                )
            await self._verify_base_model_exists(base_model)

        hf_files = sorted(file_inventory.keys())
        create_payload = await self._create_model_resource(
            model_name,
            model_type,
            base_model,
            progress_callback,
            huggingface_files=hf_files,
            adapter_config=adapter_config,
        )
        await self._upload_model_files_with_resolver(
            model_name,
            file_inventory,
            file_resolver,
            progress_callback,
        )
        await self._wait_and_validate(model_name, progress_callback)
        return UploadedModel(
            provider_model_id=f"accounts/{self.account_id}/models/{model_name}",
            status="validating",
            request_payload=create_payload,
        )

    async def _upload_model_files_with_resolver(
        self,
        model_id: str,
        file_sizes: dict[str, int],
        file_resolver: FileResolver,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Upload files using a resolver that provides each file on demand.

        Fetches signed upload URLs for all files, then iterates them in upload
        order.  For each file, it calls ``file_resolver(filename)`` as an async
        context manager to obtain a local path, uploads the file, and lets the
        resolver clean up (e.g., delete the temp file) before moving on.

        Args:
            model_id: Fireworks model ID used to request signed upload URLs.
            file_sizes: Mapping of relative filename to file size in bytes.
            file_resolver: Async context manager factory yielding a local Path.
            progress_callback: Optional async progress callback.
        """
        total_bytes = sum(file_sizes.values())
        _MB = 1024 * 1024
        if "config.json" in file_sizes:
            logger.info("config.json found (%d bytes)", file_sizes["config.json"])
        else:
            logger.error(
                "config.json NOT found in model files: %s",
                list(file_sizes.keys()),
            )
        logger.info(
            "Uploading %d files (%.1f MB) via resolver",
            len(file_sizes),
            total_bytes / _MB,
        )

        upload_items = await self._get_signed_urls_ordered(model_id, file_sizes)
        total_files = len(upload_items)

        await self._notify(
            progress_callback,
            "uploading",
            f"Starting upload of {total_files} files ({total_bytes / _MB:.1f} MB)",
            {"total_files": total_files, "total_bytes": total_bytes},
        )

        uploaded_bytes = 0
        for idx, (filename, signed_url) in enumerate(upload_items, 1):
            file_size = file_sizes[filename]
            logger.info(
                "[%d/%d] Uploading %s (%.2f MB)",
                idx,
                total_files,
                filename,
                file_size / _MB,
            )

            async with file_resolver(filename) as file_path:
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
            "All %d files uploaded via resolver (%.1f MB total)",
            total_files,
            total_bytes / _MB,
        )
        await self._notify(
            progress_callback,
            "uploading",
            f"All {total_files} files uploaded ({total_bytes / _MB:.1f} MB total)",
            {"status": "complete", "total_files": total_files},
        )

    async def _create_model_resource(
        self,
        model_id: str,
        model_type: ModelType,
        base_model: str | None,
        progress_callback: ProgressCallback | None,
        huggingface_files: list[str] | None = None,
        adapter_config: dict[str, Any] | None = None,
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
            adapter_config: Parsed adapter_config.json dict (for LoRA adapters).
                Used to populate peftDetails with the real ``r`` and
                ``target_modules`` values.
        """
        logger.info(
            "Creating model on Fireworks: model_id=%s, model_type=%s, base_model=%s",
            model_id,
            model_type,
            base_model,
        )

        if model_type == ModelType.ADAPTER and base_model:
            lora_r = (adapter_config or {}).get("r", 8)
            target_modules = (adapter_config or {}).get("target_modules", [])
            if not target_modules:
                raise ValueError(
                    "Cannot create LoRA adapter model: target_modules is empty. "
                    "Ensure the model directory contains an adapter_config.json "
                    "with a non-empty 'target_modules' list."
                )
            logger.info(
                "LoRA adapter config: r=%d, target_modules=%s", lora_r, target_modules
            )
            body = GatewayCreateModelBody(
                modelId=model_id,
                model=GatewayModel(
                    kind=GatewayModelKind.HF_PEFT_ADDON,
                    peftDetails=GatewayPEFTDetails(
                        baseModel=base_model,
                        r=lora_r,
                        targetModules=target_modules,
                    ),
                ),
            )
        else:
            body = GatewayCreateModelBody(
                modelId=model_id,
                model=GatewayModel(
                    kind=GatewayModelKind.HF_BASE_MODEL,
                    baseModelDetails=GatewayBaseModelDetails(
                        checkpointFormat=BaseModelDetailsCheckpointFormat.HUGGINGFACE,
                        worldSize=1,
                        huggingfaceFiles=huggingface_files,
                    ),
                ),
            )

        create_payload = body.model_dump(by_alias=True, exclude_none=True)

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

        created = GatewayModel.model_validate(response.json())
        logger.info("Model created: %s", created.name)

        await self._notify(
            progress_callback,
            "creating",
            f"Model resource created on Fireworks: {model_id}",
            {"provider_model_id": created.name},
        )
        return create_payload

    @staticmethod
    def _check_model_source_supported(model_source: str) -> None:
        """Validates that model_source is a supported source for Fireworks uploads.

        Fireworks only supports models stored on local disk (a HuggingFace model
        downloaded to a local directory, or an Oumi training checkpoint). All other
        source types are rejected here before any API calls are made, preventing
        orphaned model resources from being created.

        Args:
            model_source: The model source string provided by the user.

        Raises:
            ValueError: With a descriptive message indicating the detected source
                type, why it is not supported, and how to remediate.
        """
        if model_source.startswith("s3://"):
            raise ValueError(
                f"Fireworks: we do not support deploying models from S3"
                f" ('{model_source}'). Download the model to a local directory"
                " first, then provide the local path."
            )

        if model_source.startswith("gs://"):
            raise ValueError(
                f"Fireworks: we do not support deploying models from GCS"
                f" ('{model_source}'). Download the model to a local directory"
                " first, then provide the local path."
            )

        if model_source.startswith(("az://", "abfs://")):
            raise ValueError(
                "Fireworks: we do not support deploying models from Azure Blob "
                f"storage ('{model_source}'). Download the model to a local "
                "directory first, then provide the local path."
            )

        if model_source.startswith(("http://", "https://")):
            from urllib.parse import urlparse  # noqa: PLC0415

            parsed = urlparse(model_source)
            hostname = (parsed.hostname or "").lower().removeprefix("www.")
            if hostname == "huggingface.co":
                raise ValueError(
                    "Fireworks: we do not support deploying models from HuggingFace"
                    " URLs. Download the model first to a local directory, then"
                    " provide the local path."
                )
            raise ValueError(
                f"Fireworks: we do not support deploying models from remote URLs "
                f"('{model_source}'). Only local directory paths are supported."
            )

        model_path = Path(model_source)

        # Detect HuggingFace repo IDs: non-absolute paths containing "/" that do not
        # exist on disk (e.g. "meta-llama/Llama-3-8B", "Qwen/Qwen2.5-72B-Instruct").
        if (
            not model_source.startswith("/")
            and "/" in model_source
            and not model_path.exists()
        ):
            raise ValueError(
                f"Fireworks: we do not support deploying models directly from a"
                f" HuggingFace repo ID ('{model_source}'). Download the model"
                f" first using 'huggingface-cli download {model_source}"
                " --local-dir /path/to/local/dir', then provide the local"
                " directory path."
            )

        if not model_path.exists():
            raise ValueError(
                f"Model path '{model_source}' does not exist. "
                "Provide a valid local directory containing the model weights."
            )

        if not model_path.is_dir():
            raise ValueError(
                f"Model path '{model_source}' is a file, not a directory. "
                "Provide the parent directory containing all model weight files."
            )

    @staticmethod
    def _read_adapter_config(model_dir: Path) -> dict[str, Any] | None:
        """Reads adapter_config.json from a model directory if present.

        Returns:
            Parsed JSON dict, or None if the file doesn't exist.

        Raises:
            ValueError: If the file exists but cannot be parsed.
        """
        config_path = model_dir / "adapter_config.json"
        if not config_path.is_file():
            return None
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Failed to read adapter config at '{config_path}': {exc}"
            ) from exc

    async def _verify_base_model_exists(self, base_model: str) -> None:
        """Verifies that the base model exists on Fireworks before creating an adapter.

        Called as a pre-flight check during adapter uploads so that a missing
        base model is caught *before* any model resource is created, avoiding
        orphaned adapter entries.

        Args:
            base_model: Full Fireworks model path
                (e.g. ``accounts/oumi/models/qwen3-1-7b-hf-vanilla``).

        Raises:
            ValueError: If the base model does not exist or is not ready.
        """
        model = await self.get_model(base_model)
        if model is None:
            raise ValueError(
                f"Base model '{base_model}' not found on Fireworks. "
                f"Upload the base model first with 'oumi deploy upload "
                f"--model-type full', then retry the adapter upload."
            )
        state = (model.state.value if model.state else "unknown").lower()
        if state != "ready":
            raise ValueError(
                f"Base model '{base_model}' exists but is not ready "
                f"(state: {state}). Wait for it to reach 'ready' before "
                f"uploading an adapter."
            )
        logger.info("Base model '%s' verified (state: %s)", base_model, state)

    async def _resolve_model_source(
        self,
        model_source: str,
        progress_callback: ProgressCallback | None,
    ) -> tuple[Path, Path | None]:
        """Resolves model_source to a local directory.

        Only local directory paths are supported. All other source types
        (S3, GCS, Azure Blob, HuggingFace URLs and repo IDs, generic remote
        URLs) are rejected by ``_check_model_source_supported`` before this
        method is called.

        Returns:
            (model_dir, None) — no temporary directory is created for local paths.
        """
        return Path(model_source), None

    def _collect_file_inventory(
        self, model_dir: Path, model_type: ModelType = ModelType.FULL
    ) -> dict[str, int]:
        """Builds filenameToSize inventory for the model directory.

        Excludes HuggingFace cache artifacts (e.g. *.lock, *.metadata), paths
        under .cache or huggingface, and training-state files that are not needed
        for inference (optimizer, scheduler, RNG state, trainer bookkeeping).

        For adapter uploads (``HF_PEFT_ADDON``), only adapter-specific files
        are included (``adapter_model.safetensors``, ``adapter_config.json``).
        Fireworks rejects other files (tokenizer, README, etc.) for PEFT addon
        models.

        Args:
            model_dir: Local directory containing model weight files.
            model_type: FULL or ADAPTER.  Adapter uploads are restricted to
                PEFT-specific files only.

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
        _TRAINING_STATE_NAMES = frozenset(
            {
                "optimizer.pt",
                "optimizer.bin",
                "scheduler.pt",
                "trainer_state.json",
                "training_args.bin",
                "scaler.pt",
            }
        )
        # For HF_PEFT_ADDON models, Fireworks only accepts adapter files.
        # Tokenizer files, README, chat templates etc. belong to the base
        # model and are rejected with INVALID_ARGUMENT "unexpected file".
        _ADAPTER_ALLOWED_NAMES = frozenset(
            {
                "adapter_model.safetensors",
                "adapter_model.bin",
                "adapter_config.json",
            }
        )
        is_adapter = model_type == ModelType.ADAPTER

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
                if is_adapter and fname not in _ADAPTER_ALLOWED_NAMES:
                    logger.debug("Skipping non-adapter file for PEFT upload: %s", rel)
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
        body = GatewayGetModelUploadEndpointBody(
            filenameToSize=cast(dict[str, str | int], file_sizes),
            enableResumableUpload=False,
        )
        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/models/{model_id}:getUploadEndpoint",
            json=body.model_dump(by_alias=True, exclude_none=True),
        )
        self._check_response(response, f"get upload endpoint for model '{model_id}'")
        resp = GatewayGetModelUploadEndpointResponse.model_validate(response.json())
        file_upload_urls = resp.filename_to_signed_urls or {}
        if not file_upload_urls:
            raise ValueError("No upload URLs received from Fireworks API.")

        return sorted(file_upload_urls.items(), key=self._upload_order_key)

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
        progress_callback: ProgressCallback | None,
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
        await self._notify(
            progress_callback, "validating", "Validating uploaded model..."
        )

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

        Raises:
            ValueError: If the model does not exist.
        """
        model = await self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' does not exist")
        return model.state.value if model.state else "unknown"

    async def get_model(self, model_id: str) -> GatewayModel | None:
        """Gets the full model resource for a Fireworks model, or None if missing.

        Unlike ``get_model_status`` which returns only the state string, this
        returns the full ``GatewayModel`` pydantic object so callers can
        inspect additional fields (``kind``, ``peft_details``, etc.). Treats
        HTTP 404 as "model does not exist" and returns ``None`` instead of
        raising, so callers can distinguish "missing" from transient errors.

        Args:
            model_id: Fireworks model ID (short ID or full path).

        Returns:
            Parsed ``GatewayModel`` if the model exists, else ``None``.
        """
        model_path = self._model_api_path(model_id)
        response = await self._client.get(model_path)
        if response.status_code == 404:
            return None
        self._check_response(response, f"get model '{model_id}'")
        return GatewayModel.model_validate(response.json())

    async def prepare_model(
        self, model_id: str, precision: str | None = None
    ) -> dict[str, Any]:
        """Prepares a model for deployment (optional precision conversion).

        Args:
            model_id: Fireworks model ID
            precision: Target precision (e.g., "FP16", "FP8", "BF16").
                Case-insensitive; normalised to uppercase for the API.

        Returns:
            Preparation result
        """
        model_path = self._model_api_path(model_id, ":prepare")
        body = GatewayPrepareModelBody(
            precision=DeploymentPrecision(precision.upper()) if precision else None,
        )
        response = await self._client.post(
            model_path,
            json=body.model_dump(by_alias=True, exclude_none=True),
        )
        self._check_response(response, f"prepare model '{model_id}'")
        return response.json()

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
        endpoint_id: str | None = None,
    ) -> Endpoint:
        """Creates an inference endpoint (deployment) for a model.

        Args:
            model_id: Fireworks model ID
            hardware: Hardware configuration
            autoscaling: Autoscaling configuration
            display_name: Optional display name
            endpoint_id: Optional caller-supplied deployment ID. When provided,
                it is passed as the ``deploymentId`` query parameter so the
                resulting deployment has a deterministic resource path
                ``accounts/{account_id}/deployments/{endpoint_id}``. When
                omitted, Fireworks generates a random ID (the default
                behavior).

        Returns:
            Created Endpoint
        """
        deployment = GatewayDeployment(
            baseModel=model_id,
            acceleratorType=cast(
                GatewayAcceleratorType, self._to_fireworks_accelerator(hardware)
            ),
            acceleratorCount=hardware.count,
            minReplicaCount=autoscaling.min_replicas,
            maxReplicaCount=autoscaling.max_replicas,
            displayName=display_name,
        )

        params: dict[str, Any] = {}
        if endpoint_id is not None:
            params["deploymentId"] = endpoint_id

        response = await self._client.post(
            f"/v1/accounts/{self.account_id}/deployments",
            json=deployment.model_dump(by_alias=True, exclude_none=True),
            params=params,
        )
        self._check_response(response, f"create endpoint for model '{model_id}'")

        return self._parse_deployment(GatewayDeployment.model_validate(response.json()))

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

        return self._parse_deployment(GatewayDeployment.model_validate(response.json()))

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
        current = await self.get_endpoint(endpoint_id)

        deployment = GatewayDeployment(
            baseModel=current.model_id,
            minReplicaCount=autoscaling.min_replicas if autoscaling else None,
            maxReplicaCount=autoscaling.max_replicas if autoscaling else None,
            acceleratorType=cast(
                GatewayAcceleratorType | None,
                self._to_fireworks_accelerator(hardware) if hardware else None,
            ),
            acceleratorCount=hardware.count if hardware else None,
        )

        response = await self._client.patch(
            f"/v1/accounts/{self.account_id}/deployments/{endpoint_id}",
            json=deployment.model_dump(by_alias=True, exclude_none=True),
        )
        self._check_response(response, f"update endpoint '{endpoint_id}'")

        return self._parse_deployment(GatewayDeployment.model_validate(response.json()))

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
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {}
            if page_token:
                params["pageToken"] = page_token

            response = await self._client.get(
                f"/v1/accounts/{self.account_id}/deployments",
                params=params,
            )
            self._check_response(response, "list endpoints")
            resp = GatewayListDeploymentsResponse.model_validate(response.json())

            for item in resp.deployments or []:
                endpoints.append(self._parse_deployment(item))

            page_token = resp.next_page_token
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
            HardwareConfig(accelerator=name, count=1) for name in FIREWORKS_ACCELERATORS
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
        response = await self._client.get(f"/v1/accounts/{self.account_id}/models")
        self._check_response(response, "list models")
        data = response.json()

        if isinstance(data, list):
            gw_models = [GatewayModel.model_validate(item) for item in data]
        else:
            resp = GatewayListModelsResponse.model_validate(data)
            gw_models = list(resp.models or [])

        if include_public:
            try:
                public_response = await self._client.get(
                    "/v1/accounts/fireworks/models"
                )
                self._check_response(public_response, "list public models")
                public_data = public_response.json()
                if isinstance(public_data, list):
                    gw_models.extend(
                        GatewayModel.model_validate(item) for item in public_data
                    )
                else:
                    public_resp = GatewayListModelsResponse.model_validate(public_data)
                    gw_models.extend(public_resp.models or [])
            except Exception:
                logger.warning("Failed to fetch public models", exc_info=True)

        return [self._parse_model(item) for item in gw_models]

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
