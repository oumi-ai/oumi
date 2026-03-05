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

"""Parasail deployment client for dedicated inference endpoints.

Parasail is a distributed inference platform with two modes:
- Serverless: Run any HuggingFace model instantly via OpenAI-compatible API.
- Dedicated Endpoints: Deploy private HuggingFace models with auto-scaling.

Models are referenced by HuggingFace ID or URL — no weight uploads needed.

References:
- Parasail Docs: https://docs.parasail.io
- Dedicated Endpoint Management API:
  https://docs.parasail.io/parasail-docs/dedicated-instance/dedicated-endpoint-management-api
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from oumi.deploy.parasail_api import (
    CreateDeploymentRequest,
    DeploymentResponse,
    DeviceConfig,
    ParasailDeploymentStatus,
    ParasailScaleDownPolicy,
    SupportCheckResponse,
)

logger = logging.getLogger(__name__)

_CONTROL_BASE_URL = "https://api.parasail.io/api/v1"
_INFERENCE_BASE_URL = "https://api.parasail.io/v1"

_PARASAIL_STATE_MAP: dict[ParasailDeploymentStatus, EndpointState] = {
    ParasailDeploymentStatus.ONLINE: EndpointState.RUNNING,
    ParasailDeploymentStatus.STARTING: EndpointState.STARTING,
    ParasailDeploymentStatus.PAUSED: EndpointState.STOPPED,
    ParasailDeploymentStatus.STOPPING: EndpointState.STOPPING,
    ParasailDeploymentStatus.OFFLINE: EndpointState.STOPPED,
}


def _is_huggingface_url(source: str) -> bool:
    return source.startswith("https://huggingface.co/") or source.startswith(
        "http://huggingface.co/"
    )


def _is_huggingface_repo_id(source: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.\-]+$", source))


def _validate_model_source(model_source: str) -> None:
    """Validates that model_source is a HuggingFace repo ID or URL.

    Raises:
        ValueError: If model_source is not supported by Parasail.
    """
    if model_source.startswith("s3://"):
        raise ValueError(
            f"Parasail does not support S3 model sources ('{model_source}'). "
            "Provide a HuggingFace repo ID or HuggingFace URL instead."
        )
    if model_source.startswith("gs://"):
        raise ValueError(
            f"Parasail does not support GCS model sources ('{model_source}'). "
            "Provide a HuggingFace repo ID or HuggingFace URL instead."
        )
    if model_source.startswith("az://") or model_source.startswith("abfs://"):
        raise ValueError(
            f"Parasail does not support Azure Blob storage ('{model_source}'). "
            "Provide a HuggingFace repo ID or HuggingFace URL instead."
        )
    if _is_huggingface_url(model_source):
        return
    if model_source.startswith("https://") or model_source.startswith("http://"):
        raise ValueError(
            f"Parasail does not support deploying models from arbitrary URLs "
            f"('{model_source}'). Provide a HuggingFace repo ID or HuggingFace URL."
        )
    if Path(model_source).is_dir() or Path(model_source).is_file():
        raise ValueError(
            "Parasail does not support local model uploads. "
            "Provide a HuggingFace repo ID (e.g., 'meta-llama/Llama-3-8B') or "
            "HuggingFace URL. Parasail pulls models directly from HuggingFace."
        )
    if not _is_huggingface_repo_id(model_source):
        raise ValueError(
            f"Unrecognized model source: '{model_source}'. "
            "Parasail accepts a HuggingFace repo ID "
            "(e.g., 'Qwen/Qwen2.5-72B-Instruct') or a HuggingFace URL."
        )


class ParasailDeploymentClient(BaseDeploymentClient):
    """Parasail deployment client for dedicated inference endpoints.

    Parasail deploys HuggingFace models directly — no weight uploads needed.
    Models are referenced by HuggingFace ID or URL.

    Authentication: ``PARASAIL_API_KEY`` environment variable.
    """

    provider = DeploymentProvider.PARASAIL

    def __init__(self, api_key: str | None = None):
        """Initialize the Parasail client.

        Args:
            api_key: Parasail API key. Resolved from ``PARASAIL_API_KEY`` env var
                if not provided.

        Raises:
            ValueError: If no API key is found.
        """
        self._api_key = api_key or os.environ.get("PARASAIL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Parasail API key required. Set the PARASAIL_API_KEY environment "
                "variable or pass api_key to the constructor."
            )
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ParasailDeploymentClient":
        """Opens the underlying HTTP session."""
        self._http_client = httpx.AsyncClient(
            base_url=_CONTROL_BASE_URL,
            headers=self._headers,
            timeout=60.0,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Closes the underlying HTTP session."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError(
                "ParasailDeploymentClient must be used as an async context manager."
            )
        return self._http_client

    # -------------------------------------------------------------------------
    # Model methods
    # -------------------------------------------------------------------------

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        progress_callback: Any | None = None,
    ) -> UploadedModel:
        """Validates the HuggingFace model source and returns metadata.

        Parasail does not require uploading weights — models are pulled directly
        from HuggingFace. This method validates the model source is a supported
        HuggingFace identifier and optionally checks model compatibility via
        the Parasail ``/dedicated/support`` API.

        Args:
            model_source: HuggingFace repo ID (e.g., ``Qwen/Qwen2.5-72B-Instruct``)
                or HuggingFace URL.
            model_name: Display name (informational only).
            model_type: Must be ``FULL`` (Parasail does not support LoRA adapters).
            base_model: Unused; included for interface compatibility.
            progress_callback: Unused; included for interface compatibility.

        Returns:
            :class:`UploadedModel` with ``provider_model_id`` set to *model_source*.

        Raises:
            ValueError: If *model_source* is not a valid HuggingFace identifier,
                or if ``model_type`` is ``ADAPTER``.
        """
        _ = progress_callback
        _validate_model_source(model_source)

        if model_type == ModelType.ADAPTER:
            raise ValueError(
                "Parasail does not support LoRA adapter deployments. "
                "Only full model deployments (model_type='full') are supported."
            )

        logger.info(f"Checking Parasail model compatibility: {model_source}")
        response = await self._client.get(
            "/dedicated/support",
            params={"modelName": model_source, "modelAccessKey": ""},
        )
        if response.status_code == 200:
            support = SupportCheckResponse.model_validate(response.json())
            if not support.supported:
                reason = support.error_message or "unknown reason"
                logger.warning(
                    f"Parasail reports model '{model_source}' may not be "
                    f"supported: {reason}"
                )
        else:
            logger.warning(
                "Could not verify model compatibility "
                f"(status {response.status_code}). Proceeding anyway."
            )

        return UploadedModel(provider_model_id=model_source, status="ready")

    async def get_model_status(self, model_id: str) -> str:
        """Returns ``"ready"`` — Parasail models are HuggingFace-hosted."""
        _validate_model_source(model_id)
        return "ready"

    async def list_models(
        self, include_public: bool = False, organization: str | None = None
    ) -> list[Model]:
        """Lists dedicated deployments as model records.

        Parasail has no separate model registry; this returns the currently
        deployed models derived from ``list_endpoints()``.
        """
        endpoints = await self.list_endpoints()
        return [
            Model(
                model_id=ep.model_id,
                model_name=ep.display_name or ep.model_id,
                status=ep.state.value.lower(),
                provider=self.provider,
                model_type=ModelType.FULL,
                created_at=ep.created_at,
            )
            for ep in endpoints
        ]

    async def delete_model(self, model_id: str) -> None:
        """Not supported — models are HuggingFace-hosted, not managed by Parasail."""
        raise NotImplementedError(
            "Parasail does not manage model storage. Models are hosted on "
            "HuggingFace. To remove a model from Parasail, delete the associated "
            "dedicated endpoint with delete_endpoint()."
        )

    # -------------------------------------------------------------------------
    # Endpoint methods
    # -------------------------------------------------------------------------

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
        model_access_key: str | None = None,
        context_length: int | None = None,
        scale_down_policy: ParasailScaleDownPolicy | None = None,
        scale_down_threshold_ms: int | None = None,
    ) -> Endpoint:
        """Creates a dedicated Parasail endpoint for a HuggingFace model.

        Fetches compatible device configurations from ``/dedicated/devices``,
        selects the requested hardware, then POSTs to ``/dedicated/deployments``.

        Args:
            model_id: HuggingFace repo ID or URL.
            hardware: Desired GPU type and count.
            autoscaling: Replica configuration (``min_replicas`` used as replica count).
            display_name: Unique deployment name (lowercase letters, numbers, dashes).
            model_access_key: HuggingFace token for private models.
            context_length: Override context window length.
            scale_down_policy: Auto-scaling policy.
            scale_down_threshold_ms: Idle threshold in ms before scaling down.

        Returns:
            Created :class:`Endpoint`.

        Raises:
            ValueError: If the requested hardware is not available for the model.
            httpx.HTTPStatusError: If the Parasail API returns an error.
        """
        _validate_model_source(model_id)

        deployment_name = _to_deployment_name(display_name or model_id)
        replicas = max(1, autoscaling.min_replicas)

        logger.info(
            f"Fetching device configs for model '{model_id}' "
            f"(device={hardware.accelerator}, count={hardware.count})"
        )
        devices = await self._get_and_select_device_configs(
            model_id=model_id,
            desired_device=hardware.accelerator,
            desired_count=hardware.count,
            model_access_key=model_access_key or "",
        )

        request = CreateDeploymentRequest(
            deploymentName=deployment_name,
            modelName=model_id,
            deviceConfigs=devices,
            replicas=replicas,
            scaleDownPolicy=scale_down_policy,
            scaleDownThreshold=scale_down_threshold_ms,
            draftModelName=None,
            contextLength=context_length,
            modelAccessKey=model_access_key,
        )

        logger.info(f"Creating Parasail deployment '{deployment_name}'...")
        response = await self._client.post(
            "/dedicated/deployments", json=request.to_api_dict()
        )
        response.raise_for_status()
        deployment = DeploymentResponse.model_validate(response.json())
        return _to_endpoint(deployment)

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Gets details of a dedicated endpoint.

        Args:
            endpoint_id: Numeric deployment ID (as a string).

        Returns:
            :class:`Endpoint` with current status.
        """
        response = await self._client.get(f"/dedicated/deployments/{endpoint_id}")
        response.raise_for_status()
        deployment = DeploymentResponse.model_validate(response.json())
        return _to_endpoint(deployment)

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Updates a dedicated endpoint's replica count.

        Parasail's update flow requires fetching the current deployment and
        PUTting the modified object back. Only replica count is updated via
        the ``autoscaling`` parameter; hardware changes require a new endpoint.

        Args:
            endpoint_id: Numeric deployment ID (as a string).
            autoscaling: If provided, updates the replica count to
                ``autoscaling.min_replicas``.
            hardware: Ignored (hardware changes require a new deployment).

        Returns:
            Updated :class:`Endpoint`.
        """
        get_response = await self._client.get(f"/dedicated/deployments/{endpoint_id}")
        get_response.raise_for_status()
        current_raw = get_response.json()

        if autoscaling is not None:
            current_raw["replicas"] = max(1, autoscaling.min_replicas)

        put_response = await self._client.put(
            f"/dedicated/deployments/{endpoint_id}", json=current_raw
        )
        put_response.raise_for_status()
        deployment = DeploymentResponse.model_validate(put_response.json())
        return _to_endpoint(deployment)

    async def delete_endpoint(self, endpoint_id: str, *, force: bool = False) -> None:
        """Permanently deletes a dedicated endpoint.

        Args:
            endpoint_id: Numeric deployment ID (as a string).
            force: Unused; Parasail always performs a hard delete.
        """
        response = await self._client.delete(f"/dedicated/deployments/{endpoint_id}")
        response.raise_for_status()
        logger.info(f"Parasail deployment {endpoint_id} deleted.")

    async def list_endpoints(self) -> list[Endpoint]:
        """Lists all dedicated endpoints for this account."""
        response = await self._client.get("/dedicated/deployments")
        response.raise_for_status()
        items: list[dict] = response.json()
        return [_to_endpoint(DeploymentResponse.model_validate(item)) for item in items]

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """Lists available hardware configurations for Parasail.

        If *model_id* is provided, queries the ``/dedicated/devices`` API to
        return only hardware compatible with that model.  Otherwise queries
        without a model filter to get the full catalogue.

        Args:
            model_id: Optional HuggingFace repo ID to filter compatible hardware.

        Returns:
            List of :class:`HardwareConfig` objects.
        """
        params: dict[str, str] = {"engineName": "VLLM", "modelAccessKey": ""}
        if model_id is not None:
            _validate_model_source(model_id)
            params["modelName"] = model_id

        response = await self._client.get("/dedicated/devices", params=params)
        response.raise_for_status()
        devices = [DeviceConfig.model_validate(d) for d in response.json()]
        return [HardwareConfig(accelerator=d.device, count=d.count) for d in devices]

    # -------------------------------------------------------------------------
    # Start / stop (Parasail pause/resume)
    # -------------------------------------------------------------------------

    async def start_endpoint(self, endpoint_id: str, min_replicas: int = 1) -> Endpoint:
        """Resumes a paused / offline endpoint.

        Args:
            endpoint_id: Numeric deployment ID (as a string).
            min_replicas: Unused; Parasail resumes with the original config.

        Returns:
            Updated :class:`Endpoint`.
        """
        response = await self._client.post(
            f"/dedicated/deployments/{endpoint_id}/resume"
        )
        response.raise_for_status()
        return await self.get_endpoint(endpoint_id)

    async def stop_endpoint(self, endpoint_id: str) -> Endpoint:
        """Pauses a running endpoint to save costs.

        Args:
            endpoint_id: Numeric deployment ID (as a string).

        Returns:
            Updated :class:`Endpoint`.
        """
        response = await self._client.post(
            f"/dedicated/deployments/{endpoint_id}/pause"
        )
        response.raise_for_status()
        return await self.get_endpoint(endpoint_id)

    # -------------------------------------------------------------------------
    # Inference auth
    # -------------------------------------------------------------------------

    def _get_inference_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def _get_and_select_device_configs(
        self,
        model_id: str,
        desired_device: str,
        desired_count: int,
        model_access_key: str = "",
    ) -> list[DeviceConfig]:
        """Fetches device configs from Parasail and marks the desired one selected.

        Raises:
            ValueError: If no matching device config is found.
            httpx.HTTPStatusError: If the API call fails.
        """
        response = await self._client.get(
            "/dedicated/devices",
            params={
                "engineName": "VLLM",
                "modelName": model_id,
                "modelAccessKey": model_access_key,
            },
        )
        response.raise_for_status()
        devices = [DeviceConfig.model_validate(d) for d in response.json()]

        matched = False
        for d in devices:
            is_match = d.device == desired_device and d.count == desired_count
            d.selected = is_match
            matched = matched or is_match

        if not matched:
            available = [f"{d.device} x{d.count}" for d in devices]
            raise ValueError(
                f"No matching hardware config found for "
                f"'{desired_device}' x{desired_count}. "
                f"Available configurations: {available}"
            )

        return devices


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_endpoint(dep: DeploymentResponse) -> Endpoint:
    """Converts a :class:`DeploymentResponse` to an :class:`Endpoint`."""
    state = _PARASAIL_STATE_MAP.get(dep.deployment_status, EndpointState.PENDING)

    selected = dep.selected_device
    hw = (
        HardwareConfig(accelerator=selected.device, count=selected.count)
        if selected
        else HardwareConfig(accelerator="unknown", count=1)
    )

    replicas = dep.replicas
    asc = AutoscalingConfig(min_replicas=replicas, max_replicas=replicas)

    endpoint_url = (
        f"{_INFERENCE_BASE_URL}/chat/completions" if dep.external_alias else None
    )

    created_at: datetime = dep.created_at or datetime.now(tz=timezone.utc)

    return Endpoint(
        endpoint_id=str(dep.id),
        provider=DeploymentProvider.PARASAIL,
        model_id=dep.model_name,
        endpoint_url=endpoint_url,
        state=state,
        hardware=hw,
        autoscaling=asc,
        created_at=created_at,
        display_name=dep.deployment_name,
        inference_model_name=dep.external_alias,
    )


def _to_deployment_name(raw: str) -> str:
    """Converts an arbitrary string to a valid Parasail deployment name.

    Parasail deployment names must contain only lowercase letters, numbers,
    and dashes.
    """
    name = raw.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = name.strip("-")
    return name or "oumi-deployment"
