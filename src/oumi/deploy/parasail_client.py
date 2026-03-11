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

Parasail is a distributed inference platform. This client supports **only**
dedicated endpoints — serverless inference is NOT supported.

- Serverless: NOT supported by this client.
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
from oumi.deploy.utils import (
    is_huggingface_repo_id,
    is_huggingface_url,
    resolve_hf_token,
    warn_if_private_model_missing_token,
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


def _validate_endpoint_id(endpoint_id: str) -> None:
    """Validates that endpoint_id is a Parasail deployment ID, not a model name.

    Parasail dedicated deployment IDs are numeric strings (e.g., ``"33665"``).
    This guard catches the common mistake of passing a HuggingFace model
    identifier where a deployment ID is expected.

    Raises:
        ValueError: If endpoint_id looks like a HuggingFace model name or URL.
    """
    if is_huggingface_repo_id(endpoint_id) or is_huggingface_url(endpoint_id):
        raise ValueError(
            f"'{endpoint_id}' appears to be a HuggingFace model identifier, "
            "not a Parasail deployment ID. "
            "Provide a numeric deployment ID (e.g., '33665') — you can find it in "
            "the Parasail SaaS URL: https://www.saas.parasail.io/dedicated/<id>, "
            "or by running `oumi deploy list --provider parasail`."
        )


def _validate_model_source(model_source: str) -> None:
    """Validates that model_source is a HuggingFace repo ID or URL.

    Raises:
        NotImplementedError: If model_source looks like a Parasail serverless model
            name.
        ValueError: If model_source is not supported by Parasail.
    """
    if model_source.startswith("parasail-"):
        raise NotImplementedError(
            f"Parasail serverless inference is not supported ('{model_source}'). "
            "This client only manages dedicated endpoints. "
            "Provide a HuggingFace repo ID (e.g., 'deepseek-ai/DeepSeek-R1') "
            "and create a dedicated endpoint with `oumi deploy create-endpoint`."
        )
    _unsupported = {
        "s3://": "S3",
        "gs://": "GCS",
        "az://": "Azure Blob",
        "abfs://": "Azure Blob",
    }
    for prefix, label in _unsupported.items():
        if model_source.startswith(prefix):
            raise ValueError(
                f"Parasail does not support {label} model sources ('{model_source}'). "
                "Provide a HuggingFace repo ID or HuggingFace URL instead."
            )
    if is_huggingface_url(model_source):
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
    if not is_huggingface_repo_id(model_source):
        raise ValueError(
            f"Unrecognized model source: '{model_source}'. "
            "Parasail accepts a HuggingFace repo ID "
            "(e.g., 'Qwen/Qwen2.5-72B-Instruct') or a HuggingFace URL."
        )


class ParasailDeploymentClient(BaseDeploymentClient):
    """Parasail deployment client for dedicated inference endpoints.

    Parasail deploys HuggingFace models directly — no weight uploads needed.
    Models are referenced by HuggingFace ID or URL.

    Only dedicated endpoints are supported. Parasail serverless inference
    (model names such as ``parasail-deepseek-r1``) is NOT supported.

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
        """Validates the HuggingFace model source and checks compatibility.

        Args:
            model_source: HuggingFace repo ID or URL.
            model_name: Display name (informational only).
            model_type: ``FULL`` or ``ADAPTER``.
            base_model: Informational only; Parasail resolves from the HF repo.
            progress_callback: Unused; interface compatibility.

        Returns:
            :class:`UploadedModel` with ``provider_model_id`` set to *model_source*.

        Raises:
            ValueError: If *model_source* is not a valid HuggingFace identifier.
        """
        _validate_model_source(model_source)
        hf_token = resolve_hf_token()
        warn_if_private_model_missing_token(model_source, hf_token)

        if model_type == ModelType.ADAPTER:
            logger.info(
                "LoRA adapter deployment: Parasail will auto-detect the PEFT "
                "format and resolve the base model from adapter_config.json "
                "in the HuggingFace repo '%s'.",
                model_source,
            )

        logger.info(
            "Note: Parasail does not support model uploads. "
            "Models are pulled directly from HuggingFace at deployment time."
        )
        logger.info(f"Checking Parasail model compatibility: {model_source}")
        response = await self._client.get(
            "/dedicated/support",
            params={"modelName": model_source, "modelAccessKey": hf_token},
        )
        if response.status_code == 200:
            support = SupportCheckResponse.model_validate(response.json())
            error_msgs = [
                m for m in support.messages if (m.level or "").upper() == "ERROR"
            ]
            if error_msgs:
                raise ValueError(
                    f"Parasail model validation failed for '{model_source}': "
                    + "; ".join(m.content for m in error_msgs)
                )
            if not support.supported:
                reason = (
                    support.error_message
                    or (support.messages[0].content if support.messages else None)
                    or "unknown reason"
                )
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
        hf_token = resolve_hf_token(model_access_key)
        warn_if_private_model_missing_token(model_id, hf_token)

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
            model_access_key=hf_token,
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
            modelAccessKey=hf_token or None,
        )

        logger.info(f"Creating Parasail deployment '{deployment_name}'...")
        response = await self._client.post(
            "/dedicated/deployments", json=request.to_api_dict()
        )
        response.raise_for_status()
        raw = response.json()
        deployment = DeploymentResponse.model_validate(raw)
        endpoint = _to_endpoint(deployment)
        logger.debug(
            "create_endpoint result: id=%s, parasail_status=%s, mapped_state=%s",
            deployment.id,
            deployment.deployment_status,
            endpoint.state,
        )
        return endpoint

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Gets details of a dedicated endpoint.

        Args:
            endpoint_id: Numeric deployment ID (as a string).

        Returns:
            :class:`Endpoint` with current status.

        Raises:
            ValueError: If endpoint_id is a HuggingFace model identifier.
        """
        _validate_endpoint_id(endpoint_id)
        response = await self._client.get(f"/dedicated/deployments/{endpoint_id}")
        response.raise_for_status()
        raw = response.json()
        deployment = DeploymentResponse.model_validate(raw)
        logger.debug(
            "get_endpoint(%s): parasail_status=%s, mapped_state=%s",
            endpoint_id,
            deployment.deployment_status,
            _PARASAIL_STATE_MAP.get(
                deployment.deployment_status, EndpointState.PENDING
            ),
        )
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

        Raises:
            ValueError: If endpoint_id is a HuggingFace model identifier.
        """
        _validate_endpoint_id(endpoint_id)
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

        Raises:
            ValueError: If endpoint_id is a HuggingFace model identifier.
        """
        _validate_endpoint_id(endpoint_id)
        response = await self._client.delete(f"/dedicated/deployments/{endpoint_id}")
        response.raise_for_status()
        logger.info(f"Parasail deployment {endpoint_id} deleted.")

    async def list_endpoints(self) -> list[Endpoint]:
        """Lists all dedicated endpoints for this account."""
        response = await self._client.get("/dedicated/deployments")
        response.raise_for_status()
        items: list[dict] = response.json()
        return [_to_endpoint(DeploymentResponse.model_validate(item)) for item in items]

    async def endpoint_exists(self, endpoint_id: str) -> bool:
        """Returns True if a dedicated endpoint with this ID exists.

        Args:
            endpoint_id: Numeric deployment ID (as a string).

        Raises:
            ValueError: If endpoint_id is a HuggingFace model identifier.
        """
        _validate_endpoint_id(endpoint_id)
        # Uses list_endpoints() because GET /dedicated/deployments/{id}
        # returns ambiguous error codes for invalid IDs vs. not-found.
        endpoints = await self.list_endpoints()
        return any(ep.endpoint_id == endpoint_id for ep in endpoints)

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """Lists available hardware configurations for Parasail.

        Queries the ``/dedicated/devices`` API for hardware compatible with
        *model_id*.  Parasail's API requires a model name — calling without
        one raises :class:`ValueError`.

        Args:
            model_id: HuggingFace repo ID (e.g., ``Qwen/Qwen2.5-72B-Instruct``)
                or HuggingFace URL. **Required** for Parasail.

        Returns:
            List of :class:`HardwareConfig` objects compatible with the model.

        Raises:
            ValueError: If *model_id* is not provided or is not a valid
                HuggingFace identifier.
        """
        if model_id is None:
            raise ValueError(
                "Parasail requires a model ID to list compatible hardware. "
                "Provide a HuggingFace repo ID, e.g. 'Qwen/Qwen2.5-72B-Instruct'."
            )
        _validate_model_source(model_id)
        hf_token = resolve_hf_token()
        warn_if_private_model_missing_token(model_id, hf_token)
        devices = await self._fetch_device_configs(model_id, hf_token)
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

        Raises:
            ValueError: If endpoint_id is a HuggingFace model identifier.
        """
        _validate_endpoint_id(endpoint_id)
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

        Raises:
            ValueError: If endpoint_id is a HuggingFace model identifier.
        """
        _validate_endpoint_id(endpoint_id)
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

    async def _fetch_device_configs(
        self, model_id: str, model_access_key: str = ""
    ) -> list[DeviceConfig]:
        """Fetches device configs from the ``/dedicated/devices`` API."""
        response = await self._client.get(
            "/dedicated/devices",
            params={
                "engineName": "VLLM",
                "modelName": model_id,
                "modelAccessKey": model_access_key,
            },
        )
        response.raise_for_status()
        return [DeviceConfig.model_validate(d) for d in response.json()]

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
        devices = await self._fetch_device_configs(model_id, model_access_key)

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
