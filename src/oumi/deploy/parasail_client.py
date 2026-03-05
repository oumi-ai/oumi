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

logger = logging.getLogger(__name__)

_CONTROL_BASE_URL = "https://api.parasail.io/api/v1"
_INFERENCE_BASE_URL = "https://api.parasail.io/v1"

# Parasail status → EndpointState mapping
_PARASAIL_STATE_MAP: dict[str, EndpointState] = {
    "ONLINE": EndpointState.RUNNING,
    "STARTING": EndpointState.STARTING,
    "PAUSED": EndpointState.STOPPED,
    "STOPPING": EndpointState.STOPPING,
    "OFFLINE": EndpointState.STOPPED,
}

# Known Parasail GPU device names
_KNOWN_HARDWARE: list[HardwareConfig] = [
    HardwareConfig(accelerator="H100SXM", count=1),
    HardwareConfig(accelerator="H100SXM", count=2),
    HardwareConfig(accelerator="H100SXM", count=4),
    HardwareConfig(accelerator="H100SXM", count=8),
    HardwareConfig(accelerator="A100SXM", count=1),
    HardwareConfig(accelerator="A100SXM", count=2),
    HardwareConfig(accelerator="A100SXM", count=4),
    HardwareConfig(accelerator="RTX4090", count=1),
    HardwareConfig(accelerator="RTX4090", count=2),
    HardwareConfig(accelerator="RTX4090", count=4),
    HardwareConfig(accelerator="L40S", count=1),
    HardwareConfig(accelerator="L40S", count=2),
]


def _is_huggingface_url(source: str) -> bool:
    return source.startswith("https://huggingface.co/") or source.startswith(
        "http://huggingface.co/"
    )


def _is_huggingface_repo_id(source: str) -> bool:
    """Returns True if source looks like a HuggingFace repo ID (org/model)."""
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
    if os.path.isdir(model_source) or os.path.isfile(model_source):
        raise ValueError(
            f"Parasail does not support local model uploads. "
            "Provide a HuggingFace repo ID (e.g., 'meta-llama/Llama-3-8B') or "
            "HuggingFace URL. Parasail pulls models directly from HuggingFace."
        )
    if not _is_huggingface_repo_id(model_source):
        raise ValueError(
            f"Unrecognized model source: '{model_source}'. "
            "Parasail accepts a HuggingFace repo ID (e.g., 'Qwen/Qwen2.5-72B-Instruct') "
            "or a HuggingFace URL."
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
        self._http_client = httpx.AsyncClient(
            base_url=_CONTROL_BASE_URL,
            headers=self._headers,
            timeout=60.0,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
        from HuggingFace.  This method validates that the model source is a
        supported HuggingFace identifier and optionally checks model compatibility
        via the Parasail ``/dedicated/support`` API.

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

        logger.info(f"Validating Parasail model source: {model_source}")
        response = await self._client.get(
            "/dedicated/support",
            params={"modelName": model_source, "modelAccessKey": ""},
        )
        if response.status_code == 200:
            data = response.json()
            if not data.get("supported", True):
                reason = data.get("errorMessage", "unknown reason")
                logger.warning(
                    f"Parasail reports model '{model_source}' may not be "
                    f"supported: {reason}"
                )
        else:
            logger.warning(
                f"Could not verify model compatibility (status {response.status_code}). "
                "Proceeding anyway."
            )

        return UploadedModel(
            provider_model_id=model_source,
            status="ready",
        )

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
        models: list[Model] = []
        for ep in endpoints:
            models.append(
                Model(
                    model_id=ep.model_id,
                    model_name=ep.display_name or ep.model_id,
                    status=ep.state.value.lower(),
                    provider=self.provider,
                    model_type=ModelType.FULL,
                    created_at=ep.created_at,
                )
            )
        return models

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
        scale_down_policy: str | None = None,
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
            scale_down_policy: Auto-scaling policy (``"NONE"``, ``"TIMER"``,
                ``"INACTIVE"``).
            scale_down_threshold_ms: Idle threshold in milliseconds before scaling down.

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

        payload: dict[str, Any] = {
            "deploymentName": deployment_name,
            "modelName": model_id,
            "deviceConfigs": devices,
            "replicas": replicas,
        }
        if model_access_key:
            payload["modelAccessKey"] = model_access_key
        if context_length is not None:
            payload["contextLength"] = context_length
        if scale_down_policy is not None:
            payload["scaleDownPolicy"] = scale_down_policy
        if scale_down_threshold_ms is not None:
            payload["scaleDownThreshold"] = scale_down_threshold_ms

        logger.info(f"Creating Parasail deployment '{deployment_name}'...")
        response = await self._client.post("/dedicated/deployments", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_endpoint(data)

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Gets details of a dedicated endpoint.

        Args:
            endpoint_id: Numeric deployment ID (as a string).

        Returns:
            :class:`Endpoint` with current status.
        """
        response = await self._client.get(f"/dedicated/deployments/{endpoint_id}")
        response.raise_for_status()
        return self._parse_endpoint(response.json())

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Updates a dedicated endpoint's replica count.

        Parasail's update flow requires fetching the current deployment and
        PUTting the modified object back.  Only replica count is updated via
        the ``autoscaling`` parameter; hardware changes require creating a new
        endpoint.

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
        current = get_response.json()

        if autoscaling is not None:
            current["replicas"] = max(1, autoscaling.min_replicas)

        put_response = await self._client.put(
            f"/dedicated/deployments/{endpoint_id}", json=current
        )
        put_response.raise_for_status()
        return self._parse_endpoint(put_response.json())

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
        """Lists all dedicated endpoints for this account.

        Returns:
            List of :class:`Endpoint` objects.
        """
        response = await self._client.get("/dedicated/deployments")
        response.raise_for_status()
        raw = response.json()
        if isinstance(raw, dict) and "deployments" in raw:
            items = raw["deployments"]
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        return [self._parse_endpoint(item) for item in items]

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """Lists available hardware configurations for Parasail.

        If *model_id* is provided, queries the ``/dedicated/devices`` API to
        return only hardware that is compatible with that model.  Otherwise
        returns a static list of known Parasail GPU types.

        Args:
            model_id: Optional HuggingFace repo ID to filter compatible hardware.

        Returns:
            List of :class:`HardwareConfig` objects.
        """
        if model_id is None:
            return list(_KNOWN_HARDWARE)

        _validate_model_source(model_id)
        try:
            response = await self._client.get(
                "/dedicated/devices",
                params={"engineName": "VLLM", "modelName": model_id, "modelAccessKey": ""},
            )
            response.raise_for_status()
            devices: list[dict] = response.json()
            return [
                HardwareConfig(
                    accelerator=d["device"],
                    count=d["count"],
                )
                for d in devices
                if isinstance(d, dict) and "device" in d and "count" in d
            ]
        except Exception as exc:
            logger.warning(
                f"Could not fetch hardware from Parasail API: {exc}. "
                "Returning static hardware list."
            )
            return list(_KNOWN_HARDWARE)

    # -------------------------------------------------------------------------
    # Start / stop
    # -------------------------------------------------------------------------

    async def start_endpoint(
        self, endpoint_id: str, min_replicas: int = 1
    ) -> Endpoint:
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
    ) -> list[dict]:
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
        devices: list[dict] = response.json()

        matched = False
        for d in devices:
            is_match = (
                d.get("device") == desired_device and d.get("count") == desired_count
            )
            d["selected"] = is_match
            matched = matched or is_match

        if not matched:
            available = [
                f"{d.get('device')} x{d.get('count')}"
                for d in devices
                if isinstance(d, dict)
            ]
            raise ValueError(
                f"No matching hardware config found for "
                f"'{desired_device}' x{desired_count}. "
                f"Available configurations: {available}"
            )

        return devices

    def _parse_endpoint(self, data: dict) -> Endpoint:
        """Converts a Parasail API response dict to an :class:`Endpoint`."""
        deployment_id = str(data.get("id", ""))
        model_name: str = data.get("modelName", "")
        display_name: str | None = data.get("deploymentName")
        external_alias: str | None = data.get("externalAlias")

        status_block = data.get("status", {})
        raw_status = (
            status_block.get("status", "").upper()
            if isinstance(status_block, dict)
            else str(status_block).upper()
        )
        state = _PARASAIL_STATE_MAP.get(raw_status, EndpointState.PENDING)

        endpoint_url = (
            f"{_INFERENCE_BASE_URL}/chat/completions" if external_alias else None
        )

        device_configs: list[dict] = data.get("deviceConfigs", [])
        selected_config = next(
            (d for d in device_configs if d.get("selected")), None
        )
        if selected_config is None and device_configs:
            selected_config = device_configs[0]

        if selected_config:
            hw = HardwareConfig(
                accelerator=selected_config.get("device", "unknown"),
                count=selected_config.get("count", 1),
            )
        else:
            hw = HardwareConfig(accelerator="unknown", count=1)

        replicas: int = data.get("replicas", 1)
        asc = AutoscalingConfig(min_replicas=replicas, max_replicas=replicas)

        created_at: datetime | None = None
        created_str = data.get("createdAt") or data.get("created_at")
        if created_str:
            try:
                created_at = datetime.fromisoformat(
                    created_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass
        if created_at is None:
            created_at = datetime.now(tz=timezone.utc)

        return Endpoint(
            endpoint_id=deployment_id,
            provider=self.provider,
            model_id=model_name,
            endpoint_url=endpoint_url,
            state=state,
            hardware=hw,
            autoscaling=asc,
            created_at=created_at,
            display_name=display_name,
            inference_model_name=external_alias,
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
