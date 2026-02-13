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

"""Base types and interfaces for deployment clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import httpx


class DeploymentProvider(str, Enum):
    """Supported deployment providers."""

    FIREWORKS = "fireworks"


class EndpointState(str, Enum):
    """State of a deployed endpoint."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ModelType(str, Enum):
    """Type of model being deployed."""

    FULL = "full"
    ADAPTER = "adapter"


@dataclass
class HardwareConfig:
    """Hardware configuration for deployment."""

    accelerator: str
    count: int = 1


@dataclass
class AutoscalingConfig:
    """Autoscaling configuration for deployment."""

    min_replicas: int = 1
    max_replicas: int = 1


@dataclass
class UploadedModel:
    """Result of uploading a model to a provider."""

    provider_model_id: str
    job_id: str | None = None
    status: str = "pending"
    request_payload: dict | None = None


@dataclass
class Model:
    """Information about an uploaded model."""

    model_id: str
    model_name: str
    status: str
    provider: DeploymentProvider
    model_type: ModelType | None = None
    created_at: datetime | None = None
    base_model: str | None = None
    organization: str | None = None


@dataclass
class Endpoint:
    """A deployed model endpoint."""

    endpoint_id: str
    provider: DeploymentProvider
    model_id: str
    endpoint_url: str | None
    state: EndpointState
    hardware: HardwareConfig
    autoscaling: AutoscalingConfig
    created_at: datetime | None = None
    display_name: str | None = None
    inference_model_name: str | None = None  # Model name to use for inference calls


class BaseDeploymentClient(ABC):
    """Abstract base class for deployment clients."""

    provider: DeploymentProvider

    async def __aenter__(self) -> "BaseDeploymentClient":
        """Enters the async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the async context manager."""

    @abstractmethod
    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        progress_callback: Any | None = None,
    ) -> UploadedModel:
        """Uploads a model to the provider.

        Args:
            model_source: Path to model (S3/GCS URL or local path)
            model_name: Display name for the model
            model_type: Type of model (FULL or ADAPTER)
            base_model: Base model for LoRA adapters
            progress_callback: Optional async callback for progress updates.
                Signature: async def callback(stage: str, message: str, details: dict)

        Returns:
            UploadedModel with provider-specific model ID
        """
        pass

    @abstractmethod
    async def get_model_status(self, model_id: str) -> str:
        """Gets the status of an uploaded model.

        Args:
            model_id: Provider-specific model ID

        Returns:
            Status string (e.g., "ready", "pending", "failed")
        """
        pass

    @abstractmethod
    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
    ) -> Endpoint:
        """Creates an inference endpoint for a model.

        Args:
            model_id: Provider-specific model ID
            hardware: Hardware configuration
            autoscaling: Autoscaling configuration
            display_name: Optional display name

        Returns:
            Created Endpoint
        """
        pass

    @abstractmethod
    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Gets details of an endpoint.

        Args:
            endpoint_id: Provider-specific endpoint ID

        Returns:
            Endpoint details
        """
        pass

    @abstractmethod
    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Updates an endpoint's configuration.

        Args:
            endpoint_id: Provider-specific endpoint ID
            autoscaling: New autoscaling configuration
            hardware: New hardware configuration

        Returns:
            Updated Endpoint
        """
        pass

    @abstractmethod
    async def delete_endpoint(self, endpoint_id: str, *, force: bool = False) -> None:
        """Deletes an endpoint.

        Args:
            endpoint_id: Provider-specific endpoint ID
            force: If True, skip provider safety checks and perform a hard deletion
                (e.g. delete even if the deployment recently received requests).
        """
        pass

    @abstractmethod
    async def list_endpoints(self) -> list[Endpoint]:
        """Lists all endpoints owned by this account.

        Returns:
            List of Endpoints
        """
        pass

    @abstractmethod
    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """Lists available hardware configurations.

        Args:
            model_id: Optional model ID to filter compatible hardware

        Returns:
            List of available HardwareConfigs
        """
        pass

    @abstractmethod
    async def list_models(
        self, include_public: bool = False, organization: str | None = None
    ) -> list["Model"]:
        """Lists models uploaded to this provider.

        Args:
            include_public: If True, include public/platform models. If False (default),
                           only return user-uploaded custom models.
            organization: If provided, filter results to only models belonging to this
                         organization (provider-specific).

        Returns:
            List of Model objects with status information
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Deletes an uploaded model.

        Args:
            model_id: Provider-specific model ID

        Raises:
            NotImplementedError: If provider doesn't support model deletion
            httpx.HTTPError: If deletion fails
        """
        pass

    # --- Optional lifecycle & testing (override in providers that support them) ---

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Gets the status of an asynchronous upload/processing job.

        Override in providers that use job-based workflows.

        Args:
            job_id: Provider-specific job identifier.

        Returns:
            Dict with at least a ``"status"`` key.

        Raises:
            NotImplementedError: If provider does not use job-based
                status tracking.
        """
        raise NotImplementedError(
            f"{self.provider.value} does not support job-based status tracking"
        )

    def _get_inference_auth_headers(self) -> dict[str, str]:
        """Returns headers required for inference requests.

        Override in subclasses to add Authorization or other headers.
        Default: no auth (for public or provider-specific handling).
        """
        return {}

    async def start_endpoint(
        self, endpoint_id: str, min_replicas: int = 1
    ) -> "Endpoint":
        """Starts a stopped endpoint (scales up from 0 replicas).

        Args:
            endpoint_id: Provider-specific endpoint ID
            min_replicas: Minimum replicas when started

        Returns:
            Updated Endpoint

        Raises:
            NotImplementedError: If provider does not support start/stop
        """
        raise NotImplementedError(
            f"{self.provider.value} does not support endpoint start/stop operations"
        )

    async def stop_endpoint(self, endpoint_id: str) -> "Endpoint":
        """Stops an endpoint by scaling to 0 replicas (cost savings).

        Args:
            endpoint_id: Provider-specific endpoint ID

        Returns:
            Updated Endpoint

        Raises:
            NotImplementedError: If provider does not support start/stop
        """
        raise NotImplementedError(
            f"{self.provider.value} does not support endpoint start/stop operations"
        )

    async def test_endpoint(
        self,
        endpoint_url: str,
        prompt: str,
        model_id: str | None = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Sends a test prompt to a deployed endpoint.

        Uses the OpenAI-compatible chat completions format.

        Args:
            endpoint_url: Full URL for chat completions
                (e.g. ``.../v1/chat/completions``).
            prompt: User message to send.
            model_id: Optional model name for the request body
                (some endpoints require it).
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, request is sent with stream=True
                (caller must handle NDJSON).

        Returns:
            Response JSON (e.g. ``{"choices": [...]}``).
        """
        payload: dict[str, Any] = {
            "model": model_id or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        headers = {
            "Content-Type": "application/json",
            **self._get_inference_auth_headers(),
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            if stream:
                return {"stream": True, "content": response.text}
            return response.json()
