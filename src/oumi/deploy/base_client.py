"""Base types and interfaces for deployment clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DeploymentProvider(str, Enum):
    """Supported deployment providers."""

    TOGETHER = "together"
    FIREWORKS = "fireworks"
    MODAL = "modal"


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


class BaseDeploymentClient(ABC):
    """Abstract base class for deployment clients."""

    provider: DeploymentProvider

    @abstractmethod
    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
    ) -> UploadedModel:
        """Upload a model to the provider.

        Args:
            model_source: Path to model (S3/GCS URL or local path)
            model_name: Display name for the model
            model_type: Type of model (FULL or ADAPTER)
            base_model: Base model for LoRA adapters

        Returns:
            UploadedModel with provider-specific model ID
        """
        pass

    @abstractmethod
    async def get_model_status(self, model_id: str) -> str:
        """Get the status of an uploaded model.

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
        """Create an inference endpoint for a model.

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
        """Get details of an endpoint.

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
        """Update an endpoint's configuration.

        Args:
            endpoint_id: Provider-specific endpoint ID
            autoscaling: New autoscaling configuration
            hardware: New hardware configuration

        Returns:
            Updated Endpoint
        """
        pass

    @abstractmethod
    async def delete_endpoint(self, endpoint_id: str) -> None:
        """Delete an endpoint.

        Args:
            endpoint_id: Provider-specific endpoint ID
        """
        pass

    @abstractmethod
    async def list_endpoints(self) -> list[Endpoint]:
        """List all endpoints owned by this account.

        Returns:
            List of Endpoints
        """
        pass

    @abstractmethod
    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """List available hardware configurations.

        Args:
            model_id: Optional model ID to filter compatible hardware

        Returns:
            List of available HardwareConfigs
        """
        pass

    @abstractmethod
    async def list_models(self, include_public: bool = False) -> list["Model"]:
        """List models uploaded to this provider.

        Args:
            include_public: If True, include public/platform models. If False (default),
                           only return user-uploaded custom models.

        Returns:
            List of Model objects with status information
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete an uploaded model.

        Args:
            model_id: Provider-specific model ID

        Raises:
            NotImplementedError: If provider doesn't support model deletion
            httpx.HTTPError: If deletion fails
        """
        pass
