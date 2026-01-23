"""Together.ai deployment client."""

import os
import re
from datetime import datetime
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

# Mapping from Together.ai endpoint states to our EndpointState enum
TOGETHER_STATE_MAP = {
    "PENDING": EndpointState.PENDING,
    "STARTING": EndpointState.STARTING,
    "STARTED": EndpointState.RUNNING,
    "RUNNING": EndpointState.RUNNING,
    "STOPPING": EndpointState.STOPPING,
    "STOPPED": EndpointState.STOPPED,
    "ERROR": EndpointState.ERROR,
    "FAILED": EndpointState.ERROR,
}


class TogetherDeploymentClient(BaseDeploymentClient):
    """Together.ai deployment client (async).

    API Reference: https://docs.together.ai/reference

    Authentication: Bearer token via TOGETHER_API_KEY env var or constructor.

    Hardware format: "{count}x_nvidia_{gpu}_sxm" (e.g., "1x_nvidia_a100_80gb_sxm")

    Endpoint states: PENDING, STARTING, STARTED, STOPPING, STOPPED, ERROR
    """

    BASE_URL = "https://api.together.xyz/v1"
    provider = DeploymentProvider.TOGETHER

    def __init__(self, api_key: str | None = None):
        """Initialize the Together.ai deployment client.

        Args:
            api_key: Together API key. If not provided, reads from
                     TOGETHER_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Together API key must be provided or set via TOGETHER_API_KEY env var"
            )
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
            follow_redirects=True,  # Handle 308 redirects from Together.ai
        )

    async def __aenter__(self) -> "TogetherDeploymentClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._client.aclose()

    def _to_together_hardware(self, hw: HardwareConfig) -> str:
        """Convert HardwareConfig to Together.ai hardware string format.

        Args:
            hw: HardwareConfig with accelerator and count

        Returns:
            Together.ai hardware string (e.g., "2x_nvidia_a100_80gb_sxm")
        """
        return f"{hw.count}x_{hw.accelerator}_sxm"

    def _from_together_hardware(self, hw_str: str) -> HardwareConfig:
        """Convert Together.ai hardware string to HardwareConfig.

        Args:
            hw_str: Together.ai hardware string (e.g., "2x_nvidia_a100_80gb_sxm")

        Returns:
            HardwareConfig with accelerator and count
        """
        match = re.match(r"(\d+)x_(.+)_sxm", hw_str)
        if not match:
            # Handle simpler format without _sxm suffix
            match = re.match(r"(\d+)x_(.+)", hw_str)
            if not match:
                return HardwareConfig(accelerator=hw_str, count=1)
        return HardwareConfig(accelerator=match.group(2), count=int(match.group(1)))

    def _parse_endpoint(self, data: dict[str, Any]) -> Endpoint:
        """Parse Together.ai endpoint response into Endpoint dataclass."""
        state_str = data.get("state", "PENDING").upper()
        state = TOGETHER_STATE_MAP.get(state_str, EndpointState.PENDING)

        hardware_str = data.get("hardware", "")
        hardware = self._from_together_hardware(hardware_str)

        # Parse autoscaling from nested object or fall back to top-level fields
        autoscaling_data = data.get("autoscaling", {})
        autoscaling = AutoscalingConfig(
            min_replicas=autoscaling_data.get(
                "min_replicas", data.get("min_replicas", 1)
            ),
            max_replicas=autoscaling_data.get(
                "max_replicas", data.get("max_replicas", 1)
            ),
        )

        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(
                    data["created_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return Endpoint(
            endpoint_id=data.get("id", ""),
            provider=DeploymentProvider.TOGETHER,
            model_id=data.get("model", ""),
            endpoint_url=data.get("endpoint_url"),
            state=state,
            hardware=hardware,
            autoscaling=autoscaling,
            created_at=created_at,
            display_name=data.get("display_name"),
        )

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
    ) -> UploadedModel:
        """Upload a model to Together.ai.

        Args:
            model_source: HuggingFace repo name or S3/GCS URL
            model_name: Display name for the model
            model_type: Type of model (FULL or ADAPTER)
            base_model: Base model for LoRA adapters

        Returns:
            UploadedModel with provider-specific model ID
        """
        import logging

        logger = logging.getLogger(__name__)

        payload: dict[str, Any] = {
            "model_source": model_source,
            "model_name": model_name,
        }

        if model_type == ModelType.ADAPTER and base_model:
            payload["base_model"] = base_model
            payload["model_type"] = "lora"

        # Log the payload being sent to Together.ai
        logger.info(f"Sending upload request to Together.ai with payload: {payload}")

        response = await self._client.post("/models", json=payload)

        # Log error details if upload fails
        if response.status_code >= 400:
            logger.error(
                f"Together.ai model upload failed with status {response.status_code}. "
                f"Request payload: {payload}. "
                f"Response: {response.text}"
            )

            # Check if this is an "upload already in progress" error
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", "")
                    if "upload already in progress" in error_message.lower():
                        logger.warning(
                            f"Model upload already in progress for {model_name}. "
                            f"Attempting to find existing upload..."
                        )
                        # Try to find the existing model by name
                        existing_models = await self.list_models()
                        for model in existing_models:
                            if (
                                model.model_name == model_name
                                or model.model_id == model_name
                            ):
                                logger.info(
                                    f"Found existing model: {model.model_id} "
                                    f"with status: {model.status}"
                                )
                                # Return the existing model information
                                # Note: job_id might not be available for existing models
                                return UploadedModel(
                                    provider_model_id=model.model_id,
                                    job_id=None,
                                    status=model.status,
                                    request_payload=payload,
                                )

                        # Model not found yet - it may still be initializing
                        # Construct expected model ID with username prefix
                        expected_model_id = model_name
                        try:
                            models_response = await self._client.get("/models")
                            if models_response.status_code == 200:
                                models_data = models_response.json()
                                username = None
                                # Look for our own deployment models
                                for model in models_data:
                                    existing_id = model.get("id", "")
                                    if (
                                        "/" in existing_id
                                        and "oumi-deployment-" in existing_id
                                    ):
                                        username = existing_id.split("/")[0]
                                        break
                                # Fallback to organization field
                                if not username:
                                    for model in models_data:
                                        org = model.get("organization", "")
                                        if org and org not in (
                                            "together",
                                            "meta",
                                            "mistral",
                                        ):
                                            username = org
                                            break
                                if username:
                                    expected_model_id = f"{username}/{model_name}"
                        except Exception:
                            pass

                        logger.info(
                            f"Could not find existing model yet. "
                            f"Returning expected model ID: {expected_model_id}"
                        )
                        return UploadedModel(
                            provider_model_id=expected_model_id,
                            job_id=None,
                            status="pending",
                            request_payload=payload,
                        )
                except Exception as e:
                    if "upload already in progress" in str(e).lower():
                        raise
                    # If parsing failed, continue to raise the original HTTP error
                    logger.warning(f"Failed to parse 400 error response: {e}")

        response.raise_for_status()
        data = response.json()

        # Together.ai may wrap the response in a 'data' field
        # Check if response has a 'data' key and use that
        if "data" in data and isinstance(data["data"], dict):
            response_data = data["data"]
        else:
            response_data = data

        # Together.ai may return the model ID in different fields
        # Try: model_id, id, name
        provider_model_id = (
            response_data.get("model_id")
            or response_data.get("id")
            or response_data.get("name")
            or ""
        )

        # Together.ai stores models with username prefix (e.g., "username/model-name")
        # If the response doesn't include the full model ID, construct it from the model_name
        if not provider_model_id or "/" not in provider_model_id:
            # Get username by finding our own uploaded models (matching our naming pattern)
            try:
                models_response = await self._client.get("/models")
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    # Look for models matching our deployment naming pattern
                    username = None
                    for model in models_data:
                        model_id = model.get("id", "")
                        # Look for models with our naming pattern: */oumi-deployment-*
                        if "/" in model_id and "oumi-deployment-" in model_id:
                            # Extract username from our own model
                            username = model_id.split("/")[0]
                            logger.info(
                                f"Found username from existing deployment: {username}"
                            )
                            break

                    # Fallback: use organization field from model metadata
                    if not username:
                        for model in models_data:
                            org = model.get("organization", "")
                            if org and org not in ("together", "meta", "mistral"):
                                username = org
                                logger.info(
                                    f"Using username from organization field: {username}"
                                )
                                break

                    if username:
                        provider_model_id = f"{username}/{model_name}"
                        logger.info(f"Constructed full model ID: {provider_model_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to extract username for model ID construction: {e}"
                )

        if not provider_model_id:
            # Log the full response for debugging
            logger.error(
                f"Together.ai upload response missing model ID. "
                f"Full response: {data}, "
                f"Response data: {response_data}"
            )
            raise ValueError(
                f"Together.ai did not return a model ID in the upload response. "
                f"Response keys: {list(data.keys())}, "
                f"Data keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'N/A'}"
            )

        return UploadedModel(
            provider_model_id=provider_model_id,
            job_id=response_data.get("job_id"),
            status=response_data.get("status", "pending"),
            request_payload=payload,
        )

    async def get_job_status(self, job_id: str) -> dict:
        """Get the status of a model upload job.

        For custom uploaded models, Together.ai uses a job-based status system.
        This checks the job status, not the model status directly.

        Args:
            job_id: Together.ai job ID from the upload response

        Returns:
            Dictionary with job details including 'status' field and potentially 'error'

        Raises:
            ValueError: If job_id is empty or None
            HTTPError: If the API request fails
        """
        if not job_id:
            raise ValueError(
                "Cannot get job status: job_id is empty. "
                "This usually means the model upload did not return a valid job ID."
            )

        response = await self._client.get(f"/jobs/{job_id}")

        # Log errors for debugging
        if response.status_code >= 400:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Together.ai job status check failed with status {response.status_code}. "
                f"Job ID: {job_id}. "
                f"Response: {response.text}"
            )

        response.raise_for_status()
        data = response.json()

        # Log the full response for debugging if there's an error
        if data.get("status", "").lower() in ["failed", "error"] or "error" in data:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Together.ai job {job_id} has error status. Full response: {data}"
            )

        return data

    async def get_model_status(self, model_id: str) -> str:
        """Get the status of an uploaded model.

        Note: For custom uploaded models, use get_job_status() instead.
        This method is for checking pre-existing models in the Together.ai catalog.

        Args:
            model_id: Together.ai model ID

        Returns:
            Status string (e.g., "ready", "pending", "failed")

        Raises:
            ValueError: If model_id is empty or None
        """
        if not model_id:
            raise ValueError(
                "Cannot get model status: model_id is empty. "
                "This usually means the model upload did not return a valid model ID."
            )

        response = await self._client.get(f"/models/{model_id}")
        response.raise_for_status()
        data = response.json()
        return data.get("status", "unknown")

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
    ) -> Endpoint:
        """Create an inference endpoint for a model.

        Args:
            model_id: Together.ai model ID
            hardware: Hardware configuration
            autoscaling: Autoscaling configuration
            display_name: Optional display name

        Returns:
            Created Endpoint
        """
        payload: dict[str, Any] = {
            "model": model_id,
            "hardware": self._to_together_hardware(hardware),
            "autoscaling": {
                "min_replicas": autoscaling.min_replicas,
                "max_replicas": autoscaling.max_replicas,
            },
        }

        if display_name:
            payload["display_name"] = display_name

        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Creating endpoint with payload: {payload}")

        response = await self._client.post("/endpoints", json=payload)

        # Log the actual request details for debugging
        logger.info(
            f"Request URL: {response.request.url}, "
            f"Method: {response.request.method}, "
            f"Status: {response.status_code}"
        )
        logger.info(f"Request headers: {dict(response.request.headers)}")

        if response.status_code >= 400:
            logger.error(
                f"Endpoint creation failed. Status: {response.status_code}, "
                f"Response: {response.text}"
            )

        response.raise_for_status()
        data = response.json()

        return self._parse_endpoint(data)

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Get details of an endpoint.

        Args:
            endpoint_id: Together.ai endpoint ID

        Returns:
            Endpoint details
        """
        response = await self._client.get(f"/endpoints/{endpoint_id}")
        response.raise_for_status()
        data = response.json()

        return self._parse_endpoint(data)

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Update an endpoint's configuration.

        Args:
            endpoint_id: Together.ai endpoint ID
            autoscaling: New autoscaling configuration
            hardware: New hardware configuration

        Returns:
            Updated Endpoint
        """
        payload: dict[str, Any] = {}

        if autoscaling:
            payload["autoscaling"] = {
                "min_replicas": autoscaling.min_replicas,
                "max_replicas": autoscaling.max_replicas,
            }

        if hardware:
            payload["hardware"] = self._to_together_hardware(hardware)

        response = await self._client.patch(f"/endpoints/{endpoint_id}", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_endpoint(data)

    async def delete_endpoint(self, endpoint_id: str) -> None:
        """Delete an endpoint.

        Args:
            endpoint_id: Together.ai endpoint ID
        """
        response = await self._client.delete(f"/endpoints/{endpoint_id}")
        response.raise_for_status()

    async def list_endpoints(self) -> list[Endpoint]:
        """List all endpoints owned by this account.

        Returns:
            List of Endpoints
        """
        response = await self._client.get("/endpoints", params={"mine": "true"})
        response.raise_for_status()
        data = response.json()

        endpoints = []
        items = data if isinstance(data, list) else data.get("endpoints", [])
        for item in items:
            endpoints.append(self._parse_endpoint(item))

        return endpoints

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """List available hardware configurations.

        Args:
            model_id: Optional model ID to filter compatible hardware

        Returns:
            List of available HardwareConfigs
        """
        params = {}
        if model_id:
            params["model"] = model_id

        response = await self._client.get("/hardware", params=params)
        response.raise_for_status()
        data = response.json()

        hardware_list = []
        items = data if isinstance(data, list) else data.get("hardware", [])
        for item in items:
            if isinstance(item, str):
                hardware_list.append(self._from_together_hardware(item))
            elif isinstance(item, dict):
                hw_str = item.get("name", item.get("id", ""))
                hardware_list.append(self._from_together_hardware(hw_str))

        return hardware_list

    async def list_models(self, include_public: bool = False) -> list[Model]:
        """List models uploaded to Together.ai.

        Args:
            include_public: If True, include public/platform models. If False (default),
                           only return user-uploaded custom models (upload + fine-tuning jobs).

        Returns:
            List of Model objects with status information
        """
        models = []

        # Fetch model upload jobs (job-* IDs) - these are direct model uploads
        try:
            response = await self._client.get("/jobs")
            response.raise_for_status()
            data = response.json()

            # Parse model upload jobs
            items = data if isinstance(data, list) else data.get("data", [])

            # If items is a dict (job_id -> job_data mapping), convert to list
            if isinstance(items, dict):
                items = [{"id": k, **v} for k, v in items.items()]

            for item in items:
                # Job fields - try multiple possible field names
                job_id = (
                    item.get("id")
                    or item.get("job_id")
                    or item.get("jobId")
                    or item.get("_id")
                    or ""
                )

                # If still no ID found, try to extract from modelName if it contains the job ID
                if not job_id:
                    model_name_value = item.get("modelName", "")
                    if "/" in model_name_value:
                        # Sometimes the full path includes the job ID
                        parts = model_name_value.split("/")
                        for part in parts:
                            if part.startswith("job-"):
                                job_id = part
                                break

                # For upload jobs, the model name might be in different fields
                # Prioritize modelName (camelCase) as that's the actual field from Together.ai
                model_name = (
                    item.get("modelName")
                    or item.get("model_name")
                    or item.get("output_name")
                    or item.get("outputName")
                    or item.get("output_model")
                    or item.get("outputModel")
                    or item.get("model")
                    or item.get("name")
                    or job_id
                    or ""
                )

                status = (item.get("status") or item.get("state") or "unknown").lower()

                # Parse created_at - try multiple field names and formats
                created_at = None
                for field in [
                    "created_at",
                    "createdAt",
                    "created",
                    "timestamp",
                    "create_time",
                ]:
                    if item.get(field):
                        try:
                            # Try as unix timestamp
                            created_at = datetime.fromtimestamp(item[field])
                            break
                        except (ValueError, TypeError):
                            try:
                                # Try as ISO string
                                created_at = datetime.fromisoformat(
                                    item[field].replace("Z", "+00:00")
                                )
                                break
                            except (ValueError, TypeError, AttributeError):
                                pass

                # Determine model type
                model_type = ModelType.FULL
                base_model = None

                models.append(
                    Model(
                        model_id=job_id,
                        model_name=model_name,
                        status=status,
                        provider=DeploymentProvider.TOGETHER,
                        model_type=model_type,
                        created_at=created_at,
                        base_model=base_model,
                    )
                )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to fetch model upload jobs: {e}")

        # Also fetch fine-tuning jobs (ft-* IDs)
        try:
            response = await self._client.get("/fine-tunes")
            response.raise_for_status()
            data = response.json()

            # Parse fine-tuning jobs
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                # Fine-tuning job fields - try multiple possible field names
                job_id = (
                    item.get("id")
                    or item.get("job_id")
                    or item.get("jobId")
                    or item.get("fine_tune_id")
                    or ""
                )

                model_name = (
                    item.get("model_output_name")
                    or item.get("modelOutputName")
                    or item.get("output_name")
                    or item.get("outputName")
                    or item.get("fine_tuned_model")
                    or item.get("model_name")
                    or item.get("modelName")
                    or item.get("name")
                    or job_id
                    or ""
                )

                status = (item.get("status") or item.get("state") or "unknown").lower()

                # Parse created_at - try multiple field names and formats
                created_at = None
                for field in [
                    "created_at",
                    "createdAt",
                    "created",
                    "timestamp",
                    "create_time",
                ]:
                    if item.get(field):
                        try:
                            # Try as unix timestamp
                            created_at = datetime.fromtimestamp(item[field])
                            break
                        except (ValueError, TypeError):
                            try:
                                # Try as ISO string
                                created_at = datetime.fromisoformat(
                                    item[field].replace("Z", "+00:00")
                                )
                                break
                            except (ValueError, TypeError, AttributeError):
                                pass

                # Determine model type
                model_type = ModelType.FULL
                base_model = item.get("model")  # The base model used for fine-tuning

                models.append(
                    Model(
                        model_id=job_id,
                        model_name=model_name,
                        status=status,
                        provider=DeploymentProvider.TOGETHER,
                        model_type=model_type,
                        created_at=created_at,
                        base_model=base_model,
                    )
                )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to fetch fine-tuning jobs: {e}")

        # If include_public is True, also fetch public models from the catalog
        if include_public:
            try:
                response = await self._client.get("/models")
                response.raise_for_status()
                data = response.json()

                items = data if isinstance(data, list) else data.get("models", [])
                for item in items:
                    # Parse model information from Together.ai catalog
                    model_id = item.get("id", item.get("name", ""))
                    model_name = item.get("display_name", item.get("name", model_id))
                    # Catalog models don't have upload status - they're just "available"
                    status = "ready"

                    # Determine model type
                    model_type = None
                    if item.get("type") == "lora":
                        model_type = ModelType.ADAPTER
                    else:
                        model_type = ModelType.FULL

                    # Parse created timestamp
                    created_at = None
                    if item.get("created"):
                        try:
                            created_at = datetime.fromtimestamp(item["created"])
                        except (ValueError, TypeError):
                            pass

                    models.append(
                        Model(
                            model_id=model_id,
                            model_name=model_name,
                            status=status,
                            provider=DeploymentProvider.TOGETHER,
                            model_type=model_type,
                            created_at=created_at,
                            base_model=None,
                        )
                    )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to fetch catalog models: {e}")

        return models

    async def delete_model(self, model_id: str) -> None:
        """Delete a model on Together.ai.

        Args:
            model_id: Together.ai model ID

        Raises:
            NotImplementedError: Together.ai does not currently support model deletion
        """
        raise NotImplementedError(
            "Together.ai does not currently support model deletion via their API. "
            "Uploaded models can only be deleted through the Together.ai dashboard at "
            "https://api.together.xyz/playground. "
            "However, deleting the endpoint will stop any associated inference costs."
        )
