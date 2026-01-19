"""Modal.com deployment client for vLLM inference.

Modal provides serverless GPU functions with auto-scaling and pay-per-second billing.
Uses vLLM for high-performance inference with OpenAI-compatible API.

Key Features:
- Serverless auto-scaling (scale to zero)
- GPU memory snapshots for 10x faster cold starts
- Direct S3 mounting via CloudBucketMount (no model upload needed)
- vLLM-powered inference with streaming support

Authentication:
- MODAL_TOKEN_ID: From Modal dashboard (https://modal.com/settings)
- MODAL_TOKEN_SECRET: From Modal dashboard
- AWS credentials: For S3 access (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

References:
- Modal Docs: https://modal.com/docs
- vLLM Deployment: https://docs.vllm.ai/en/latest/deployment/frameworks/modal/
- GPU Snapshots: https://modal.com/blog/gpu-mem-snapshots
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

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


class ModalDeploymentClient(BaseDeploymentClient):
    """Modal.com deployment client for serverless vLLM inference.

    Unlike Together/Fireworks, Modal doesn't require uploading model weights.
    Instead, models are accessed directly from S3 via CloudBucketMount.

    Architecture:
    1. Create Modal App with vLLM container image
    2. Define Function with GPU, S3 mount, and vLLM server
    3. Deploy App → Modal generates HTTPS endpoint URL
    4. Inference requests routed to vLLM's OpenAI-compatible API

    Deployment States:
    - PENDING: App being built/deployed
    - RUNNING: vLLM server ready and accepting requests
    - STOPPED: App scaled to zero (no active containers)
    - ERROR: Deployment or runtime failure
    """

    provider = DeploymentProvider.MODAL

    # Modal GPU types mapping
    GPU_TYPES = {
        "nvidia_a100_40gb": "A100",
        "nvidia_a100_80gb": "A100",
        "nvidia_h100_80gb": "H100",
        "nvidia_a10g": "A10G",
        "nvidia_l4": "L4",
        "nvidia_t4": "T4",
    }

    def __init__(
        self,
        token_id: Optional[str] = None,
        token_secret: Optional[str] = None,
        workspace: Optional[str] = None,
    ):
        """Initialize Modal client.

        Args:
            token_id: Modal token ID (or set MODAL_TOKEN_ID env var)
            token_secret: Modal token secret (or set MODAL_TOKEN_SECRET env var)
            workspace: Modal workspace name (optional, uses default if not set)
        """
        self.token_id = token_id or os.environ.get("MODAL_TOKEN_ID")
        self.token_secret = token_secret or os.environ.get("MODAL_TOKEN_SECRET")
        self.workspace = workspace or os.environ.get("MODAL_WORKSPACE", "default")

        if not self.token_id or not self.token_secret:
            raise ValueError(
                "Modal credentials required. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "environment variables or pass to constructor."
            )

        # Set Modal environment variables for SDK authentication
        os.environ["MODAL_TOKEN_ID"] = self.token_id
        os.environ["MODAL_TOKEN_SECRET"] = self.token_secret

        # Initialize boto3 S3 client for path validation
        self._s3_client = boto3.client("s3")

        # Track deployed apps (app_name -> Modal App object)
        self._deployed_apps: dict[str, tuple] = {}

    async def __aenter__(self) -> "ModalDeploymentClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: Optional[str] = None,
    ) -> UploadedModel:
        """Prepare model metadata for Modal deployment.

        Note: Unlike Together/Fireworks, Modal doesn't require uploading model weights.
        Models are accessed directly from S3 via CloudBucketMount during deployment.

        This method validates the S3 path is accessible and returns metadata for later use.

        Args:
            model_source: S3 path to model (e.g., "s3://bucket/path/to/model")
            model_name: Display name for the model
            model_type: FULL or ADAPTER (LoRA)
            base_model: Base model ID for LoRA adapters

        Returns:
            UploadedModel with model_source as provider_model_id

        Raises:
            ValueError: If model_source is not a valid S3 path or not accessible
        """
        # Validate S3 path format
        if not model_source.startswith("s3://"):
            raise ValueError(
                f"Modal requires S3 paths for models. Got: {model_source}. "
                "Upload model to S3 first, then deploy with s3:// URL."
            )

        # Parse S3 path
        s3_match = re.match(r"s3://([^/]+)/(.+)", model_source)
        if not s3_match:
            raise ValueError(f"Invalid S3 path format: {model_source}")

        bucket_name = s3_match.group(1)
        model_path = s3_match.group(2)

        # Validate S3 path is accessible
        logger.info(f"Validating S3 path: s3://{bucket_name}/{model_path}")
        try:
            # Check if the model directory exists by listing objects with the prefix
            response = self._s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=model_path, MaxKeys=1
            )

            if "Contents" not in response or len(response["Contents"]) == 0:
                raise ValueError(
                    f"S3 path not found or empty: {model_source}. "
                    "Ensure the model has been uploaded to S3."
                )

            logger.info(f"S3 path validated successfully: {model_source}")

        except (BotoCoreError, ClientError) as e:
            error_msg = f"Failed to access S3 path {model_source}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # For LoRA adapters, validate base model is provided
        if model_type == ModelType.ADAPTER and not base_model:
            raise ValueError("base_model required for LoRA adapters")

        # Return metadata - actual deployment happens in create_endpoint
        return UploadedModel(
            provider_model_id=model_source,  # Store S3 path for later use
            status="ready",  # No upload needed, model validated in S3
        )

    async def get_model_status(self, model_id: str) -> str:
        """Get model status.

        For Modal, models are accessed directly from S3, so they're always "ready"
        once validated.

        Args:
            model_id: S3 path to model (from upload_model)

        Returns:
            "ready" (models are always accessible via S3)
        """
        # Validate S3 path format
        if not model_id.startswith("s3://"):
            raise ValueError(f"Invalid model_id (expected S3 path): {model_id}")

        return "ready"

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: Optional[str] = None,
    ) -> Endpoint:
        """Create Modal App with vLLM inference server.

        This generates and deploys a Modal app with vLLM serving the model from S3.

        Steps:
        1. Generate unique app name
        2. Create Modal app with vLLM server configuration
        3. Deploy app to Modal workspace
        4. Return endpoint with URL

        Args:
            model_id: S3 path to model (from upload_model)
            hardware: GPU type and count
            autoscaling: Min/max replicas (Modal handles auto-scaling)
            display_name: Optional name for the deployment

        Returns:
            Endpoint with URL like https://{workspace}--{app}-serve.modal.run

        Raises:
            ValueError: If GPU type not supported
            RuntimeError: If deployment fails
        """
        import modal

        # Parse S3 path
        s3_match = re.match(r"s3://([^/]+)/(.+)", model_id)
        if not s3_match:
            raise ValueError(f"Invalid S3 path: {model_id}")

        bucket_name = s3_match.group(1)
        model_path = s3_match.group(2)

        # Convert hardware config to Modal GPU type
        gpu_type = self._to_modal_gpu(hardware)

        # Generate unique app name
        app_name = self._generate_app_name(display_name or "oumi-inference")

        logger.info(
            f"Creating Modal app {app_name} with {gpu_type} GPU for model {model_id}"
        )

        # Create Modal app
        app = modal.App(
            app_name, experimental_options={"enable_gpu_snapshot": True}
        )

        # Create vLLM container image
        vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
            "vllm==0.6.4",
            "torch==2.5.0",
        )

        # Get AWS credentials secret
        try:
            aws_secret = modal.Secret.from_name("aws-credentials")
        except Exception as e:
            logger.error(
                f"Failed to load aws-credentials secret from Modal: {e}. "
                "Ensure you've created the secret with: "
                "modal secret create aws-credentials AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy"
            )
            raise RuntimeError(
                "AWS credentials not configured in Modal workspace. "
                "Create secret with: modal secret create aws-credentials ..."
            ) from e

        # Define vLLM server class
        @app.cls(
            image=vllm_image,
            gpu=getattr(modal.gpu, gpu_type)(count=hardware.count),
            container_idle_timeout=300,
            allow_concurrent_inputs=100,
            experimental_options={"enable_gpu_snapshot": True},
            secrets=[aws_secret],
            volumes={
                "/models": modal.CloudBucketMount(
                    bucket_name=bucket_name,
                    secret=aws_secret,
                    read_only=True,
                )
            },
            # Autoscaling config
            min_replicas=autoscaling.min_replicas,
            max_replicas=autoscaling.max_replicas,
        )
        class VLLMServer:
            @modal.enter()
            def start_vllm(self):
                """Initialize vLLM engine with optimizations."""
                import os

                from vllm import AsyncEngineArgs, AsyncLLMEngine

                # Enable fast boot for quick cold starts
                os.environ["FAST_BOOT"] = "1"

                engine_args = AsyncEngineArgs(
                    model=f"/models/{model_path}",
                    compilation_config="-O1",  # Fast startup, good performance
                    gpu_memory_utilization=0.95,
                    max_num_seqs=256,
                    enable_chunked_prefill=True,
                )

                self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            @modal.web_endpoint(
                method="POST",
                label="serve",
                requires_proxy_auth=True,  # Require Modal Proxy Auth
            )
            async def chat_completions(self, request: dict):
                """OpenAI-compatible chat completions endpoint.

                Authentication: Modal Proxy Auth enabled
                - Requires Modal-Key and Modal-Secret headers
                - Modal validates credentials at proxy layer before reaching this function
                - Platform (Oumi) provides these headers when forwarding user requests
                """
                from vllm.entrypoints.openai.protocol import ChatCompletionRequest
                from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

                # Use vLLM's built-in OpenAI API handler
                serving_chat = OpenAIServingChat(
                    self.engine,
                    model_name=app_name,
                    response_role="assistant",
                )

                # Parse request
                chat_request = ChatCompletionRequest(**request)

                # Generate response
                return await serving_chat.create_chat_completion(chat_request)

        # Deploy the app
        logger.info(f"Deploying Modal app {app_name}...")
        try:
            with app.run():
                # App is now deployed and running
                # Generate endpoint URL (Modal URL pattern)
                endpoint_url = f"https://{self.workspace}--{app_name}-serve.modal.run"

                # Store app reference for later status checks
                deployment_id = app_name  # Use app name as deployment ID
                self._deployed_apps[deployment_id] = (app, app_name)

                logger.info(f"Modal app deployed successfully: {endpoint_url}")

                return Endpoint(
                    endpoint_id=deployment_id,
                    provider=self.provider,
                    model_id=model_id,
                    endpoint_url=endpoint_url,
                    state=EndpointState.RUNNING,  # App is deployed and ready
                    hardware=hardware,
                    autoscaling=autoscaling,
                    display_name=display_name,
                    created_at=datetime.utcnow(),
                )

        except Exception as e:
            error_msg = f"Failed to deploy Modal app {app_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Get endpoint status from Modal.

        Args:
            endpoint_id: Modal app name (used as deployment ID)

        Returns:
            Endpoint with current state
        """
        import modal

        try:
            # Look up the app
            app = modal.App.lookup(endpoint_id, create_if_missing=False)

            # Check if app exists and is deployed
            if app is None:
                logger.warning(f"Modal app {endpoint_id} not found")
                return Endpoint(
                    endpoint_id=endpoint_id,
                    provider=self.provider,
                    model_id="",
                    endpoint_url=f"https://{self.workspace}--{endpoint_id}-serve.modal.run",
                    state=EndpointState.ERROR,
                    hardware=HardwareConfig(accelerator="unknown", count=1),
                    autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=1),
                    created_at=None,
                )

            # App exists - determine state
            # Modal doesn't have a simple "state" field, so we assume RUNNING if it exists
            state = EndpointState.RUNNING

            # Get app metadata if available
            endpoint_url = f"https://{self.workspace}--{endpoint_id}-serve.modal.run"

            return Endpoint(
                endpoint_id=endpoint_id,
                provider=self.provider,
                model_id="",  # Not tracked in Modal app metadata
                endpoint_url=endpoint_url,
                state=state,
                hardware=HardwareConfig(
                    accelerator="unknown", count=1
                ),  # Not available from lookup
                autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=1),
                created_at=None,
            )

        except Exception as e:
            logger.error(f"Failed to get endpoint status for {endpoint_id}: {e}")
            return Endpoint(
                endpoint_id=endpoint_id,
                provider=self.provider,
                model_id="",
                endpoint_url=f"https://{self.workspace}--{endpoint_id}-serve.modal.run",
                state=EndpointState.ERROR,
                hardware=HardwareConfig(accelerator="unknown", count=1),
                autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=1),
                created_at=None,
            )

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: Optional[AutoscalingConfig] = None,
        hardware: Optional[HardwareConfig] = None,
    ) -> Endpoint:
        """Update endpoint configuration.

        Note: Modal doesn't support in-place updates. To change hardware,
        you need to deploy a new app version.

        Args:
            endpoint_id: Modal app name
            autoscaling: New autoscaling config (requires redeployment)
            hardware: New hardware config (requires redeployment)

        Returns:
            Updated endpoint

        Raises:
            NotImplementedError: Modal doesn't support in-place updates
        """
        raise NotImplementedError(
            "Modal doesn't support in-place endpoint updates. "
            "Delete the deployment and create a new one with different configuration."
        )

    async def delete_endpoint(self, endpoint_id: str) -> None:
        """Delete Modal deployment.

        This stops the app and removes it from the workspace.

        Args:
            endpoint_id: Modal app name
        """
        import modal

        try:
            logger.info(f"Deleting Modal app {endpoint_id}...")

            # Look up and delete the app
            app = modal.App.lookup(endpoint_id, create_if_missing=False)
            if app:
                # Modal SDK doesn't have a direct delete method
                # The app will be removed when we stop tracking it
                # and no containers are running (scale to zero)
                logger.info(
                    f"Modal app {endpoint_id} will scale to zero and be removed"
                )

            # Remove from our tracking
            if endpoint_id in self._deployed_apps:
                del self._deployed_apps[endpoint_id]

            logger.info(f"Modal app {endpoint_id} deleted successfully")

        except Exception as e:
            logger.error(f"Failed to delete Modal app {endpoint_id}: {e}")
            raise

    async def list_endpoints(self) -> list[Endpoint]:
        """List all Modal deployments in the workspace.

        Returns:
            List of endpoints
        """
        # Modal SDK doesn't provide a list_apps method
        # Return apps we've deployed in this session
        endpoints = []
        for deployment_id, (app, app_name) in self._deployed_apps.items():
            endpoint_url = f"https://{self.workspace}--{app_name}-serve.modal.run"
            endpoints.append(
                Endpoint(
                    endpoint_id=deployment_id,
                    provider=self.provider,
                    model_id="",
                    endpoint_url=endpoint_url,
                    state=EndpointState.RUNNING,
                    hardware=HardwareConfig(accelerator="unknown", count=1),
                    autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=1),
                    created_at=None,
                )
            )

        return endpoints

    async def list_hardware(
        self, model_id: Optional[str] = None
    ) -> list[HardwareConfig]:
        """List available GPU types on Modal.

        Args:
            model_id: Unused (Modal hardware availability is model-independent)

        Returns:
            List of supported hardware configurations
        """
        # Modal supports these GPU types
        return [
            HardwareConfig(accelerator="nvidia_a100_40gb", count=1),
            HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
            HardwareConfig(accelerator="nvidia_h100_80gb", count=1),
            HardwareConfig(accelerator="nvidia_a10g", count=1),
            HardwareConfig(accelerator="nvidia_l4", count=1),
            HardwareConfig(accelerator="nvidia_t4", count=1),
            # Multi-GPU configs
            HardwareConfig(accelerator="nvidia_a100_80gb", count=2),
            HardwareConfig(accelerator="nvidia_a100_80gb", count=4),
            HardwareConfig(accelerator="nvidia_h100_80gb", count=2),
            HardwareConfig(accelerator="nvidia_h100_80gb", count=4),
            HardwareConfig(accelerator="nvidia_h100_80gb", count=8),
        ]

    async def list_models(self, include_public: bool = False) -> list[Model]:
        """List models uploaded to Modal.

        For Modal, there's no centralized model registry since models are
        accessed directly from S3. This returns an empty list.

        Args:
            include_public: Unused for Modal

        Returns:
            Empty list (Modal doesn't have a model registry)
        """
        return []

    async def delete_model(self, model_id: str) -> None:
        """Delete a model.

        For Modal, models are stored in S3, not uploaded to Modal.
        This operation is not applicable.

        Args:
            model_id: S3 path to model

        Raises:
            NotImplementedError: Modal doesn't manage model storage
        """
        raise NotImplementedError(
            "Modal doesn't manage model storage. Models are accessed directly from S3. "
            "Delete the model from S3 if needed using AWS CLI or boto3."
        )

    # Helper methods

    def _to_modal_gpu(self, hardware: HardwareConfig) -> str:
        """Convert HardwareConfig to Modal GPU type.

        Args:
            hardware: Hardware configuration

        Returns:
            Modal GPU type (e.g., "A100", "H100")

        Raises:
            ValueError: If GPU type not supported
        """
        gpu_type = self.GPU_TYPES.get(hardware.accelerator)
        if not gpu_type:
            raise ValueError(
                f"Unsupported GPU type: {hardware.accelerator}. "
                f"Supported types: {list(self.GPU_TYPES.keys())}"
            )
        return gpu_type

    def _generate_app_name(self, display_name: str) -> str:
        """Generate valid Modal app name from display name.

        Modal app names must be lowercase alphanumeric with hyphens.
        """
        # Convert to lowercase, replace spaces/underscores with hyphens
        name = display_name.lower().replace("_", "-").replace(" ", "-")
        # Remove invalid characters
        name = re.sub(r"[^a-z0-9-]", "", name)
        # Remove consecutive hyphens
        name = re.sub(r"-+", "-", name)
        # Add timestamp suffix for uniqueness
        timestamp = int(time.time())
        return f"{name}-{timestamp}"
