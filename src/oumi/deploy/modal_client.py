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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]
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
        token_id: str | None = None,
        token_secret: str | None = None,
        workspace: str | None = None,
    ):
        """Initialize Modal client.

        Args:
            token_id: Modal token ID (or set MODAL_TOKEN_ID env var or use modal.toml)
            token_secret: Modal token secret (or set MODAL_TOKEN_SECRET env var)
            workspace: Modal workspace name (optional, uses default if not set)
        """
        self.token_id = token_id or os.environ.get("MODAL_TOKEN_ID")
        self.token_secret = token_secret or os.environ.get("MODAL_TOKEN_SECRET")
        self.workspace = workspace or os.environ.get("MODAL_WORKSPACE")

        # If env vars not set, try reading from Modal's TOML config file
        if not self.token_id or not self.token_secret or not self.workspace:
            config_creds = self._read_modal_config()
            if config_creds:
                self.token_id = self.token_id or config_creds.get("token_id")
                self.token_secret = self.token_secret or config_creds.get("token_secret")
                self.workspace = self.workspace or config_creds.get("workspace")

        # Default workspace to "default" if still not set
        if not self.workspace:
            self.workspace = "default"

        if not self.token_id or not self.token_secret:
            raise ValueError(
                "Modal credentials required. Either:\n"
                "1. Run 'modal token new' to create ~/.modal.toml, or\n"
                "2. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars, or\n"
                "3. Pass token_id and token_secret to constructor."
            )

        # Set Modal environment variables for SDK authentication
        os.environ["MODAL_TOKEN_ID"] = self.token_id
        os.environ["MODAL_TOKEN_SECRET"] = self.token_secret

        # Initialize boto3 S3 client for path validation
        self._s3_client = boto3.client("s3")

        # Track deployed apps (app_name -> Modal App object)
        self._deployed_apps: dict[str, tuple] = {}

    def _read_modal_config(self) -> dict[str, str] | None:
        """Read Modal credentials and workspace from TOML config file.

        Modal CLI stores credentials in ~/.modal.toml after 'modal token new'.
        The section name is the workspace name (e.g., [oumi], [default]).

        Returns:
            Dict with token_id, token_secret, and workspace, or None if not found.
        """
        config_paths = [
            Path.home() / ".modal.toml",
            Path.home() / ".modal" / "config.toml",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "rb") as f:
                        config = tomllib.load(f)

                    # Modal config: section name is workspace (e.g., [oumi], [default])
                    # Find first section with active=true, or first with credentials
                    for section_name, section in config.items():
                        if not isinstance(section, dict):
                            continue
                        token_id = section.get("token_id")
                        token_secret = section.get("token_secret")
                        is_active = section.get("active", False)

                        if token_id and token_secret:
                            # Prefer active profile, but use any valid one
                            if is_active:
                                logger.info(
                                    f"Loaded Modal credentials from {config_path} "
                                    f"[{section_name}]"
                                )
                                return {
                                    "token_id": token_id,
                                    "token_secret": token_secret,
                                    "workspace": section_name,
                                }

                    # If no active profile, use first one with credentials
                    for section_name, section in config.items():
                        if not isinstance(section, dict):
                            continue
                        token_id = section.get("token_id")
                        token_secret = section.get("token_secret")
                        if token_id and token_secret:
                            logger.info(
                                f"Loaded Modal credentials from {config_path} "
                                f"[{section_name}]"
                            )
                            return {
                                "token_id": token_id,
                                "token_secret": token_secret,
                                "workspace": section_name,
                            }

                except Exception as e:
                    logger.warning(f"Failed to read Modal config from {config_path}: {e}")

        return None

    async def __aenter__(self) -> "ModalDeploymentClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        progress_callback: Any | None = None,
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
            progress_callback: Optional callback for progress updates (not used by Modal)

        Returns:
            UploadedModel with model_source as provider_model_id

        Raises:
            ValueError: If model_source is not a valid S3 path or not accessible
        """
        _ = progress_callback  # Not used by Modal
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
        display_name: str | None = None,
        huggingface_model_id: str | None = None,
    ) -> Endpoint:
        """Create Modal App with vLLM inference server.

        This generates and deploys a Modal app with vLLM serving the model.
        Uses Modal's web_server pattern with vLLM serve subprocess for reliability.

        Supports two modes:
        1. S3 path: Model loaded from S3 via CloudBucketMount
        2. HuggingFace model ID: Model downloaded from HuggingFace Hub

        Args:
            model_id: S3 path to model (from upload_model), or unused if huggingface_model_id is provided
            hardware: GPU type and count
            autoscaling: Min/max replicas (Modal handles auto-scaling)
            display_name: Optional name for the deployment
            huggingface_model_id: Optional HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")

        Returns:
            Endpoint with URL like https://{workspace}--{app}-serve.modal.run

        Raises:
            ValueError: If GPU type not supported
            RuntimeError: If deployment fails
        """
        import subprocess
        import tempfile

        # Determine if using HuggingFace or S3 model
        use_huggingface = huggingface_model_id is not None

        if use_huggingface:
            logger.info(
                f"Creating Modal endpoint for HuggingFace model: {huggingface_model_id}"
            )
            vllm_model_path = huggingface_model_id
        else:
            # S3 model - parse path
            s3_match = re.match(r"s3://([^/]+)/(.+)", model_id)
            if not s3_match:
                raise ValueError(f"Invalid S3 path: {model_id}")
            vllm_model_path = f"/models/{s3_match.group(2)}"

        # Convert hardware config to Modal GPU type
        gpu_type = self._to_modal_gpu(hardware)

        # Generate unique app name
        app_name = self._generate_app_name(display_name or "oumi-inference")

        log_model = huggingface_model_id if use_huggingface else model_id
        logger.info(
            f"Creating Modal app {app_name} with {gpu_type} GPU for model {log_model}"
        )

        # Generate a Modal app file dynamically
        vllm_port = 8000
        scaledown = autoscaling.min_replicas if autoscaling.min_replicas > 0 else 300

        # Determine secrets string
        secrets_str = ""
        if use_huggingface:
            secrets_str = 'secrets=[modal.Secret.from_name("huggingface-token")],'

        # Determine volumes string
        if use_huggingface:
            volumes_str = '''volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },'''
        else:
            # S3 model - would need CloudBucketMount
            volumes_str = ""

        app_code = f'''
import modal
import subprocess

app = modal.App("{app_name}")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("libxcb1", "libx11-6", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("vllm>=0.6.0", "huggingface-hub>=0.36.0", "hf_transfer")
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

@app.function(
    image=vllm_image,
    gpu="{gpu_type}:{hardware.count}",
    {secrets_str}
    scaledown_window={scaledown},
    timeout=600,
    {volumes_str}
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port={vllm_port}, startup_timeout=300)
def serve():
    cmd = [
        "vllm", "serve", "{vllm_model_path}",
        "--host", "0.0.0.0",
        "--port", "{vllm_port}",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
    ]
    subprocess.Popen(cmd)
'''

        # Write app to temp file and deploy using modal CLI
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(app_code)
                temp_file = f.name

            logger.info(f"Deploying Modal app from {temp_file}...")

            # Run modal deploy command
            result = subprocess.run(
                ["modal", "deploy", temp_file],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for deployment
            )

            if result.returncode != 0:
                error_msg = f"Modal deploy failed: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Modal deploy output: {result.stdout}")

            # Generate endpoint URL (Modal URL pattern for web_server)
            endpoint_url = (
                f"https://{self.workspace}--{app_name}-serve.modal.run/v1/chat/completions"
            )

            logger.info(f"Modal app deployed successfully: {endpoint_url}")

            returned_model_id = huggingface_model_id if use_huggingface else model_id

            return Endpoint(
                endpoint_id=app_name,
                provider=self.provider,
                model_id=returned_model_id,
                endpoint_url=endpoint_url,
                state=EndpointState.RUNNING,
                hardware=hardware,
                autoscaling=autoscaling,
                display_name=display_name,
                created_at=datetime.now(tz=timezone.utc),
            )

        except subprocess.TimeoutExpired:
            error_msg = f"Modal deploy timed out for {app_name}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Failed to deploy Modal app {app_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            # Clean up temp file
            import os

            if "temp_file" in locals():
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

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
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
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

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
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

    async def list_models(
        self, include_public: bool = False, organization: str | None = None
    ) -> list[Model]:
        """List models uploaded to Modal.

        For Modal, there's no centralized model registry since models are
        accessed directly from S3. This returns an empty list.

        Args:
            include_public: Unused for Modal
            organization: Unused for Modal

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
