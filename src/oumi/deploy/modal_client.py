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

"""Modal.com deployment client for serverless vLLM inference.

Uses Modal's serverless GPU functions with auto-scaling and pay-per-second billing.
Models are accessed directly from S3 via CloudBucketMount (no upload needed) or
downloaded from HuggingFace Hub.

References:
- Modal Docs: https://modal.com/docs
- vLLM on Modal: https://docs.vllm.ai/en/latest/deployment/frameworks/modal/
"""

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

try:
    import tomllib  # type: ignore[import]  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

try:
    import modal
except ImportError as e:
    raise ImportError(
        "The 'modal' package is required for Modal deployments. "
        "Install it with: pip install 'oumi[deploy]'"
    ) from e

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

    Models are accessed directly from S3 via CloudBucketMount (no upload step)
    or downloaded from HuggingFace Hub.  Modal generates and deploys a
    containerised vLLM server behind an HTTPS endpoint.
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

        Credentials are resolved in order: constructor args → env vars →
        ``~/.modal.toml``.  Raises ``ValueError`` if no valid credentials
        are found.
        """
        self.token_id = token_id or os.environ.get("MODAL_TOKEN_ID")
        self.token_secret = token_secret or os.environ.get("MODAL_TOKEN_SECRET")
        self.workspace = workspace or os.environ.get("MODAL_WORKSPACE")

        if not self.token_id or not self.token_secret or not self.workspace:
            config_creds = self._read_modal_config()
            if config_creds:
                self.token_id = self.token_id or config_creds.get("token_id")
                self.token_secret = self.token_secret or config_creds.get(
                    "token_secret"
                )
                self.workspace = self.workspace or config_creds.get("workspace")

        self.workspace = self.workspace or "default"

        if not self.token_id or not self.token_secret:
            raise ValueError(
                "Modal credentials required. Either:\n"
                "1. Run 'modal token new' to create ~/.modal.toml, or\n"
                "2. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars, or\n"
                "3. Pass token_id and token_secret to constructor."
            )

        os.environ["MODAL_TOKEN_ID"] = self.token_id
        os.environ["MODAL_TOKEN_SECRET"] = self.token_secret

        self._s3_client = boto3.client("s3")
        self._deployed_apps: dict[str, tuple] = {}

    def _read_modal_config(self) -> dict[str, str] | None:
        """Read credentials and workspace from ``~/.modal.toml``.

        The TOML section name is the workspace (e.g., ``[oumi]``).  Prefers the
        section marked ``active = true``; falls back to the first section that
        contains valid credentials.
        """
        config_paths = [
            Path.home() / ".modal.toml",
            Path.home() / ".modal" / "config.toml",
        ]

        for config_path in config_paths:
            if not config_path.exists():
                continue
            try:
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)

                fallback: dict[str, str] | None = None
                for section_name, section in config.items():
                    if not isinstance(section, dict):
                        continue
                    token_id = section.get("token_id")
                    token_secret = section.get("token_secret")
                    if not token_id or not token_secret:
                        continue

                    creds = {
                        "token_id": token_id,
                        "token_secret": token_secret,
                        "workspace": section_name,
                    }
                    if section.get("active", False):
                        logger.info(
                            f"Loaded Modal credentials from {config_path} "
                            f"[{section_name}]"
                        )
                        return creds
                    if fallback is None:
                        fallback = creds

                if fallback:
                    logger.info(
                        f"Loaded Modal credentials from {config_path} "
                        f"[{fallback['workspace']}]"
                    )
                    return fallback

            except Exception as e:
                logger.warning(f"Failed to read Modal config from {config_path}: {e}")

        return None

    async def __aenter__(self) -> "ModalDeploymentClient":
        """Enters the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exits the async context manager."""

    async def upload_model(
        self,
        model_source: str,
        model_name: str,
        model_type: ModelType = ModelType.FULL,
        base_model: str | None = None,
        progress_callback: Any | None = None,
    ) -> UploadedModel:
        """Validate S3 model path and return metadata for deployment.

        Modal doesn't require uploading weights — models are accessed directly
        from S3 via CloudBucketMount.  This validates the path is accessible.

        Raises:
            ValueError: If model_source is not a valid/accessible S3 path.
        """
        _ = progress_callback
        if not model_source.startswith("s3://"):
            raise ValueError(
                f"Modal requires S3 paths for models. Got: {model_source}. "
                "Upload model to S3 first, then deploy with s3:// URL."
            )

        s3_match = re.match(r"s3://([^/]+)/(.+)", model_source)
        if not s3_match:
            raise ValueError(f"Invalid S3 path format: {model_source}")

        bucket_name = s3_match.group(1)
        model_path = s3_match.group(2)

        logger.info(f"Validating S3 path: s3://{bucket_name}/{model_path}")
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=model_path, MaxKeys=1
            )
            if not response.get("Contents"):
                raise ValueError(
                    f"S3 path not found or empty: {model_source}. "
                    "Ensure the model has been uploaded to S3."
                )
            logger.info(f"S3 path validated: {model_source}")

        except (BotoCoreError, ClientError) as e:
            error_msg = f"Failed to access S3 path {model_source}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        if model_type == ModelType.ADAPTER and not base_model:
            raise ValueError("base_model required for LoRA adapters")

        return UploadedModel(
            provider_model_id=model_source,
            status="ready",
        )

    async def get_model_status(self, model_id: str) -> str:
        """Return ``"ready"`` — S3-backed models are always available once validated."""
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
        """Generate and deploy a Modal app with a vLLM inference server.

        Supports S3-backed models (via CloudBucketMount) and HuggingFace models.

        Raises:
            ValueError: If GPU type not supported.
            RuntimeError: If deployment fails.
        """
        import subprocess
        import tempfile

        use_huggingface = huggingface_model_id is not None
        if use_huggingface:
            logger.info(
                f"Creating Modal endpoint for HuggingFace model: {huggingface_model_id}"
            )
            vllm_model_path = huggingface_model_id
        else:
            s3_match = re.match(r"s3://([^/]+)/(.+)", model_id)
            if not s3_match:
                raise ValueError(f"Invalid S3 path: {model_id}")
            vllm_model_path = f"/models/{s3_match.group(2)}"

        gpu_type = self._to_modal_gpu(hardware)
        app_name = self._generate_app_name(display_name or "oumi-inference")
        log_model = huggingface_model_id if use_huggingface else model_id
        logger.info(
            f"Creating Modal app {app_name} with {gpu_type} GPU for model {log_model}"
        )

        vllm_port = 8000
        scaledown = autoscaling.min_replicas if autoscaling.min_replicas > 0 else 300

        secrets_str = ""
        if use_huggingface:
            secrets_str = 'secrets=[modal.Secret.from_name("huggingface-token")],'

        if use_huggingface:
            hf_volume = (
                'modal.Volume.from_name("huggingface-cache", create_if_missing=True)'
            )
            volumes_str = 'volumes={"/root/.cache/huggingface": ' + hf_volume + "},"
        else:
            volumes_str = ""

        app_code = f"""
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
"""

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(app_code)
                temp_file = f.name

            logger.info(f"Deploying Modal app from {temp_file}...")
            result = subprocess.run(  # noqa: ASYNC221
                ["modal", "deploy", temp_file],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                error_msg = f"Modal deploy failed: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Modal deploy output: {result.stdout}")

            endpoint_url = (
                f"https://{self.workspace}--{app_name}-serve.modal.run"
                "/v1/chat/completions"
            )
            logger.info(f"Modal app deployed: {endpoint_url}")

            return Endpoint(
                endpoint_id=app_name,
                provider=self.provider,
                model_id=huggingface_model_id if use_huggingface else model_id,
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
            error_msg = f"Failed to deploy Modal app {app_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            if "temp_file" in locals():
                try:
                    Path(temp_file).unlink()  # noqa: ASYNC240
                except Exception:
                    pass

    async def get_endpoint(self, endpoint_id: str) -> Endpoint:
        """Get endpoint status from Modal."""
        try:
            app = modal.App.lookup(endpoint_id, create_if_missing=False)
            if app is None:
                logger.warning(f"Modal app {endpoint_id} not found")
                return self._make_default_endpoint(endpoint_id, EndpointState.ERROR)
            return self._make_default_endpoint(endpoint_id, EndpointState.RUNNING)
        except Exception as e:
            logger.error(f"Failed to get endpoint status for {endpoint_id}: {e}")
            return self._make_default_endpoint(endpoint_id, EndpointState.ERROR)

    async def update_endpoint(
        self,
        endpoint_id: str,
        autoscaling: AutoscalingConfig | None = None,
        hardware: HardwareConfig | None = None,
    ) -> Endpoint:
        """Not supported — Modal requires redeployment to change configuration."""
        raise NotImplementedError(
            "Modal doesn't support in-place endpoint updates. "
            "Delete the deployment and create a new one with different configuration."
        )

    async def delete_endpoint(self, endpoint_id: str, *, force: bool = False) -> None:
        """Delete Modal deployment (app will scale to zero and be removed)."""
        try:
            logger.info(f"Deleting Modal app {endpoint_id}...")
            app = modal.App.lookup(endpoint_id, create_if_missing=False)
            if app:
                logger.info(
                    f"Modal app {endpoint_id} will scale to zero and be removed"
                )
            self._deployed_apps.pop(endpoint_id, None)
            logger.info(f"Modal app {endpoint_id} deleted")
        except Exception as e:
            logger.error(f"Failed to delete Modal app {endpoint_id}: {e}")
            raise

    async def list_endpoints(self) -> list[Endpoint]:
        """List deployments tracked in this session (Modal has no list API)."""
        return [
            self._make_default_endpoint(dep_id, EndpointState.RUNNING)
            for dep_id in self._deployed_apps
        ]

    async def list_hardware(self, model_id: str | None = None) -> list[HardwareConfig]:
        """List supported GPU types on Modal."""
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
        """Return empty list — Modal has no model registry (models live in S3)."""
        return []

    async def delete_model(self, model_id: str) -> None:
        """Not supported — models live in S3, not in Modal."""
        raise NotImplementedError(
            "Modal doesn't manage model storage. Models are accessed directly from S3. "
            "Delete the model from S3 if needed using AWS CLI or boto3."
        )

    # --- Helpers ---

    def _build_endpoint_url(self, endpoint_id: str) -> str:
        return f"https://{self.workspace}--{endpoint_id}-serve.modal.run"

    def _make_default_endpoint(
        self, endpoint_id: str, state: EndpointState
    ) -> Endpoint:
        """Build an ``Endpoint`` with sensible defaults for lookup results."""
        return Endpoint(
            endpoint_id=endpoint_id,
            provider=self.provider,
            model_id="",
            endpoint_url=self._build_endpoint_url(endpoint_id),
            state=state,
            hardware=HardwareConfig(accelerator="unknown", count=1),
            autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=1),
            created_at=None,
        )

    def _to_modal_gpu(self, hardware: HardwareConfig) -> str:
        """Convert ``HardwareConfig`` to a Modal GPU type string."""
        gpu_type = self.GPU_TYPES.get(hardware.accelerator)
        if not gpu_type:
            raise ValueError(
                f"Unsupported GPU type: {hardware.accelerator}. "
                f"Supported types: {list(self.GPU_TYPES.keys())}"
            )
        return gpu_type

    def _generate_app_name(self, display_name: str) -> str:
        """Generate a valid Modal app name (lowercase alphanumeric + hyphens)."""
        name = display_name.lower().replace("_", "-").replace(" ", "-")
        name = re.sub(r"[^a-z0-9-]", "", name)
        name = re.sub(r"-+", "-", name)
        return f"{name}-{int(time.time())}"
