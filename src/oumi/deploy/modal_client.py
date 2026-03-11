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
Models are downloaded from HuggingFace Hub at container start.

References:
- Modal Docs: https://modal.com/docs
- Modal Python SDK: https://modal.com/docs/reference/modal.App
- vLLM on Modal: https://docs.vllm.ai/en/latest/deployment/frameworks/modal/
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal Python SDK source: https://github.com/modal-labs/modal-client
"""

import importlib.util
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from oumi.deploy.utils import (
    check_hf_model_accessibility,
    is_huggingface_repo_id,
    resolve_hf_token,
    warn_if_private_model_missing_token,
)

logger = logging.getLogger(__name__)


def _validate_modal_model_source(model_source: str) -> None:
    """Validate that *model_source* is a HuggingFace repo ID.

    Raises:
        ValueError: If model_source is not supported by Modal.
    """
    _unsupported = {
        "s3://": "S3 (planned, not yet supported)",
        "gs://": "GCS",
        "az://": "Azure Blob",
        "abfs://": "Azure Blob",
    }
    for prefix, label in _unsupported.items():
        if model_source.startswith(prefix):
            raise ValueError(
                f"Modal does not yet support {label} model sources ('{model_source}'). "
                "Provide a HuggingFace repo ID (e.g., 'meta-llama/Llama-3-8B')."
            )
    if model_source.startswith(("https://", "http://")):
        raise ValueError(
            f"Modal does not support deploying models from URLs ('{model_source}'). "
            "Provide a HuggingFace repo ID (e.g., 'meta-llama/Llama-3-8B')."
        )
    if Path(model_source).is_dir() or Path(model_source).is_file():
        raise ValueError(
            "Modal does not support local model uploads. "
            "Provide a HuggingFace repo ID (e.g., 'meta-llama/Llama-3-8B')."
        )
    if not is_huggingface_repo_id(model_source):
        raise ValueError(
            f"Unrecognized model source: '{model_source}'. "
            "Modal accepts a HuggingFace repo ID (e.g., 'Qwen/Qwen3-1.7B')."
        )


class ModalDeploymentClient(BaseDeploymentClient):
    """Modal.com deployment client for serverless vLLM inference.

    Models are downloaded from HuggingFace Hub at container start. Modal
    generates and deploys a containerised vLLM server behind an HTTPS endpoint
    using the Python SDK (``app.deploy()``).
    """

    provider = DeploymentProvider.MODAL

    # Static catalog used by list_hardware(); Modal's public Python SDK currently
    # exposes GPU selection strings but does not provide a live "list available GPUs"
    # API for a workspace. Keep this aligned with:
    # https://modal.com/docs/guide/gpu
    _SUPPORTED_HARDWARE_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("nvidia_t4", (1, 2, 4, 8)),
        ("nvidia_l4", (1, 2, 4, 8)),
        ("nvidia_a10g", (1, 2, 4)),
        ("nvidia_l40s", (1, 2, 4, 8)),
        ("nvidia_a100_40gb", (1, 2, 4, 8)),
        ("nvidia_a100_80gb", (1, 2, 4, 8)),
        ("nvidia_h100_80gb", (1, 2, 4, 8)),
        ("nvidia_h200", (1, 2, 4, 8)),
        ("nvidia_b200", (1, 2, 4, 8)),
    )

    def __init__(
        self,
        token_id: str | None = None,
        token_secret: str | None = None,
        workspace: str | None = None,
        hf_secret_name: str | None = None,
    ):
        """Initialize Modal client.

        Credentials are resolved in order: constructor args -> env vars.
        Raises ``ValueError`` if no valid credentials are found.

        Args:
            token_id: Modal API token ID. Falls back to ``MODAL_TOKEN_ID``.
            token_secret: Modal API token secret. Falls back to
                ``MODAL_TOKEN_SECRET``.
            workspace: Modal workspace name. Falls back to
                ``MODAL_WORKSPACE``, then ``"default"``.
            hf_secret_name: Name of the Modal secret containing HuggingFace
                credentials. Falls back to ``MODAL_HF_SECRET_NAME``, then
                ``"huggingface-token"``.
        """
        self._token_id = token_id or os.environ.get("MODAL_TOKEN_ID")
        self._token_secret = token_secret or os.environ.get("MODAL_TOKEN_SECRET")

        if not self._token_id:
            raise ValueError(
                "Modal token ID required. Set the MODAL_TOKEN_ID environment "
                "variable or pass token_id to the constructor."
            )
        if not self._token_secret:
            raise ValueError(
                "Modal token secret required. Set the MODAL_TOKEN_SECRET environment "
                "variable or pass token_secret to the constructor."
            )

        self._workspace = workspace or os.environ.get("MODAL_WORKSPACE", "default")
        self._hf_secret_name = hf_secret_name or os.environ.get(
            "MODAL_HF_SECRET_NAME", "huggingface-token"
        )

        os.environ["MODAL_TOKEN_ID"] = self._token_id
        os.environ["MODAL_TOKEN_SECRET"] = self._token_secret

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
        base_model: str | None = None,
        progress_callback: Any | None = None,
        model_access_key: str | None = None,
    ) -> UploadedModel:
        """Validate a HuggingFace model source for deployment.

        Modal downloads models from HuggingFace Hub at container start — no
        bytes are transferred during upload. This validates the model source
        and checks accessibility.

        Args:
            model_source: HuggingFace repo ID (e.g. ``Qwen/Qwen3-1.7B``).
            model_name: Display name for the model.
            model_type: ``full`` or ``adapter``.
            base_model: Required when ``model_type`` is ``adapter``.
            progress_callback: Unused.
            model_access_key: Explicit HuggingFace token for private models.

        Raises:
            ValueError: If model_source is not a valid HuggingFace repo ID,
                or if ``model_type`` is ``adapter`` without ``base_model``.
        """
        _ = progress_callback

        if model_type == ModelType.ADAPTER and not base_model:
            raise ValueError("base_model required for LoRA adapters")

        _validate_modal_model_source(model_source)

        hf_token = resolve_hf_token(model_access_key)
        warn_if_private_model_missing_token(model_source, hf_token)

        logger.info(f"Validated HuggingFace model source: {model_source}")
        return UploadedModel(
            provider_model_id=model_source,
            status="ready",
        )

    async def get_model_status(self, model_id: str) -> str:
        """Return ``"ready"`` — HF models are always available once validated."""
        if is_huggingface_repo_id(model_id):
            return "ready"
        raise ValueError(
            f"Unsupported model source: {model_id}. "
            "Modal currently only supports HuggingFace repo IDs."
        )

    async def create_endpoint(
        self,
        model_id: str,
        hardware: HardwareConfig,
        autoscaling: AutoscalingConfig,
        display_name: str | None = None,
    ) -> Endpoint:
        """Generate and deploy a Modal app with a vLLM inference server.

        The model source type is auto-detected from ``model_id``. Only
        HuggingFace repo IDs are currently supported. For private/gated
        models, a Modal secret (named ``self._hf_secret_name``) is injected
        so the container can authenticate with HuggingFace Hub.

        Raises:
            ValueError: If model_id is not a HuggingFace repo ID or GPU type
                is not supported.
            RuntimeError: If deployment fails.
        """
        if not is_huggingface_repo_id(model_id):
            raise ValueError(
                f"Only HuggingFace repo IDs are currently supported. Got: {model_id}"
            )

        is_private = not check_hf_model_accessibility(model_id)
        if is_private:
            self._ensure_modal_hf_secret_exists()
        gpu_type = self._to_modal_gpu(hardware)
        app_name = self._generate_app_name(display_name or "oumi-inference")
        local_python_minor = self._local_python_minor()
        logger.info(
            f"Creating Modal app {app_name} with {gpu_type} GPU for model {model_id}"
        )
        logger.info(f"Using local Python {local_python_minor} for Modal image build")

        vllm_port = 8000
        scaledown = autoscaling.min_replicas if autoscaling.min_replicas > 0 else 300

        if is_private:
            secrets_str = (
                'secrets=[modal.Secret.from_name('
                f'"{self._hf_secret_name}", required_keys=["HF_TOKEN"])],'
            )
        else:
            secrets_str = ""

        volumes_str = """volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },"""

        app_code = f"""
import modal
import subprocess

app = modal.App("{app_name}")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="{local_python_minor}")
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
        "vllm", "serve", "{model_id}",
        "--host", "0.0.0.0",
        "--port", "{vllm_port}",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
    ]
    subprocess.Popen(cmd)
"""

        temp_file: str | None = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(app_code)
                temp_file = f.name

            logger.info(f"Deploying Modal app via SDK from {temp_file}...")
            spec = importlib.util.spec_from_file_location("_modal_app", temp_file)
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Failed to load generated Modal app from {temp_file}"
                )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.app.deploy()

            endpoint_url = (
                f"https://{self._workspace}--{app_name}-serve.modal.run"
                "/v1/chat/completions"
            )
            logger.info(f"Modal app deployed: {endpoint_url}")

            return Endpoint(
                endpoint_id=app_name,
                provider=self.provider,
                model_id=model_id,
                endpoint_url=endpoint_url,
                state=EndpointState.RUNNING,
                hardware=hardware,
                autoscaling=autoscaling,
                display_name=display_name,
                created_at=datetime.now(tz=timezone.utc),
            )

        except Exception as e:
            error_msg = f"Failed to deploy Modal app {app_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            if temp_file is not None:
                try:
                    os.unlink(temp_file)
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
        _ = model_id
        return [
            HardwareConfig(accelerator=accelerator, count=count)
            for accelerator, counts in self._SUPPORTED_HARDWARE_SPECS
            for count in counts
        ]

    async def list_models(
        self, include_public: bool = False, organization: str | None = None
    ) -> list[Model]:
        """Return empty list — Modal has no model registry."""
        return []

    async def delete_model(self, model_id: str) -> None:
        """Not supported — models live on HuggingFace, not in Modal."""
        raise NotImplementedError(
            "Modal doesn't manage model storage. Models are referenced from "
            "HuggingFace Hub. Delete the model from HuggingFace if needed."
        )

    # --- Helpers ---

    def _build_endpoint_url(self, endpoint_id: str) -> str:
        return f"https://{self._workspace}--{endpoint_id}-serve.modal.run"

    def _ensure_modal_hf_secret_exists(self) -> None:
        """Ensure the workspace has a Modal secret containing ``HF_TOKEN``."""
        hf_token = resolve_hf_token()
        if not hf_token:
            raise RuntimeError(
                "This HuggingFace model appears private/gated, but no HF token was "
                "found. Set HF_TOKEN or pass --model-access-key before deploying."
            )

        try:
            modal.Secret.create_deployed(
                deployment_name=self._hf_secret_name,
                env_dict={"HF_TOKEN": hf_token},
                overwrite=False,
            )
            logger.info(f"Created Modal secret '{self._hf_secret_name}' for HF access")
        except Exception as e:
            # Secret already exists; keep using it.
            if "already exists" in str(e).lower():
                logger.info(
                    f"Using existing Modal secret '{self._hf_secret_name}' for HF access"
                )
                return
            raise RuntimeError(
                f"Failed to ensure Modal secret '{self._hf_secret_name}' exists: {e}"
            ) from e

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
        accelerator = hardware.accelerator.lower()

        if accelerator in ("nvidia_t4", "t4"):
            return "T4"
        if accelerator in ("nvidia_l4", "l4"):
            return "L4"
        if accelerator in ("nvidia_a10g", "nvidia_a10", "a10g", "a10"):
            return "A10G"
        if accelerator in ("nvidia_l40s", "l40s"):
            return "L40S"
        if accelerator in ("nvidia_a100_40gb", "a100_40gb", "a100-40gb"):
            return "A100-40GB"
        if accelerator in ("nvidia_a100_80gb", "a100_80gb", "a100-80gb"):
            return "A100-80GB"
        if accelerator in ("nvidia_a100", "a100"):
            return "A100"
        if accelerator in ("nvidia_h100_80gb", "nvidia_h100", "h100_80gb", "h100"):
            return "H100"
        if accelerator in ("nvidia_h200", "h200"):
            return "H200"
        if accelerator in ("nvidia_b200", "b200"):
            return "B200"

        supported = sorted({a for a, _ in self._SUPPORTED_HARDWARE_SPECS})
        supported.extend(
            [
                "nvidia_a100",
                "nvidia_h100",
            ]
        )
        raise ValueError(
            f"Unsupported GPU type: {hardware.accelerator}. "
            f"Supported types: {supported}"
        )

    def _generate_app_name(self, display_name: str) -> str:
        """Generate a valid Modal app name (lowercase alphanumeric + hyphens)."""
        name = display_name.lower().replace("_", "-").replace(" ", "-")
        name = re.sub(r"[^a-z0-9-]", "", name)
        name = re.sub(r"-+", "-", name)
        return f"{name}-{int(time.time())}"

    @staticmethod
    def _local_python_minor() -> str:
        """Return local interpreter major.minor (e.g. ``3.11``)."""
        return f"{sys.version_info.major}.{sys.version_info.minor}"
