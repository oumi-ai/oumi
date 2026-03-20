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

import asyncio
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

_SCALEDOWN_WINDOW_SECONDS = 15 * 60  # Modal range: [2, 3600]
_VLLM_PROBE_TIMEOUT_SECONDS = 10  # Quick probe for get_endpoint / list_endpoints
_VLLM_READY_TIMEOUT_SECONDS = 300  # Full cold-start wait for test_endpoint
_VLLM_POLL_INTERVAL_SECONDS = 5

# ``modal app list --json`` State strings → EndpointState
_MODAL_CLI_STATE_MAP: dict[str, EndpointState] = {
    "deployed": EndpointState.RUNNING,
    "ephemeral": EndpointState.RUNNING,
    "ephemeral (detached)": EndpointState.RUNNING,
    "initializing...": EndpointState.STARTING,
    "stopping...": EndpointState.STOPPING,
    "stopped": EndpointState.STOPPED,
    "disabled": EndpointState.STOPPED,
}


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
        hf_secret_name: str | None = None,
    ):
        """Initialize Modal client.

        Credentials are resolved in order: constructor args -> env vars.
        Raises ``ValueError`` if no valid credentials are found.

        Args:
            token_id: Modal API token ID. Falls back to ``MODAL_TOKEN_ID``.
            token_secret: Modal API token secret. Falls back to
                ``MODAL_TOKEN_SECRET``.
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

        if is_private:
            secrets_str = (
                "secrets=[modal.Secret.from_name("
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
    scaledown_window={_SCALEDOWN_WINDOW_SECONDS},
    timeout=600,
    {volumes_str}
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port={vllm_port}, startup_timeout=600)
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
            await mod.app.deploy.aio()

            endpoint_url = await self._build_endpoint_url(app_name)
            logger.info(f"Modal app deployed: {endpoint_url}")

            self._deployed_apps[app_name] = (None, app_name)

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
        """Get endpoint status from Modal.

        If the app is found, a quick probe (≤ 10 s) is made against the
        vLLM ``/v1/models`` endpoint to resolve ``model_id``.  When the
        container is cold the probe times out harmlessly and ``model_id``
        is left as ``None``.
        """
        try:
            app = await modal.App.lookup.aio(endpoint_id, create_if_missing=False)
            if app is None:
                logger.warning(f"Modal app {endpoint_id} not found")
                return await self._make_default_endpoint(
                    endpoint_id, EndpointState.ERROR
                )
            model_id = await self._try_fetch_vllm_model_id(endpoint_id)
            return await self._make_default_endpoint(
                endpoint_id, EndpointState.RUNNING, model_id=model_id
            )
        except Exception as e:
            logger.error(f"Failed to get endpoint status for {endpoint_id}: {e}")
            return await self._make_default_endpoint(endpoint_id, EndpointState.ERROR)

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
            app = await modal.App.lookup.aio(endpoint_id, create_if_missing=False)
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
        """List all Modal apps by invoking the ``modal`` CLI.

        Runs ``modal app list --json`` in a subprocess — this is the
        **stable, documented** way to enumerate apps.  The Modal Python
        SDK does not expose a public ``list`` API; the CLI internally
        calls a gRPC endpoint (``client.stub.AppList``) and formats the
        result.  We parse the JSON output to avoid depending on private
        SDK internals that may change across versions.

        For each **running** app a quick vLLM probe (≤ 10 s) is
        attempted to populate ``model_id``.
        """
        import json as _json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        try:
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "-m", "modal", "app", "list", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )
            if proc.returncode != 0:
                logger.error(
                    "modal app list failed (rc=%d): %s",
                    proc.returncode,
                    proc.stderr.strip(),
                )
                return []
            apps: list[dict[str, Any]] = _json.loads(proc.stdout)
        except Exception as e:
            logger.error("Failed to list Modal apps: %s", e)
            return []

        endpoints: list[Endpoint] = []
        for app_info in apps:
            endpoint_id = app_info.get("Description", "")
            cli_state = app_info.get("State", "").lower()
            state = _MODAL_CLI_STATE_MAP.get(cli_state, EndpointState.ERROR)

            try:
                endpoint_url = await self._build_endpoint_url(endpoint_id)
            except Exception:
                endpoint_url = None

            created_str = app_info.get("Created at")
            created_at = datetime.fromisoformat(created_str) if created_str else None

            endpoints.append(
                Endpoint(
                    endpoint_id=endpoint_id,
                    provider=self.provider,
                    model_id=None,
                    endpoint_url=endpoint_url,
                    state=state,
                    hardware=HardwareConfig(accelerator="unknown", count=1),
                    autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=1),
                    created_at=created_at,
                )
            )
        return endpoints

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

    # --- vLLM model discovery & health check ---

    @staticmethod
    def _vllm_base_url(endpoint_url: str) -> str:
        """Strip ``/v1/chat/completions`` to get the vLLM server root."""
        return endpoint_url.removesuffix("/v1/chat/completions")

    async def _fetch_vllm_model_id(
        self,
        base_url: str,
        timeout: float = _VLLM_READY_TIMEOUT_SECONDS,
        poll_interval: float = _VLLM_POLL_INTERVAL_SECONDS,
    ) -> str:
        """Poll ``GET /v1/models`` until vLLM reports a loaded model.

        Blocks with retries until the server is warm and returns the model
        name.  Doubles as a readiness / health check — the endpoint is
        considered healthy once this method returns successfully.

        Returns:
            The model ID reported by vLLM (e.g. ``"Qwen/Qwen3-1.7B"``).

        Raises:
            RuntimeError: If *timeout* is exceeded before a model is reported.
        """
        import httpx  # noqa: PLC0415

        models_url = f"{base_url.rstrip('/')}/v1/models"
        headers = self._get_inference_auth_headers()
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() < deadline:
                sleep_time = poll_interval
                try:
                    resp = await client.get(models_url, headers=headers)
                    if resp.status_code == 429:
                        # Modal rate-limits during cold start; back off before retrying.
                        retry_after = resp.headers.get("retry-after")
                        sleep_time = float(retry_after) if retry_after else 30.0
                        last_error = httpx.HTTPStatusError(
                            f"Client error '429 Too Many Requests' for url '{models_url}'",
                            request=resp.request,
                            response=resp,
                        )
                        logger.debug(
                            "vLLM endpoint rate-limited (429) at %s, backing off %.0fs",
                            models_url,
                            sleep_time,
                        )
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        models = data.get("data", [])
                        if models:
                            model_id: str = models[0]["id"]
                            logger.info(
                                "vLLM reports model %r at %s", model_id, models_url
                            )
                            return model_id
                except Exception as e:
                    last_error = e
                    logger.debug("vLLM not ready at %s: %s", models_url, e)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(sleep_time, remaining))

        raise RuntimeError(
            f"Timed out after {timeout:.0f}s waiting for vLLM to report a "
            f"model at {models_url}. Last error: {last_error}"
        )

    async def _try_fetch_vllm_model_id(
        self, endpoint_id: str, timeout: float = _VLLM_PROBE_TIMEOUT_SECONDS
    ) -> str | None:
        """Best-effort probe for the model name from a live vLLM server.

        Returns ``None`` (instead of raising) when the server is unreachable,
        cold, or the timeout expires.
        """
        try:
            endpoint_url = await self._build_endpoint_url(endpoint_id)
            base_url = self._vllm_base_url(endpoint_url)
            return await self._fetch_vllm_model_id(base_url, timeout=timeout)
        except Exception as e:
            logger.debug(
                "Could not fetch model name from vLLM for %s: %s", endpoint_id, e
            )
            return None

    # --- Overrides ---

    async def test_endpoint(
        self,
        endpoint_url: str,
        prompt: str,
        model_id: str | None = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Send a test prompt, auto-discovering the model name from vLLM.

        If *model_id* is not provided, polls the vLLM ``/v1/models`` endpoint
        (handling cold starts with up to a 300 s wait) to discover the served
        model name before sending the inference request.
        """
        if not model_id:
            base_url = self._vllm_base_url(endpoint_url)
            model_id = await self._fetch_vllm_model_id(base_url)
        return await super().test_endpoint(
            endpoint_url=endpoint_url,
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    # --- Helpers ---

    async def _build_endpoint_url(self, endpoint_id: str) -> str:
        """Resolve the chat completions URL for a deployed Modal app."""
        fn = modal.Function.from_name(endpoint_id, "serve")
        base_url = await fn.get_web_url.aio()
        if not base_url:
            raise RuntimeError(
                f"Modal SDK returned no web URL for app '{endpoint_id}'. "
                "Is the app deployed?"
            )
        return f"{base_url.rstrip('/')}/v1/chat/completions"

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

    async def _make_default_endpoint(
        self,
        endpoint_id: str,
        state: EndpointState,
        model_id: str | None = None,
    ) -> Endpoint:
        """Build an ``Endpoint`` for a looked-up Modal app.

        ``model_id`` is not stored in Modal app metadata, so it is ``None``
        when an endpoint is looked up by ID alone (e.g. via ``get_endpoint``
        or ``list_endpoints``).  Callers that do know the model (e.g.
        ``create_endpoint``) should pass it explicitly.

        Note: because ``model_id`` is unknown after a lookup, calling
        ``test_endpoint`` on such an endpoint will raise ``ValueError``
        asking for the model name.  This is intentional — sending a wrong
        model name to vLLM causes a 404.
        """
        return Endpoint(
            endpoint_id=endpoint_id,
            provider=self.provider,
            model_id=model_id,
            endpoint_url=await self._build_endpoint_url(endpoint_id),
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
