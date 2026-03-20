"""Unit tests for Modal deployment client."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.deploy.base_client import (
    AutoscalingConfig,
    EndpointState,
    HardwareConfig,
    ModelType,
)
from oumi.deploy.modal_client import ModalDeploymentClient


def _mock_httpx_client(response_data=None, side_effect=None):
    """Build a mock ``httpx.AsyncClient`` usable as an async context manager."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = response_data or {}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.post.return_value = mock_resp
    if side_effect:
        mock_client.get.side_effect = side_effect
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    return mock_client


@pytest.fixture
def modal_client():
    """Create Modal client with mock credentials."""
    with patch.dict(
        "os.environ",
        {
            "MODAL_TOKEN_ID": "test-token-id",
            "MODAL_TOKEN_SECRET": "test-token-secret",
        },
    ):
        client = ModalDeploymentClient(
            token_id="test-token-id",
            token_secret="test-token-secret",
        )
        return client


def test_gpu_type_mapping(modal_client):
    """Test GPU type conversion to Modal format."""
    hw = HardwareConfig(accelerator="nvidia_a100_80gb", count=1)
    assert modal_client._to_modal_gpu(hw) == "A100-80GB"

    hw = HardwareConfig(accelerator="nvidia_h100_80gb", count=2)
    assert modal_client._to_modal_gpu(hw) == "H100"

    hw = HardwareConfig(accelerator="nvidia_v100", count=1)
    with pytest.raises(ValueError, match="Unsupported GPU type"):
        modal_client._to_modal_gpu(hw)


def test_local_python_minor(modal_client):
    """Uses the current interpreter version for generated Modal images."""
    assert modal_client._local_python_minor() == (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )


def test_app_name_generation(modal_client):
    """Test Modal app name generation from display name."""
    name = modal_client._generate_app_name("My Test Model")
    assert name.startswith("my-test-model-")
    assert name.islower()
    assert " " not in name

    name = modal_client._generate_app_name("Model@123_test")
    assert "@" not in name
    assert "_" not in name


@pytest.mark.asyncio
async def test_upload_model_hf_repo(modal_client):
    """Test upload_model accepts a HuggingFace repo ID."""
    with patch(
        "oumi.deploy.modal_client.check_hf_model_accessibility", return_value=True
    ):
        result = await modal_client.upload_model(
            model_source="Qwen/Qwen3-1.7B",
            model_name="qwen3-1-7b",
            model_type=ModelType.FULL,
        )

    assert result.provider_model_id == "Qwen/Qwen3-1.7B"
    assert result.status == "ready"


@pytest.mark.asyncio
async def test_upload_model_rejects_s3(modal_client):
    """Test upload_model rejects S3 paths (not yet supported)."""
    with pytest.raises(ValueError, match="does not yet support S3"):
        await modal_client.upload_model(
            model_source="s3://bucket/models/llama-7b",
            model_name="test",
        )


@pytest.mark.asyncio
async def test_upload_model_rejects_local_path(modal_client):
    """Test upload_model rejects non-HF sources."""
    with pytest.raises(ValueError, match="Unrecognized model source"):
        await modal_client.upload_model(
            model_source="/local/path/to/model",
            model_name="test",
        )


@pytest.mark.asyncio
async def test_get_model_status(modal_client):
    """Test get_model_status returns ready for HF repo IDs."""
    status = await modal_client.get_model_status("Qwen/Qwen3-1.7B")
    assert status == "ready"


@pytest.mark.asyncio
async def test_get_model_status_rejects_non_hf(modal_client):
    """Test get_model_status raises for non-HF sources."""
    with pytest.raises(ValueError, match="Unsupported model source"):
        await modal_client.get_model_status("s3://bucket/path/to/model")


@pytest.mark.asyncio
async def test_create_endpoint(modal_client):
    """Test endpoint creation generates and deploys a Modal app."""
    with (
        patch(
            "oumi.deploy.modal_client.check_hf_model_accessibility",
            return_value=True,
        ),
        patch("oumi.deploy.modal_client.importlib.util") as mock_importlib,
        patch("oumi.deploy.modal_client.modal") as mock_modal,
    ):
        mock_spec = MagicMock()
        mock_importlib.spec_from_file_location.return_value = mock_spec
        mock_mod = MagicMock()
        mock_mod.app.deploy.aio = AsyncMock()
        mock_importlib.module_from_spec.return_value = mock_mod

        mock_fn = MagicMock()
        mock_fn.get_web_url.aio = AsyncMock(
            return_value="https://myworkspace--the-app-serve.modal.run"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        endpoint = await modal_client.create_endpoint(
            model_id="Qwen/Qwen3-1.7B",
            hardware=HardwareConfig(accelerator="nvidia_h100_80gb", count=1),
            autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=2),
            display_name="test-model",
        )

        assert endpoint.provider == modal_client.provider
        assert endpoint.state == EndpointState.RUNNING
        assert endpoint.endpoint_url == (
            "https://myworkspace--the-app-serve.modal.run/v1/chat/completions"
        )


@pytest.mark.asyncio
async def test_list_hardware(modal_client):
    """Test listing available hardware."""
    hardware_list = await modal_client.list_hardware()

    assert len(hardware_list) > 0

    # Check A100 is available
    a100_configs = [hw for hw in hardware_list if "a100" in hw.accelerator]
    assert len(a100_configs) > 0

    # Check multi-GPU configs exist
    multi_gpu = [hw for hw in hardware_list if hw.count > 1]
    assert len(multi_gpu) > 0


@pytest.mark.asyncio
async def test_update_endpoint_raises(modal_client):
    """Test that updates raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="in-place endpoint updates"):
        await modal_client.update_endpoint(
            endpoint_id="test-123",
            hardware=HardwareConfig(accelerator="nvidia_h100_80gb", count=2),
        )


@pytest.mark.asyncio
async def test_delete_model_raises(modal_client):
    """Test that delete_model raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="doesn't manage model storage"):
        await modal_client.delete_model("s3://bucket/model")


@pytest.mark.asyncio
async def test_list_models_returns_empty(modal_client):
    """Test that list_models returns empty list."""
    models = await modal_client.list_models()
    assert models == []


@pytest.mark.asyncio
async def test_get_endpoint(modal_client):
    """get_endpoint probes vLLM and populates model_id when warm."""
    with (
        patch("oumi.deploy.modal_client.modal") as mock_modal,
        patch.object(
            modal_client,
            "_try_fetch_vllm_model_id",
            new_callable=AsyncMock,
            return_value="Qwen/Qwen3-1.7B",
        ),
    ):
        mock_app = MagicMock()
        mock_modal.App.lookup.aio = AsyncMock(return_value=mock_app)

        mock_fn = MagicMock()
        mock_fn.get_web_url.aio = AsyncMock(
            return_value="https://test-workspace--test-app-123-serve.modal.run"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        endpoint = await modal_client.get_endpoint("test-app-123")

        assert endpoint.endpoint_id == "test-app-123"
        assert endpoint.state == EndpointState.RUNNING
        assert endpoint.model_id == "Qwen/Qwen3-1.7B"
        assert endpoint.endpoint_url == (
            "https://test-workspace--test-app-123-serve.modal.run/v1/chat/completions"
        )
        mock_modal.App.lookup.aio.assert_called_once_with(
            "test-app-123", create_if_missing=False
        )


@pytest.mark.asyncio
async def test_get_endpoint_cold_container(modal_client):
    """get_endpoint returns model_id=None when vLLM probe times out (cold)."""
    with (
        patch("oumi.deploy.modal_client.modal") as mock_modal,
        patch.object(
            modal_client,
            "_try_fetch_vllm_model_id",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        mock_app = MagicMock()
        mock_modal.App.lookup.aio = AsyncMock(return_value=mock_app)

        mock_fn = MagicMock()
        mock_fn.get_web_url.aio = AsyncMock(
            return_value="https://ws--app-serve.modal.run"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        endpoint = await modal_client.get_endpoint("cold-app")

        assert endpoint.state == EndpointState.RUNNING
        assert endpoint.model_id is None


@pytest.mark.asyncio
async def test_get_endpoint_not_found(modal_client):
    """Test getting endpoint that doesn't exist."""
    with patch("oumi.deploy.modal_client.modal") as mock_modal:
        mock_modal.App.lookup.aio = AsyncMock(return_value=None)

        mock_fn = MagicMock()
        mock_fn.get_web_url.aio = AsyncMock(
            return_value="https://ws--missing-app-serve.modal.run"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        endpoint = await modal_client.get_endpoint("missing-app")

        assert endpoint.endpoint_id == "missing-app"
        assert endpoint.state == EndpointState.ERROR
        assert endpoint.endpoint_url.endswith("/v1/chat/completions")


@pytest.mark.asyncio
async def test_delete_endpoint(modal_client):
    """Test deleting an endpoint."""
    with patch("oumi.deploy.modal_client.modal") as mock_modal:
        mock_app = MagicMock()
        mock_modal.App.lookup.aio = AsyncMock(return_value=mock_app)

        modal_client._deployed_apps["test-app"] = (mock_app, "test-app")

        await modal_client.delete_endpoint("test-app")

        assert "test-app" not in modal_client._deployed_apps


@pytest.mark.asyncio
async def test_list_endpoints(modal_client):
    """list_endpoints parses ``modal app list --json`` and probes vLLM."""
    import json

    cli_json = json.dumps(
        [
            {
                "App ID": "ap-abc123",
                "Description": "my-app-deployed",
                "State": "deployed",
                "Tasks": "0",
                "Created at": "2025-11-14 22:13:20+00:00",
                "Stopped at": None,
            },
            {
                "App ID": "ap-def456",
                "Description": "my-app-stopped",
                "State": "stopped",
                "Tasks": "0",
                "Created at": "2025-11-13 08:00:00+00:00",
                "Stopped at": "2025-11-13 09:00:00+00:00",
            },
        ]
    )

    mock_proc = MagicMock(returncode=0, stdout=cli_json, stderr="")

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch("oumi.deploy.modal_client.modal") as mock_modal,
        patch.object(
            modal_client,
            "_try_fetch_vllm_model_id",
            new_callable=AsyncMock,
            return_value="Qwen/Qwen3-1.7B",
        ) as mock_probe,
    ):
        mock_fn = MagicMock()
        mock_fn.get_web_url.aio = AsyncMock(
            return_value="https://ws--app-serve.modal.run"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        endpoints = await modal_client.list_endpoints()

    assert len(endpoints) == 2

    ep_deployed = endpoints[0]
    assert ep_deployed.endpoint_id == "my-app-deployed"
    assert ep_deployed.state.value == "running"
    assert ep_deployed.model_id == "Qwen/Qwen3-1.7B"
    assert ep_deployed.created_at is not None

    ep_stopped = endpoints[1]
    assert ep_stopped.endpoint_id == "my-app-stopped"
    assert ep_stopped.state.value == "stopped"
    assert ep_stopped.model_id is None

    # Only the running app should have been probed
    mock_probe.assert_called_once_with("my-app-deployed")


def test_ensure_modal_hf_secret_exists_creates_secret(modal_client):
    """Creates the HF secret when token is available."""
    with (
        patch(
            "oumi.deploy.modal_client.resolve_hf_token", return_value="hf_test_token"
        ),
        patch("oumi.deploy.modal_client.modal") as mock_modal,
    ):
        modal_client._ensure_modal_hf_secret_exists()

    mock_modal.Secret.create_deployed.assert_called_once_with(
        deployment_name="huggingface-token",
        env_dict={"HF_TOKEN": "hf_test_token"},
        overwrite=False,
    )


def test_ensure_modal_hf_secret_exists_raises_without_token(modal_client):
    """Raises a clear error if no HF token is available."""
    with patch("oumi.deploy.modal_client.resolve_hf_token", return_value=""):
        with pytest.raises(RuntimeError, match="no HF token was found"):
            modal_client._ensure_modal_hf_secret_exists()


def test_ensure_modal_hf_secret_exists_allows_existing_secret(modal_client):
    """Treats 'already exists' as success."""
    with (
        patch(
            "oumi.deploy.modal_client.resolve_hf_token", return_value="hf_test_token"
        ),
        patch("oumi.deploy.modal_client.modal") as mock_modal,
    ):
        mock_modal.Secret.create_deployed.side_effect = RuntimeError(
            "Secret already exists"
        )
        modal_client._ensure_modal_hf_secret_exists()


# --- _fetch_vllm_model_id / _try_fetch_vllm_model_id ---


@pytest.mark.asyncio
async def test_fetch_vllm_model_id_success(modal_client):
    """Returns model name when vLLM /v1/models responds immediately."""
    mock_client = _mock_httpx_client(
        response_data={"data": [{"id": "Qwen/Qwen3-1.7B", "object": "model"}]}
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        model_id = await modal_client._fetch_vllm_model_id(
            "https://ws--app-serve.modal.run", timeout=5.0
        )

    assert model_id == "Qwen/Qwen3-1.7B"
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_vllm_model_id_retries_then_succeeds(modal_client):
    """Retries on connection error, then returns model on success."""
    import httpx as _httpx

    success_resp = MagicMock()
    success_resp.raise_for_status = MagicMock()
    success_resp.json.return_value = {
        "data": [{"id": "meta-llama/Llama-3-8B", "object": "model"}]
    }

    mock_client = AsyncMock()
    mock_client.get.side_effect = [
        _httpx.ConnectError("Connection refused"),
        success_resp,
    ]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("oumi.deploy.modal_client.asyncio.sleep", new_callable=AsyncMock),
    ):
        model_id = await modal_client._fetch_vllm_model_id(
            "https://ws--app-serve.modal.run", timeout=30.0, poll_interval=1.0
        )

    assert model_id == "meta-llama/Llama-3-8B"
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_vllm_model_id_timeout(modal_client):
    """Raises RuntimeError when the server never becomes ready."""
    import httpx as _httpx

    mock_client = AsyncMock()
    mock_client.get.side_effect = _httpx.ConnectError("Connection refused")
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("oumi.deploy.modal_client.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(RuntimeError, match="Timed out"):
            await modal_client._fetch_vllm_model_id(
                "https://ws--app-serve.modal.run", timeout=0.01
            )


@pytest.mark.asyncio
async def test_try_fetch_vllm_model_id_returns_none_on_failure(modal_client):
    """_try_fetch_vllm_model_id swallows errors and returns None."""
    with (
        patch.object(
            modal_client,
            "_fetch_vllm_model_id",
            new_callable=AsyncMock,
            side_effect=RuntimeError("timeout"),
        ),
        patch("oumi.deploy.modal_client.modal") as mock_modal,
    ):
        mock_fn = MagicMock()
        mock_fn.get_web_url.aio = AsyncMock(
            return_value="https://ws--app-serve.modal.run"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        result = await modal_client._try_fetch_vllm_model_id("cold-app")

    assert result is None


# --- test_endpoint override ---


@pytest.mark.asyncio
async def test_test_endpoint_auto_discovers_model(modal_client):
    """test_endpoint auto-discovers model_id from vLLM when not provided."""
    mock_client = _mock_httpx_client(
        response_data={"choices": [{"message": {"content": "Hi there!"}}]}
    )

    with (
        patch.object(
            modal_client,
            "_fetch_vllm_model_id",
            new_callable=AsyncMock,
            return_value="Qwen/Qwen3-1.7B",
        ) as mock_fetch,
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await modal_client.test_endpoint(
            endpoint_url="https://ws--app-serve.modal.run/v1/chat/completions",
            prompt="Hello",
        )

    mock_fetch.assert_called_once_with("https://ws--app-serve.modal.run")
    assert result == {"choices": [{"message": {"content": "Hi there!"}}]}
    call_json = mock_client.post.call_args[1]["json"]
    assert call_json["model"] == "Qwen/Qwen3-1.7B"


@pytest.mark.asyncio
async def test_test_endpoint_uses_explicit_model_id(modal_client):
    """test_endpoint skips auto-discovery when model_id is given."""
    mock_client = _mock_httpx_client(
        response_data={"choices": [{"message": {"content": "Hi!"}}]}
    )

    with (
        patch.object(
            modal_client,
            "_fetch_vllm_model_id",
            new_callable=AsyncMock,
        ) as mock_fetch,
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        await modal_client.test_endpoint(
            endpoint_url="https://ws--app-serve.modal.run/v1/chat/completions",
            prompt="Hello",
            model_id="explicit/model-name",
        )

    mock_fetch.assert_not_called()
    call_json = mock_client.post.call_args[1]["json"]
    assert call_json["model"] == "explicit/model-name"


def test_vllm_base_url():
    """_vllm_base_url strips the chat completions path."""
    assert (
        ModalDeploymentClient._vllm_base_url(
            "https://ws--app-serve.modal.run/v1/chat/completions"
        )
        == "https://ws--app-serve.modal.run"
    )
    # No suffix → unchanged
    assert (
        ModalDeploymentClient._vllm_base_url("https://ws--app-serve.modal.run")
        == "https://ws--app-serve.modal.run"
    )
