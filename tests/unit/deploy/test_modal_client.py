"""Unit tests for Modal deployment client."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.deploy.base_client import (
    AutoscalingConfig,
    EndpointState,
    HardwareConfig,
    ModelType,
)
from oumi.deploy.modal_client import ModalDeploymentClient


@pytest.fixture
def modal_client():
    """Create Modal client with mock credentials."""
    with patch.dict(
        "os.environ",
        {
            "MODAL_TOKEN_ID": "test-token-id",
            "MODAL_TOKEN_SECRET": "test-token-secret",
            "MODAL_WORKSPACE": "test-workspace",
        },
    ):
        client = ModalDeploymentClient(
            token_id="test-token-id",
            token_secret="test-token-secret",
            workspace="test-workspace",
        )
        # Mock the S3 client
        client._s3_client = MagicMock()
        return client


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = MagicMock()
    client.list_objects_v2.return_value = {"Contents": [{"Key": "model/file"}]}
    return client


def test_gpu_type_mapping(modal_client):
    """Test GPU type conversion to Modal format."""
    hw = HardwareConfig(accelerator="nvidia_a100_80gb", count=1)
    assert modal_client._to_modal_gpu(hw) == "A100"

    hw = HardwareConfig(accelerator="nvidia_h100_80gb", count=2)
    assert modal_client._to_modal_gpu(hw) == "H100"

    # Invalid GPU type
    hw = HardwareConfig(accelerator="nvidia_v100", count=1)
    with pytest.raises(ValueError, match="Unsupported GPU type"):
        modal_client._to_modal_gpu(hw)


def test_app_name_generation(modal_client):
    """Test Modal app name generation from display name."""
    name = modal_client._generate_app_name("My Test Model")
    assert name.startswith("my-test-model-")
    assert name.islower()
    assert " " not in name

    # Special characters removed
    name = modal_client._generate_app_name("Model@123_test")
    assert "@" not in name
    assert "_" not in name


@pytest.mark.asyncio
async def test_upload_model_s3_path(modal_client, mock_s3_client):
    """Test upload_model with S3 path."""
    modal_client._s3_client = mock_s3_client

    result = await modal_client.upload_model(
        model_source="s3://my-bucket/models/llama-7b",
        model_name="llama-7b",
        model_type=ModelType.FULL,
    )

    assert result.provider_model_id == "s3://my-bucket/models/llama-7b"
    assert result.status == "ready"

    # Verify S3 was checked
    mock_s3_client.list_objects_v2.assert_called_once_with(
        Bucket="my-bucket", Prefix="models/llama-7b", MaxKeys=1
    )


@pytest.mark.asyncio
async def test_upload_model_invalid_path(modal_client):
    """Test upload_model rejects non-S3 paths."""
    with pytest.raises(ValueError, match="Modal requires S3 paths"):
        await modal_client.upload_model(
            model_source="/local/path/to/model",
            model_name="test",
        )


@pytest.mark.asyncio
async def test_upload_model_s3_not_found(modal_client, mock_s3_client):
    """Test upload_model fails when S3 path doesn't exist."""
    mock_s3_client.list_objects_v2.return_value = {}  # No contents
    modal_client._s3_client = mock_s3_client

    with pytest.raises(ValueError, match="S3 path not found"):
        await modal_client.upload_model(
            model_source="s3://bucket/missing/model",
            model_name="test",
        )


@pytest.mark.asyncio
async def test_get_model_status(modal_client):
    """Test get_model_status always returns ready for S3 paths."""
    status = await modal_client.get_model_status("s3://bucket/path/to/model")
    assert status == "ready"


@pytest.mark.asyncio
async def test_create_endpoint(modal_client, mock_s3_client):
    """Test endpoint creation with mocked Modal SDK."""
    modal_client._s3_client = mock_s3_client

    with patch("oumi.deploy.modal_client.modal") as mock_modal:
        # Mock Modal App
        mock_app = MagicMock()
        mock_app.__enter__ = MagicMock(return_value=mock_app)
        mock_app.__exit__ = MagicMock(return_value=None)
        mock_modal.App.return_value = mock_app

        # Mock Image
        mock_image = MagicMock()
        mock_image.pip_install.return_value = mock_image
        mock_modal.Image.debian_slim.return_value = mock_image

        # Mock Secret
        mock_secret = MagicMock()
        mock_modal.Secret.from_name.return_value = mock_secret

        # Mock CloudBucketMount
        mock_mount = MagicMock()
        mock_modal.CloudBucketMount.return_value = mock_mount

        # Mock GPU
        mock_gpu = MagicMock()
        mock_modal.gpu.A100.return_value = mock_gpu

        endpoint = await modal_client.create_endpoint(
            model_id="s3://bucket/models/llama-7b",
            hardware=HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
            autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=2),
            display_name="test-model",
        )

        assert endpoint.provider == modal_client.provider
        assert endpoint.state == EndpointState.RUNNING
        assert "test-workspace" in endpoint.endpoint_url
        assert ".modal.run" in endpoint.endpoint_url


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
    """Test getting endpoint status."""
    with patch("oumi.deploy.modal_client.modal") as mock_modal:
        # Mock App.lookup
        mock_app = MagicMock()
        mock_modal.App.lookup.return_value = mock_app

        endpoint = await modal_client.get_endpoint("test-app-123")

        assert endpoint.endpoint_id == "test-app-123"
        assert endpoint.state == EndpointState.RUNNING
        mock_modal.App.lookup.assert_called_once_with("test-app-123", create_if_missing=False)


@pytest.mark.asyncio
async def test_get_endpoint_not_found(modal_client):
    """Test getting endpoint that doesn't exist."""
    with patch("oumi.deploy.modal_client.modal") as mock_modal:
        # Mock App.lookup returns None
        mock_modal.App.lookup.return_value = None

        endpoint = await modal_client.get_endpoint("missing-app")

        assert endpoint.endpoint_id == "missing-app"
        assert endpoint.state == EndpointState.ERROR


@pytest.mark.asyncio
async def test_delete_endpoint(modal_client):
    """Test deleting an endpoint."""
    with patch("oumi.deploy.modal_client.modal") as mock_modal:
        mock_app = MagicMock()
        mock_modal.App.lookup.return_value = mock_app

        # Add app to tracking
        modal_client._deployed_apps["test-app"] = (mock_app, "test-app")

        await modal_client.delete_endpoint("test-app")

        # Verify app was removed from tracking
        assert "test-app" not in modal_client._deployed_apps


@pytest.mark.asyncio
async def test_list_endpoints(modal_client):
    """Test listing endpoints."""
    # Add some tracked apps
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    modal_client._deployed_apps["app1"] = (mock_app1, "app1")
    modal_client._deployed_apps["app2"] = (mock_app2, "app2")

    endpoints = await modal_client.list_endpoints()

    assert len(endpoints) == 2
    assert all(e.provider == modal_client.provider for e in endpoints)
