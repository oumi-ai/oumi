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

"""Unit tests for Fireworks.ai deployment client."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    EndpointState,
    HardwareConfig,
)
from oumi.deploy.fireworks_client import (
    FIREWORKS_ACCELERATORS,
    FIREWORKS_STATE_MAP,
    FireworksDeploymentClient,
)


class TestFireworksStateMap:
    """Tests for Fireworks state mapping."""

    def test_state_mapping_completeness(self):
        """Test that all expected Fireworks states are mapped."""
        expected_states = [
            "PENDING",
            "CREATING",
            "READY",
            "RUNNING",
            "DELETING",
            "DELETED",
            "FAILED",
            "ERROR",
        ]
        for state in expected_states:
            assert state in FIREWORKS_STATE_MAP

    def test_state_mapping_values(self):
        """Test specific state mappings."""
        assert FIREWORKS_STATE_MAP["PENDING"] == EndpointState.PENDING
        assert FIREWORKS_STATE_MAP["CREATING"] == EndpointState.STARTING
        assert FIREWORKS_STATE_MAP["READY"] == EndpointState.RUNNING
        assert FIREWORKS_STATE_MAP["RUNNING"] == EndpointState.RUNNING
        assert FIREWORKS_STATE_MAP["DELETING"] == EndpointState.STOPPING
        assert FIREWORKS_STATE_MAP["DELETED"] == EndpointState.STOPPED
        assert FIREWORKS_STATE_MAP["FAILED"] == EndpointState.ERROR
        assert FIREWORKS_STATE_MAP["ERROR"] == EndpointState.ERROR


class TestFireworksAccelerators:
    """Tests for Fireworks accelerator mappings."""

    def test_accelerator_mapping(self):
        """Test accelerator name mappings."""
        assert FIREWORKS_ACCELERATORS["nvidia_a100_80gb"] == "NVIDIA_A100_80GB"
        assert FIREWORKS_ACCELERATORS["nvidia_h100_80gb"] == "NVIDIA_H100_80GB"
        assert FIREWORKS_ACCELERATORS["nvidia_h200_141gb"] == "NVIDIA_H200_141GB"
        assert FIREWORKS_ACCELERATORS["amd_mi300x"] == "AMD_MI300X"


class TestFireworksDeploymentClient:
    """Tests for FireworksDeploymentClient."""

    def test_init_with_credentials(self):
        """Test client initialization with credentials."""
        client = FireworksDeploymentClient(
            api_key="test-key", account_id="test-account"
        )
        assert client.api_key == "test-key"
        assert client.account_id == "test-account"
        assert client.provider == DeploymentProvider.FIREWORKS

    def test_init_from_env(self):
        """Test client initialization from environment variables."""
        with patch.dict(
            "os.environ",
            {"FIREWORKS_API_KEY": "env-key", "FIREWORKS_ACCOUNT_ID": "env-account"},
        ):
            client = FireworksDeploymentClient()
            assert client.api_key == "env-key"
            assert client.account_id == "env-account"

    def test_init_raises_without_api_key(self):
        """Test that init raises error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Fireworks API key"):
                FireworksDeploymentClient()

    def test_init_raises_without_account_id(self):
        """Test that init raises error without account ID."""
        with patch.dict("os.environ", {"FIREWORKS_API_KEY": "key"}, clear=True):
            with pytest.raises(ValueError, match="Fireworks account ID"):
                FireworksDeploymentClient()

    def test_accelerator_conversion(self):
        """Test accelerator conversion to Fireworks format."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        hw = HardwareConfig(accelerator="nvidia_h100_80gb", count=2)
        result = client._to_fireworks_accelerator(hw)
        assert result == "NVIDIA_H100_80GB"

    def test_accelerator_conversion_unknown(self):
        """Test accelerator conversion for unknown accelerator."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        hw = HardwareConfig(accelerator="unknown_gpu", count=1)
        result = client._to_fireworks_accelerator(hw)
        assert result == "UNKNOWN_GPU"

    def test_accelerator_reverse_conversion(self):
        """Test accelerator conversion from Fireworks format."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        result = client._from_fireworks_accelerator("NVIDIA_A100_80GB")
        assert result == "nvidia_a100_80gb"

    def test_accelerator_reverse_conversion_unknown(self):
        """Test accelerator conversion for unknown Fireworks accelerator."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        result = client._from_fireworks_accelerator("UNKNOWN_GPU")
        assert result == "unknown_gpu"

    def test_parse_deployment(self):
        """Test parsing Fireworks deployment response."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        data = {
            "name": "accounts/test-account/deployments/deploy-123",
            "model": "accounts/test-account/models/model-456",
            "state": "READY",
            "config": {
                "acceleratorType": "NVIDIA_A100_80GB",
                "acceleratorCount": 2,
                "minReplicas": 1,
                "maxReplicas": 3,
            },
            "endpointUrl": "https://api.fireworks.ai/v1/chat/completions",
            "displayName": "My Deployment",
            "createTime": "2025-01-16T10:00:00Z",
        }

        endpoint = client._parse_deployment(data)

        assert endpoint.endpoint_id == "deploy-123"
        assert endpoint.model_id == "accounts/test-account/models/model-456"
        assert endpoint.state == EndpointState.RUNNING
        assert endpoint.hardware.accelerator == "nvidia_a100_80gb"
        assert endpoint.hardware.count == 2
        assert endpoint.autoscaling.min_replicas == 1
        assert endpoint.autoscaling.max_replicas == 3
        assert endpoint.display_name == "My Deployment"
        assert endpoint.provider == DeploymentProvider.FIREWORKS

    @pytest.mark.asyncio
    async def test_package_model(self, tmp_path):
        """Test model packaging creates valid tarball."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        # Create a mock model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"fake_model_data")

        tar_path = await client._package_model(str(model_dir))

        assert tar_path.exists()
        assert tar_path.suffix == ".gz"

        # Clean up
        tar_path.unlink()

    @pytest.mark.asyncio
    async def test_create_endpoint_payload(self):
        """Test create_endpoint constructs correct payload."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "accounts/test-account/deployments/deploy-123",
            "model": "model-456",
            "state": "PENDING",
            "config": {
                "acceleratorType": "NVIDIA_A100_80GB",
                "acceleratorCount": 1,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            result = await client.create_endpoint(
                model_id="model-456",
                hardware=HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
                autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=2),
                display_name="test-deployment",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/deployments" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["model"] == "model-456"
            assert payload["config"]["acceleratorType"] == "NVIDIA_A100_80GB"
            assert payload["config"]["minReplicas"] == 1
            assert payload["config"]["maxReplicas"] == 2
            assert payload["displayName"] == "test-deployment"

    @pytest.mark.asyncio
    async def test_get_endpoint(self):
        """Test get_endpoint fetches and parses correctly."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "accounts/test-account/deployments/deploy-123",
            "model": "model-456",
            "state": "READY",
            "config": {
                "acceleratorType": "NVIDIA_A100_80GB",
                "acceleratorCount": 1,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ) as mock_get:
            result = await client.get_endpoint("deploy-123")

            assert "/deployments/deploy-123" in mock_get.call_args[0][0]
            assert result.endpoint_id == "deploy-123"
            assert result.state == EndpointState.RUNNING

    @pytest.mark.asyncio
    async def test_delete_endpoint(self):
        """Test delete_endpoint calls correct endpoint."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "delete", new_callable=AsyncMock, return_value=mock_response
        ) as mock_delete:
            await client.delete_endpoint("deploy-123")

            assert "/deployments/deploy-123" in mock_delete.call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_endpoints(self):
        """Test list_endpoints fetches and parses list."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "deployments": [
                {
                    "name": "accounts/test/deployments/d1",
                    "model": "m1",
                    "state": "READY",
                    "config": {},
                },
                {
                    "name": "accounts/test/deployments/d2",
                    "model": "m2",
                    "state": "PENDING",
                    "config": {},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.list_endpoints()

            assert len(result) == 2
            assert result[0].endpoint_id == "d1"
            assert result[1].endpoint_id == "d2"

    @pytest.mark.asyncio
    async def test_list_hardware(self):
        """Test list_hardware returns hardcoded configs."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        result = await client.list_hardware()

        assert len(result) == 4
        accelerators = [hw.accelerator for hw in result]
        assert "nvidia_a100_80gb" in accelerators
        assert "nvidia_h100_80gb" in accelerators
        assert "nvidia_h200_141gb" in accelerators
        assert "amd_mi300x" in accelerators
