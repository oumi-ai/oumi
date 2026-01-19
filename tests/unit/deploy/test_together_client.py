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

"""Unit tests for Together.ai deployment client."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    EndpointState,
    HardwareConfig,
    ModelType,
)
from oumi.deploy.together_client import (
    TOGETHER_STATE_MAP,
    TogetherDeploymentClient,
)


class TestTogetherStateMap:
    """Tests for Together.ai state mapping."""

    def test_state_mapping_completeness(self):
        """Test that all expected Together states are mapped."""
        expected_states = [
            "PENDING",
            "STARTING",
            "STARTED",
            "RUNNING",
            "STOPPING",
            "STOPPED",
            "ERROR",
            "FAILED",
        ]
        for state in expected_states:
            assert state in TOGETHER_STATE_MAP

    def test_state_mapping_values(self):
        """Test specific state mappings."""
        assert TOGETHER_STATE_MAP["PENDING"] == EndpointState.PENDING
        assert TOGETHER_STATE_MAP["STARTING"] == EndpointState.STARTING
        assert TOGETHER_STATE_MAP["STARTED"] == EndpointState.RUNNING
        assert TOGETHER_STATE_MAP["RUNNING"] == EndpointState.RUNNING
        assert TOGETHER_STATE_MAP["STOPPING"] == EndpointState.STOPPING
        assert TOGETHER_STATE_MAP["STOPPED"] == EndpointState.STOPPED
        assert TOGETHER_STATE_MAP["ERROR"] == EndpointState.ERROR
        assert TOGETHER_STATE_MAP["FAILED"] == EndpointState.ERROR


class TestTogetherDeploymentClient:
    """Tests for TogetherDeploymentClient."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = TogetherDeploymentClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.provider == DeploymentProvider.TOGETHER

    def test_init_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict("os.environ", {"TOGETHER_API_KEY": "env-key"}):
            client = TogetherDeploymentClient()
            assert client.api_key == "env-key"

    def test_init_raises_without_key(self):
        """Test that init raises error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Together API key"):
                TogetherDeploymentClient()

    def test_hardware_conversion_roundtrip(self):
        """Test hardware config conversion to and from Together format."""
        client = TogetherDeploymentClient(api_key="test")
        hw = HardwareConfig(accelerator="nvidia_a100_80gb", count=2)

        together_str = client._to_together_hardware(hw)
        assert together_str == "2x_nvidia_a100_80gb_sxm"

        hw_back = client._from_together_hardware(together_str)
        assert hw_back.accelerator == hw.accelerator
        assert hw_back.count == hw.count

    def test_hardware_conversion_single_gpu(self):
        """Test hardware conversion for single GPU."""
        client = TogetherDeploymentClient(api_key="test")
        hw = HardwareConfig(accelerator="nvidia_h100_80gb", count=1)

        together_str = client._to_together_hardware(hw)
        assert together_str == "1x_nvidia_h100_80gb_sxm"

    def test_hardware_conversion_fallback(self):
        """Test hardware conversion fallback for unknown format."""
        client = TogetherDeploymentClient(api_key="test")

        # Test fallback for format without _sxm
        hw = client._from_together_hardware("unknown_format")
        assert hw.accelerator == "unknown_format"
        assert hw.count == 1

    def test_parse_endpoint(self):
        """Test parsing Together endpoint response."""
        client = TogetherDeploymentClient(api_key="test")

        data = {
            "id": "ep-123",
            "model": "model-456",
            "state": "STARTED",
            "hardware": "2x_nvidia_a100_80gb_sxm",
            "min_replicas": 1,
            "max_replicas": 3,
            "endpoint_url": "https://api.together.xyz/v1/chat/completions",
            "display_name": "My Endpoint",
            "created_at": "2025-01-16T10:00:00Z",
        }

        endpoint = client._parse_endpoint(data)

        assert endpoint.endpoint_id == "ep-123"
        assert endpoint.model_id == "model-456"
        assert endpoint.state == EndpointState.RUNNING
        assert endpoint.hardware.accelerator == "nvidia_a100_80gb"
        assert endpoint.hardware.count == 2
        assert endpoint.autoscaling.min_replicas == 1
        assert endpoint.autoscaling.max_replicas == 3
        assert endpoint.display_name == "My Endpoint"
        assert endpoint.provider == DeploymentProvider.TOGETHER

    @pytest.mark.asyncio
    async def test_upload_model_payload(self):
        """Test upload_model constructs correct payload."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_id": "uploaded-model-123",
            "job_id": "job-456",
            "status": "pending",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            result = await client.upload_model(
                model_source="hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_name="test-model",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/models"
            assert call_args[1]["json"]["model_source"].startswith("hf://")
            assert call_args[1]["json"]["model_name"] == "test-model"

            assert result.provider_model_id == "uploaded-model-123"
            assert result.job_id == "job-456"
            assert result.request_payload is not None
            assert result.request_payload["model_source"].startswith("hf://")
            assert result.request_payload["model_name"] == "test-model"

    @pytest.mark.asyncio
    async def test_upload_model_with_lora(self):
        """Test upload_model with LoRA adapter."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model_id": "lora-model-123"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            await client.upload_model(
                model_source="s3://bucket/model",
                model_name="my-lora",
                model_type=ModelType.ADAPTER,
                base_model="meta-llama/Llama-3-8B",
            )

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["model_type"] == "lora"
            assert payload["base_model"] == "meta-llama/Llama-3-8B"

    @pytest.mark.asyncio
    async def test_create_endpoint_payload(self):
        """Test create_endpoint constructs correct payload."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "ep-123",
            "model": "model-456",
            "state": "PENDING",
            "hardware": "1x_nvidia_a100_80gb_sxm",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            result = await client.create_endpoint(
                model_id="model-456",
                hardware=HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
                autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=2),
                display_name="test-endpoint",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/endpoints"
            payload = call_args[1]["json"]
            assert payload["model"] == "model-456"
            assert payload["hardware"] == "1x_nvidia_a100_80gb_sxm"
            assert payload["min_replicas"] == 1
            assert payload["max_replicas"] == 2
            assert payload["display_name"] == "test-endpoint"

    @pytest.mark.asyncio
    async def test_get_endpoint(self):
        """Test get_endpoint fetches and parses correctly."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "ep-123",
            "model": "model-456",
            "state": "STARTED",
            "hardware": "1x_nvidia_a100_80gb_sxm",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ) as mock_get:
            result = await client.get_endpoint("ep-123")

            mock_get.assert_called_once_with("/endpoints/ep-123")
            assert result.endpoint_id == "ep-123"
            assert result.state == EndpointState.RUNNING

    @pytest.mark.asyncio
    async def test_delete_endpoint(self):
        """Test delete_endpoint calls correct endpoint."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "delete", new_callable=AsyncMock, return_value=mock_response
        ) as mock_delete:
            await client.delete_endpoint("ep-123")

            mock_delete.assert_called_once_with("/endpoints/ep-123")

    @pytest.mark.asyncio
    async def test_list_endpoints(self):
        """Test list_endpoints fetches and parses list."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "ep-1", "model": "m1", "state": "STARTED", "hardware": "1x_gpu_sxm"},
            {"id": "ep-2", "model": "m2", "state": "STOPPED", "hardware": "2x_gpu_sxm"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.list_endpoints()

            assert len(result) == 2
            assert result[0].endpoint_id == "ep-1"
            assert result[1].endpoint_id == "ep-2"

    @pytest.mark.asyncio
    async def test_list_hardware(self):
        """Test list_hardware fetches and parses hardware list."""
        client = TogetherDeploymentClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = [
            "1x_nvidia_a100_80gb_sxm",
            "2x_nvidia_h100_80gb_sxm",
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.list_hardware()

            assert len(result) == 2
            assert result[0].accelerator == "nvidia_a100_80gb"
            assert result[0].count == 1
            assert result[1].accelerator == "nvidia_h100_80gb"
            assert result[1].count == 2

    @pytest.mark.asyncio
    async def test_upload_model_already_in_progress_found(self):
        """Test upload_model handles 'upload already in progress' by finding existing model."""
        client = TogetherDeploymentClient(api_key="test")

        # Mock the upload response with 400 error
        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 400
        mock_upload_response.text = (
            '{"error":{"message":"Model upload already in progress"}}'
        )
        mock_upload_response.json.return_value = {
            "error": {"message": "Model upload already in progress"}
        }

        # Mock list_models to return the existing model
        from oumi.deploy.base_client import Model

        existing_model = Model(
            model_id="existing-model-123",
            model_name="test-model",
            status="pending",
            provider=DeploymentProvider.TOGETHER,
        )

        with (
            patch.object(
                client._client,
                "post",
                new_callable=AsyncMock,
                return_value=mock_upload_response,
            ),
            patch.object(
                client,
                "list_models",
                new_callable=AsyncMock,
                return_value=[existing_model],
            ),
        ):
            result = await client.upload_model(
                model_source="s3://bucket/model",
                model_name="test-model",
            )

            # Should return the existing model
            assert result.provider_model_id == "existing-model-123"
            assert result.status == "pending"
            assert result.job_id is None
            # Should include the request payload that was attempted
            assert result.request_payload is not None
            assert result.request_payload["model_name"] == "test-model"

    @pytest.mark.asyncio
    async def test_upload_model_already_in_progress_not_found(self):
        """Test upload_model handles 'upload already in progress' when model not yet visible."""
        client = TogetherDeploymentClient(api_key="test")

        # Mock the upload response with 400 error
        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 400
        mock_upload_response.text = (
            '{"error":{"message":"Model upload already in progress"}}'
        )
        mock_upload_response.json.return_value = {
            "error": {"message": "Model upload already in progress"}
        }

        # Mock list_models to return empty list (model not yet visible)
        with (
            patch.object(
                client._client,
                "post",
                new_callable=AsyncMock,
                return_value=mock_upload_response,
            ),
            patch.object(
                client, "list_models", new_callable=AsyncMock, return_value=[]
            ),
        ):
            result = await client.upload_model(
                model_source="s3://bucket/model",
                model_name="test-model",
            )

            # Should return expected model name for polling
            assert result.provider_model_id == "test-model"
            assert result.status == "pending"
            assert result.job_id is None
            # Should include the request payload that was attempted
            assert result.request_payload is not None
            assert result.request_payload["model_name"] == "test-model"
