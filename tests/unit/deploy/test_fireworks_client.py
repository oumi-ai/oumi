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

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    EndpointState,
    HardwareConfig,
    ModelType,
)
from oumi.deploy.fireworks_api import (
    FW_STATE_TO_ENDPOINT,
    GatewayAcceleratorType,
    GatewayDeployment,
    GatewayDeploymentState,
)
from oumi.deploy.fireworks_client import (
    FIREWORKS_ACCELERATORS,
    FireworksDeploymentClient,
    FireworksInvalidModelIdError,
    _raise_api_error,
    _validate_fireworks_model_id,
)


class TestFireworksStateMap:
    """Tests for the GatewayDeploymentState → EndpointState mapping."""

    def test_state_mapping_completeness(self):
        """Every GatewayDeploymentState member has an EndpointState mapping."""
        for state in GatewayDeploymentState:
            assert state in FW_STATE_TO_ENDPOINT

    def test_state_mapping_values(self):
        """Test specific state mappings."""
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.STATE_UNSPECIFIED]
            == EndpointState.PENDING
        )
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.CREATING]
            == EndpointState.STARTING
        )
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.READY] == EndpointState.RUNNING
        )
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.DELETING]
            == EndpointState.STOPPING
        )
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.DELETED]
            == EndpointState.STOPPED
        )
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.FAILED] == EndpointState.ERROR
        )
        assert (
            FW_STATE_TO_ENDPOINT[GatewayDeploymentState.UPDATING]
            == EndpointState.STARTING
        )


class TestFireworksAccelerators:
    """Tests for Fireworks accelerator mappings."""

    def test_accelerator_mapping_completeness(self):
        """Every GatewayAcceleratorType member (except UNSPECIFIED) has a mapping."""
        expected = {
            member.value
            for member in GatewayAcceleratorType
            if member != GatewayAcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED
        }
        actual = set(FIREWORKS_ACCELERATORS.values())
        assert actual == expected

    def test_accelerator_mapping_keys_are_lowercase(self):
        """All Oumi-standard keys are lowercase versions of the API values."""
        for key, value in FIREWORKS_ACCELERATORS.items():
            assert key == key.lower()
            assert value == value.upper()


class TestFireworksDeploymentClient:
    """Tests for FireworksDeploymentClient."""

    def test_init_with_credentials(self):
        """Test client initialization with credentials."""
        client = FireworksDeploymentClient(
            api_key="test-key", account_id="test-account"
        )
        assert client._api_key == "test-key"
        assert client.account_id == "test-account"
        assert client.provider == DeploymentProvider.FIREWORKS

    def test_init_from_env(self):
        """Test client initialization from environment variables."""
        with patch.dict(
            "os.environ",
            {"FIREWORKS_API_KEY": "env-key", "FIREWORKS_ACCOUNT_ID": "env-account"},
        ):
            client = FireworksDeploymentClient()
            assert client._api_key == "env-key"
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
        """Test parsing a GatewayDeployment into an Endpoint."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        deployment = GatewayDeployment(
            name="accounts/test-account/deployments/deploy-123",
            baseModel="accounts/test-account/models/model-456",
            state=GatewayDeploymentState.READY,
            acceleratorType=GatewayAcceleratorType.NVIDIA_A100_80GB,
            acceleratorCount=2,
            minReplicaCount=1,
            maxReplicaCount=3,
            displayName="My Deployment",
            createTime=datetime(2025, 1, 16, 10, 0, 0, tzinfo=timezone.utc),
        )

        endpoint = client._parse_deployment(deployment)

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
    async def test_create_endpoint_payload(self):
        """Test create_endpoint constructs correct payload."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "name": "accounts/test-account/deployments/deploy-123",
            "baseModel": "model-456",
            "state": "CREATING",
            "acceleratorType": "NVIDIA_A100_80GB",
            "acceleratorCount": 1,
            "minReplicaCount": 1,
            "maxReplicaCount": 2,
        }

        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            await client.create_endpoint(
                model_id="model-456",
                hardware=HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
                autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=2),
                display_name="test-deployment",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/deployments" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["baseModel"] == "model-456"
            assert payload["acceleratorType"] == "NVIDIA_A100_80GB"
            assert payload["minReplicaCount"] == 1
            assert payload["maxReplicaCount"] == 2
            assert payload["displayName"] == "test-deployment"

    @pytest.mark.asyncio
    async def test_get_endpoint(self):
        """Test get_endpoint fetches and parses correctly."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "name": "accounts/test-account/deployments/deploy-123",
            "baseModel": "model-456",
            "state": "READY",
            "acceleratorType": "NVIDIA_A100_80GB",
            "acceleratorCount": 1,
        }

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
        mock_response.is_success = True

        with patch.object(
            client._client, "delete", new_callable=AsyncMock, return_value=mock_response
        ) as mock_delete:
            await client.delete_endpoint("deploy-123")

            assert "/deployments/deploy-123" in mock_delete.call_args[0][0]
            assert mock_delete.call_args[1].get("params") == {}

    @pytest.mark.asyncio
    async def test_delete_endpoint_force(self):
        """Test delete_endpoint with force passes ignoreChecks and hard to API."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.object(
            client._client, "delete", new_callable=AsyncMock, return_value=mock_response
        ) as mock_delete:
            await client.delete_endpoint("deploy-123", force=True)

            assert "/deployments/deploy-123" in mock_delete.call_args[0][0]
            assert mock_delete.call_args[1]["params"] == {
                "ignoreChecks": True,
                "hard": True,
            }

    @pytest.mark.asyncio
    async def test_list_endpoints(self):
        """Test list_endpoints fetches and parses list."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "deployments": [
                {
                    "name": "accounts/test/deployments/d1",
                    "baseModel": "m1",
                    "state": "READY",
                    "acceleratorType": "NVIDIA_A100_80GB",
                    "acceleratorCount": 1,
                    "minReplicaCount": 1,
                    "maxReplicaCount": 1,
                },
                {
                    "name": "accounts/test/deployments/d2",
                    "baseModel": "m2",
                    "state": "CREATING",
                    "acceleratorType": "NVIDIA_A100_80GB",
                    "acceleratorCount": 1,
                    "minReplicaCount": 0,
                    "maxReplicaCount": 1,
                },
            ]
        }

        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.list_endpoints()

            assert len(result) == 2
            assert result[0].endpoint_id == "d1"
            assert result[1].endpoint_id == "d2"

    @pytest.mark.asyncio
    async def test_list_hardware(self):
        """list_hardware returns one entry per FIREWORKS_ACCELERATORS key."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")

        result = await client.list_hardware()

        assert len(result) == len(FIREWORKS_ACCELERATORS)
        accelerators = {hw.accelerator for hw in result}
        assert accelerators == set(FIREWORKS_ACCELERATORS.keys())
        for hw in result:
            assert hw.count == 1

    @pytest.mark.asyncio
    async def test_start_endpoint_not_supported(self):
        """Fireworks does not support start_endpoint; raises NotImplementedError."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")
        with pytest.raises(NotImplementedError, match="fireworks.*start/stop"):
            await client.start_endpoint("some-id")

    @pytest.mark.asyncio
    async def test_stop_endpoint_not_supported(self):
        """Fireworks does not support stop_endpoint; raises NotImplementedError."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")
        with pytest.raises(NotImplementedError, match="fireworks.*start/stop"):
            await client.stop_endpoint("some-id")

    def test_get_inference_auth_headers(self):
        """Fireworks returns Bearer auth for inference."""
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")
        headers = client._get_inference_auth_headers()
        assert headers == {"Authorization": "Bearer test"}


class TestValidateFireworksModelId:
    """Tests for _validate_fireworks_model_id."""

    def test_valid_model_ids(self):
        for mid in ("my-model", "abc", "a" * 63, "model-v1-beta"):
            _validate_fireworks_model_id(mid)

    def test_empty_string(self):
        with pytest.raises(FireworksInvalidModelIdError, match="empty string"):
            _validate_fireworks_model_id("")

    def test_too_long(self):
        with pytest.raises(FireworksInvalidModelIdError, match="at most 63"):
            _validate_fireworks_model_id("a" * 64)

    def test_invalid_chars_uppercase(self):
        with pytest.raises(FireworksInvalidModelIdError, match="invalid characters"):
            _validate_fireworks_model_id("My-Model")

    def test_invalid_chars_underscore(self):
        with pytest.raises(FireworksInvalidModelIdError, match="invalid characters"):
            _validate_fireworks_model_id("my_model")

    def test_starts_with_hyphen(self):
        with pytest.raises(FireworksInvalidModelIdError, match="begin with a hyphen"):
            _validate_fireworks_model_id("-my-model")

    def test_ends_with_hyphen(self):
        with pytest.raises(FireworksInvalidModelIdError, match="end with a hyphen"):
            _validate_fireworks_model_id("my-model-")

    def test_starts_with_digit(self):
        with pytest.raises(FireworksInvalidModelIdError, match="begin with a digit"):
            _validate_fireworks_model_id("1-model")


class TestRaiseApiError:
    """Tests for _raise_api_error."""

    @staticmethod
    def _make_response(
        status_code: int, json_body: dict | None = None, text: str = ""
    ) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.text = text
        if json_body is not None:
            resp.json.return_value = json_body
        else:
            resp.json.side_effect = Exception("no json")
        req = MagicMock()
        req.method = "POST"
        req.url = "https://api.fireworks.ai/v1/test"
        req.content = b'{"key": "value"}'
        resp.request = req
        return resp

    def test_extracts_nested_error_message(self):
        resp = self._make_response(
            400, {"error": {"message": "bad request", "code": "INVALID_ARGUMENT"}}
        )
        with pytest.raises(ValueError, match="bad request"):
            _raise_api_error(resp, "create model")

    def test_extracts_top_level_message(self):
        resp = self._make_response(404, {"message": "not found"})
        with pytest.raises(ValueError, match="not found"):
            _raise_api_error(resp, "get model")

    def test_falls_back_to_text(self):
        resp = self._make_response(500, json_body=None, text="internal server error")
        with pytest.raises(ValueError, match="internal server error"):
            _raise_api_error(resp, "delete model")

    def test_does_not_include_request_body(self):
        resp = self._make_response(400, {"message": "bad"})
        with pytest.raises(ValueError) as exc_info:
            _raise_api_error(resp, "test")
        assert "request body" not in str(exc_info.value)

    def test_includes_http_status_and_method(self):
        resp = self._make_response(409, {"message": "conflict"})
        with pytest.raises(ValueError, match=r"HTTP 409.*POST"):
            _raise_api_error(resp, "create")


class TestCheckModelSourceSupported:
    """Tests for FireworksDeploymentClient._check_model_source_supported."""

    def test_rejects_s3(self):
        with pytest.raises(ValueError, match="S3"):
            FireworksDeploymentClient._check_model_source_supported("s3://bucket/model")

    def test_rejects_gcs(self):
        with pytest.raises(ValueError, match="GCS"):
            FireworksDeploymentClient._check_model_source_supported("gs://bucket/model")

    def test_rejects_azure(self):
        with pytest.raises(ValueError, match="Azure"):
            FireworksDeploymentClient._check_model_source_supported(
                "az://container/model"
            )

    def test_rejects_huggingface_url(self):
        with pytest.raises(ValueError, match="HuggingFace"):
            FireworksDeploymentClient._check_model_source_supported(
                "https://huggingface.co/meta-llama/Llama-2-7b-hf"
            )

    def test_rejects_generic_url(self):
        with pytest.raises(ValueError, match="remote URLs"):
            FireworksDeploymentClient._check_model_source_supported(
                "https://example.com/model.tar.gz"
            )

    def test_rejects_hf_repo_id(self):
        with pytest.raises(ValueError, match="HuggingFace repo ID"):
            FireworksDeploymentClient._check_model_source_supported(
                "meta-llama/Llama-3-8B"
            )

    def test_rejects_nonexistent_path(self):
        with pytest.raises(ValueError, match="does not exist"):
            FireworksDeploymentClient._check_model_source_supported(
                "/nonexistent/path/model"
            )

    def test_rejects_file_not_dir(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_text("data")
        with pytest.raises(ValueError, match="file, not a directory"):
            FireworksDeploymentClient._check_model_source_supported(str(f))

    def test_accepts_valid_local_dir(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        FireworksDeploymentClient._check_model_source_supported(str(model_dir))


class TestVerifyBaseModelExists:
    """Tests for FireworksDeploymentClient._verify_base_model_exists."""

    @pytest.mark.asyncio
    async def test_passes_when_model_ready(self):
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "name": "accounts/oumi/models/my-base",
            "state": "READY",
        }
        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            await client._verify_base_model_exists("accounts/oumi/models/my-base")

    @pytest.mark.asyncio
    async def test_raises_when_model_not_found(self):
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.is_success = False
        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ValueError, match="not found on Fireworks"):
                await client._verify_base_model_exists(
                    "accounts/oumi/models/nonexistent"
                )

    @pytest.mark.asyncio
    async def test_raises_when_model_not_ready(self):
        client = FireworksDeploymentClient(api_key="test", account_id="test-account")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "name": "accounts/oumi/models/my-base",
            "state": "UPLOADING",
        }
        with patch.object(
            client._client, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(ValueError, match="not ready.*state: uploading"):
                await client._verify_base_model_exists("accounts/oumi/models/my-base")


class TestReadAdapterConfig:
    """Tests for FireworksDeploymentClient._read_adapter_config."""

    def test_reads_valid_config(self, tmp_path):
        config = {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "peft_type": "LORA",
        }
        import json

        (tmp_path / "adapter_config.json").write_text(json.dumps(config))
        result = FireworksDeploymentClient._read_adapter_config(tmp_path)
        assert result is not None
        assert result["r"] == 16
        assert result["target_modules"] == ["q_proj", "k_proj", "v_proj"]

    def test_returns_none_when_missing(self, tmp_path):
        result = FireworksDeploymentClient._read_adapter_config(tmp_path)
        assert result is None

    def test_raises_on_invalid_json(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Failed to read adapter config"):
            FireworksDeploymentClient._read_adapter_config(tmp_path)


class TestCollectFileInventory:
    """Tests for FireworksDeploymentClient._collect_file_inventory."""

    def _make_client(self):
        return FireworksDeploymentClient(api_key="test", account_id="test")

    def test_basic_inventory(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 100)
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert "config.json" in inventory
        assert "model.safetensors" in inventory
        assert inventory["model.safetensors"] == 100

    def test_skips_lock_files(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors.lock").write_text("")
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert "config.json" in inventory
        assert "model.safetensors.lock" not in inventory

    def test_skips_metadata_files(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "something.metadata").write_text("")
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert "something.metadata" not in inventory

    def test_skips_training_state(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
            (tmp_path / name).write_text("x")
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert "optimizer.pt" not in inventory
        assert "scheduler.pt" not in inventory
        assert "trainer_state.json" not in inventory

    def test_skips_rng_state(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "rng_state_0.pth").write_text("x")
        (tmp_path / "rng_state.pth").write_text("x")
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert "rng_state_0.pth" not in inventory
        assert "rng_state.pth" not in inventory

    def test_skips_cache_dirs(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        (cache_dir / "cached_file.bin").write_text("x")
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert ".cache/cached_file.bin" not in inventory
        assert "config.json" in inventory

    def test_handles_nested_dirs(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "weights.bin").write_bytes(b"\x00" * 50)
        inventory = self._make_client()._collect_file_inventory(tmp_path)
        assert os.path.join("subdir", "weights.bin") in inventory

    def test_adapter_only_includes_adapter_files(self, tmp_path):
        """For ADAPTER uploads, only adapter_model.safetensors and
        adapter_config.json.
        """
        (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00" * 200)
        (tmp_path / "adapter_config.json").write_text('{"r": 16}')
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "vocab.json").write_text("{}")
        (tmp_path / "merges.txt").write_text("")
        (tmp_path / "README.md").write_text("readme")
        (tmp_path / "special_tokens_map.json").write_text("{}")
        (tmp_path / "chat_template.jinja").write_text("")
        inventory = self._make_client()._collect_file_inventory(
            tmp_path, model_type=ModelType.ADAPTER
        )
        assert "adapter_model.safetensors" in inventory
        assert "adapter_config.json" in inventory
        assert len(inventory) == 2

    def test_full_model_includes_all_files(self, tmp_path):
        """For FULL uploads, tokenizer and other files are included."""
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 100)
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "vocab.json").write_text("{}")
        inventory = self._make_client()._collect_file_inventory(
            tmp_path, model_type=ModelType.FULL
        )
        assert len(inventory) == 4
