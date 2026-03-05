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

"""Unit tests for Parasail deployment client."""

import logging
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import oumi.deploy.parasail_client as parasail_client_module
from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    Endpoint,
    EndpointState,
    HardwareConfig,
    ModelType,
)
from oumi.deploy.parasail_api import (
    DeploymentResponse,
    DeviceConfig,
    ParasailScaleDownPolicy,
)
from oumi.deploy.parasail_client import (
    ParasailDeploymentClient,
    _to_deployment_name,
    _to_endpoint,
    _validate_endpoint_id,
    _validate_model_source,
)


def _deployment_payload(
    *,
    deployment_id: int = 33665,
    model_name: str = "Qwen/Qwen3-4B",
    status: str = "ONLINE",
    replicas: int = 1,
    external_alias: str | None = None,
) -> dict:
    return {
        "id": deployment_id,
        "deploymentName": "qwen-qwen3-4b",
        "displayName": "Qwen Deploy",
        "externalAlias": external_alias,
        "modelName": model_name,
        "replicas": replicas,
        "status": {"id": deployment_id, "status": status, "instances": []},
        "deviceConfigs": [
            {
                "device": "NVIDIA_H100",
                "count": 1,
                "displayName": "H100",
                "available": True,
                "selected": True,
            }
        ],
        "createdAt": 1741219200000,
    }


class TestValidationHelpers:
    def test_validate_model_source_accepts_hf_repo_and_url(self):
        _validate_model_source("Qwen/Qwen3-4B")
        _validate_model_source("https://huggingface.co/Qwen/Qwen3-4B")

    def test_validate_model_source_rejects_local_path(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(ValueError, match="local model uploads"):
            _validate_model_source(str(model_dir))

    def test_validate_model_source_rejects_s3_and_gcs(self):
        with pytest.raises(ValueError, match="does not support S3"):
            _validate_model_source("s3://bucket/model")
        with pytest.raises(ValueError, match="does not support GCS"):
            _validate_model_source("gs://bucket/model")

    def test_validate_model_source_rejects_serverless_alias(self):
        with pytest.raises(NotImplementedError, match="serverless inference"):
            _validate_model_source("parasail-deepseek-r1")

    def test_validate_endpoint_id_rejects_hf_identifier(self):
        with pytest.raises(ValueError, match="HuggingFace model identifier"):
            _validate_endpoint_id("Qwen/Qwen3-4B")

    def test_to_deployment_name_normalizes_input(self):
        assert _to_deployment_name("Qwen 3.1 (Prod)") == "qwen-3-1-prod"
        assert _to_deployment_name("!!!") == "oumi-deployment"


class TestParasailDeploymentClient:
    def test_init_prefers_explicit_api_key(self):
        with patch.dict("os.environ", {"PARASAIL_API_KEY": "env-key"}, clear=True):
            client = ParasailDeploymentClient(api_key="explicit-key")
            assert client._api_key == "explicit-key"

    def test_init_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PARASAIL_API_KEY"):
                ParasailDeploymentClient()

    @pytest.mark.asyncio
    async def test_async_context_manager_initializes_client(self):
        with patch.dict("os.environ", {"PARASAIL_API_KEY": "test-key"}, clear=True):
            with patch(
                "oumi.deploy.parasail_client.httpx.AsyncClient"
            ) as mock_async_client:
                mock_http = AsyncMock()
                mock_async_client.return_value = mock_http

                client = ParasailDeploymentClient()
                async with client:
                    assert client._client is mock_http

                mock_http.aclose.assert_awaited_once()

    def test_client_property_raises_outside_context_manager(self):
        with patch.dict("os.environ", {"PARASAIL_API_KEY": "test-key"}, clear=True):
            client = ParasailDeploymentClient()
            with pytest.raises(RuntimeError, match="async context manager"):
                _ = client._client

    @pytest.mark.asyncio
    async def test_upload_model_calls_support_and_returns_ready(self):
        client = ParasailDeploymentClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "supported": True,
            "messages": [],
        }

        async with client:
            with (
                patch(
                    "oumi.deploy.parasail_client.resolve_hf_token",
                    return_value="hf-token",
                ),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=mock_response,
                ) as mock_get,
            ):
                uploaded = await client.upload_model(
                    model_source="Qwen/Qwen3-4B",
                    model_name="Qwen3",
                    model_type=ModelType.FULL,
                )

        assert uploaded.provider_model_id == "Qwen/Qwen3-4B"
        assert uploaded.status == "ready"
        assert mock_get.call_args[0][0] == "/dedicated/support"
        assert mock_get.call_args[1]["params"]["modelName"] == "Qwen/Qwen3-4B"
        assert mock_get.call_args[1]["params"]["modelAccessKey"] == "hf-token"

    @pytest.mark.asyncio
    async def test_upload_model_raises_for_error_messages(self):
        client = ParasailDeploymentClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "supported": True,
            "messages": [{"level": "ERROR", "content": "Model not found"}],
        }

        async with client:
            with (
                patch("oumi.deploy.parasail_client.resolve_hf_token", return_value=""),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=mock_response,
                ),
            ):
                with pytest.raises(ValueError, match="Model not found"):
                    await client.upload_model(
                        model_source="Qwen/Unknown",
                        model_name="bad",
                    )

    @pytest.mark.asyncio
    async def test_upload_model_allows_adapter_and_logs_info(self, caplog):
        """Regression guard for Gap P4: ADAPTER must not be rejected."""
        client = ParasailDeploymentClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"supported": True, "messages": []}

        async with client:
            with (
                patch("oumi.deploy.parasail_client.resolve_hf_token", return_value=""),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=mock_response,
                ),
            ):
                with caplog.at_level(
                    logging.INFO, logger="oumi.deploy.parasail_client"
                ):
                    uploaded = await client.upload_model(
                        model_source="idoud/oumi-dl-qwen3-1.7b-lora-modal-test",
                        model_name="adapter",
                        model_type=ModelType.ADAPTER,
                    )

        assert uploaded.status == "ready"
        assert any("LoRA adapter deployment" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_create_endpoint_builds_expected_payload(self):
        client = ParasailDeploymentClient(api_key="test-key")

        support_devices = [
            DeviceConfig(
                device="NVIDIA_H100",
                count=1,
                available=True,
                displayName=None,
                estimatedSingleUserTps=None,
                estimatedSystemTps=None,
                limitedContext=False,
            ),
            DeviceConfig(
                device="NVIDIA_A100",
                count=1,
                available=True,
                displayName=None,
                estimatedSingleUserTps=None,
                estimatedSystemTps=None,
                limitedContext=False,
            ),
        ]
        create_response = MagicMock()
        create_response.raise_for_status = MagicMock()
        create_response.json.return_value = _deployment_payload(
            external_alias="qwen-deploy"
        )

        async with client:
            with (
                patch.object(
                    client,
                    "_get_and_select_device_configs",
                    new_callable=AsyncMock,
                    return_value=support_devices,
                ) as mock_select,
                patch.object(
                    client._client,
                    "post",
                    new_callable=AsyncMock,
                    return_value=create_response,
                ) as mock_post,
                patch(
                    "oumi.deploy.parasail_client.resolve_hf_token",
                    return_value="hf-token",
                ),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
            ):
                endpoint = await client.create_endpoint(
                    model_id="Qwen/Qwen3-4B",
                    hardware=HardwareConfig(accelerator="NVIDIA_H100", count=1),
                    autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=4),
                    display_name="Qwen Deploy",
                    model_access_key="explicit-token",
                    context_length=8192,
                    scale_down_policy=ParasailScaleDownPolicy.TIMER,
                    scale_down_threshold_ms=60000,
                )

        assert endpoint.endpoint_id == "33665"
        assert endpoint.state == EndpointState.RUNNING
        assert endpoint.endpoint_url == "https://api.parasail.io/v1/chat/completions"
        mock_select.assert_awaited_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["deploymentName"] == "qwen-deploy"
        assert payload["modelName"] == "Qwen/Qwen3-4B"
        assert payload["replicas"] == 1
        assert payload["contextLength"] == 8192
        assert payload["scaleDownPolicy"] == "TIMER"
        assert payload["scaleDownThreshold"] == 60000
        assert payload["modelAccessKey"] == "hf-token"

    @pytest.mark.asyncio
    async def test_get_and_delete_endpoint_routes(self):
        client = ParasailDeploymentClient(api_key="test-key")
        get_response = MagicMock()
        get_response.raise_for_status = MagicMock()
        get_response.json.return_value = _deployment_payload()

        delete_response = MagicMock()
        delete_response.raise_for_status = MagicMock()

        async with client:
            with (
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=get_response,
                ) as mock_get,
                patch.object(
                    client._client,
                    "delete",
                    new_callable=AsyncMock,
                    return_value=delete_response,
                ) as mock_delete,
            ):
                endpoint = await client.get_endpoint("33665")
                await client.delete_endpoint("33665")

        assert endpoint.endpoint_id == "33665"
        assert mock_get.call_args[0][0] == "/dedicated/deployments/33665"
        assert mock_delete.call_args[0][0] == "/dedicated/deployments/33665"

    @pytest.mark.asyncio
    async def test_start_and_stop_endpoint_call_resume_pause_then_get(self):
        client = ParasailDeploymentClient(api_key="test-key")
        post_response = MagicMock()
        post_response.raise_for_status = MagicMock()
        mock_endpoint = Endpoint(
            endpoint_id="33665",
            provider=client.provider,
            model_id="Qwen/Qwen3-4B",
            endpoint_url=None,
            state=EndpointState.RUNNING,
            hardware=HardwareConfig(accelerator="NVIDIA_H100", count=1),
            autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=1),
        )

        async with client:
            with (
                patch.object(
                    client._client,
                    "post",
                    new_callable=AsyncMock,
                    return_value=post_response,
                ) as mock_post,
                patch.object(
                    client,
                    "get_endpoint",
                    new_callable=AsyncMock,
                    return_value=mock_endpoint,
                ) as mock_get_endpoint,
            ):
                await client.start_endpoint("33665")
                await client.stop_endpoint("33665")

        assert mock_post.await_count == 2
        assert mock_post.await_args_list[0].args[0].endswith("/resume")
        assert mock_post.await_args_list[1].args[0].endswith("/pause")
        assert mock_get_endpoint.await_count == 2

    @pytest.mark.asyncio
    async def test_list_endpoints_and_endpoint_exists(self):
        client = ParasailDeploymentClient(api_key="test-key")
        list_response = MagicMock()
        list_response.raise_for_status = MagicMock()
        list_response.json.return_value = [
            _deployment_payload(deployment_id=111),
            _deployment_payload(deployment_id=222, status="STARTING"),
        ]

        async with client:
            with patch.object(
                client._client,
                "get",
                new_callable=AsyncMock,
                return_value=list_response,
            ):
                endpoints = await client.list_endpoints()
                exists = await client.endpoint_exists("222")
                missing = await client.endpoint_exists("999")

        assert len(endpoints) == 2
        assert endpoints[1].state == EndpointState.STARTING
        assert exists is True
        assert missing is False

    @pytest.mark.asyncio
    async def test_list_hardware_requires_model_id(self):
        client = ParasailDeploymentClient(api_key="test-key")
        async with client:
            with pytest.raises(ValueError, match="requires a model ID"):
                await client.list_hardware()

    @pytest.mark.asyncio
    async def test_list_hardware_maps_device_configs(self):
        client = ParasailDeploymentClient(api_key="test-key")
        device_response = MagicMock()
        device_response.raise_for_status = MagicMock()
        device_response.json.return_value = [
            {"device": "NVIDIA_H100", "count": 1, "available": True, "selected": True},
            {"device": "NVIDIA_4090", "count": 1, "available": True, "selected": False},
        ]

        async with client:
            with (
                patch("oumi.deploy.parasail_client.resolve_hf_token", return_value=""),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=device_response,
                ) as mock_get,
            ):
                hardware = await client.list_hardware("Qwen/Qwen3-4B")

        assert len(hardware) == 2
        assert hardware[0].accelerator == "NVIDIA_H100"
        assert hardware[0].count == 1
        assert mock_get.call_args[0][0] == "/dedicated/devices"

    @pytest.mark.asyncio
    async def test_get_and_select_device_configs_marks_single_match(self):
        client = ParasailDeploymentClient(api_key="test-key")
        devices = [
            DeviceConfig(
                device="NVIDIA_H100",
                count=1,
                selected=False,
                displayName=None,
                estimatedSingleUserTps=None,
                estimatedSystemTps=None,
                limitedContext=False,
            ),
            DeviceConfig(
                device="NVIDIA_A100",
                count=1,
                selected=False,
                displayName=None,
                estimatedSingleUserTps=None,
                estimatedSystemTps=None,
                limitedContext=False,
            ),
        ]
        async with client:
            with patch.object(
                client,
                "_fetch_device_configs",
                new_callable=AsyncMock,
                return_value=devices,
            ):
                selected = await client._get_and_select_device_configs(
                    model_id="Qwen/Qwen3-4B",
                    desired_device="NVIDIA_A100",
                    desired_count=1,
                )

        assert selected[0].selected is False
        assert selected[1].selected is True

    @pytest.mark.asyncio
    async def test_get_and_select_device_configs_raises_when_missing(self):
        client = ParasailDeploymentClient(api_key="test-key")
        devices = [
            DeviceConfig(
                device="NVIDIA_H100",
                count=1,
                selected=False,
                displayName=None,
                estimatedSingleUserTps=None,
                estimatedSystemTps=None,
                limitedContext=False,
            ),
        ]
        async with client:
            with patch.object(
                client,
                "_fetch_device_configs",
                new_callable=AsyncMock,
                return_value=devices,
            ):
                with pytest.raises(
                    ValueError, match="No matching hardware config found"
                ):
                    await client._get_and_select_device_configs(
                        model_id="Qwen/Qwen3-4B",
                        desired_device="NVIDIA_A100",
                        desired_count=1,
                    )

    @pytest.mark.asyncio
    async def test_update_endpoint_updates_replicas_only(self):
        client = ParasailDeploymentClient(api_key="test-key")
        get_response = MagicMock()
        get_response.raise_for_status = MagicMock()
        get_response.json.return_value = _deployment_payload(replicas=1)

        put_response = MagicMock()
        put_response.raise_for_status = MagicMock()
        put_response.json.return_value = _deployment_payload(replicas=3)

        async with client:
            with (
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=get_response,
                ),
                patch.object(
                    client._client,
                    "put",
                    new_callable=AsyncMock,
                    return_value=put_response,
                ) as mock_put,
            ):
                endpoint = await client.update_endpoint(
                    endpoint_id="33665",
                    autoscaling=AutoscalingConfig(min_replicas=3, max_replicas=10),
                    hardware=HardwareConfig(accelerator="NVIDIA_A100", count=2),
                )

        assert endpoint.autoscaling.min_replicas == 3
        assert mock_put.call_args[1]["json"]["replicas"] == 3

    @pytest.mark.asyncio
    async def test_get_model_status_and_delete_model(self):
        client = ParasailDeploymentClient(api_key="test-key")
        async with client:
            assert await client.get_model_status("Qwen/Qwen3-4B") == "ready"
            with pytest.raises(
                NotImplementedError, match="does not manage model storage"
            ):
                await client.delete_model("Qwen/Qwen3-4B")

    def test_to_endpoint_maps_unknown_status_to_pending_and_fallback_hardware(self):
        payload = _deployment_payload(status="ONLINE")
        payload["deviceConfigs"] = []
        payload["externalAlias"] = None
        payload["createdAt"] = None

        dep = DeploymentResponse.model_validate(payload)
        with patch.dict(parasail_client_module._PARASAIL_STATE_MAP, {}, clear=True):
            endpoint = _to_endpoint(dep)

        assert endpoint.state == EndpointState.PENDING
        assert endpoint.hardware.accelerator == "unknown"
        assert endpoint.endpoint_url is None
        assert endpoint.created_at is not None
        assert endpoint.created_at.tzinfo == timezone.utc

    # ------------------------------------------------------------------
    # upload_model soft-failure branches
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upload_model_warns_on_non_200_and_returns_ready(self, caplog):
        """Non-200 from /dedicated/support logs a warning but still returns ready."""
        client = ParasailDeploymentClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 503

        async with client:
            with (
                patch("oumi.deploy.parasail_client.resolve_hf_token", return_value=""),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=mock_response,
                ),
            ):
                with caplog.at_level(
                    logging.WARNING, logger="oumi.deploy.parasail_client"
                ):
                    uploaded = await client.upload_model(
                        model_source="Qwen/Qwen3-4B", model_name="q"
                    )

        assert uploaded.status == "ready"
        assert any("Could not verify model compatibility" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_upload_model_warns_when_supported_false(self, caplog):
        """supported=False logs a warning with the reason but does not raise."""
        client = ParasailDeploymentClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "supported": False,
            "errorMessage": "Context window too large",
            "messages": [],
        }

        async with client:
            with (
                patch("oumi.deploy.parasail_client.resolve_hf_token", return_value=""),
                patch(
                    "oumi.deploy.parasail_client.warn_if_private_model_missing_token"
                ),
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=mock_response,
                ),
            ):
                with caplog.at_level(
                    logging.WARNING, logger="oumi.deploy.parasail_client"
                ):
                    uploaded = await client.upload_model(
                        model_source="Qwen/Qwen3-4B", model_name="q"
                    )

        assert uploaded.status == "ready"
        assert any("may not be supported" in m for m in caplog.messages)
        assert any("Context window too large" in m for m in caplog.messages)

    # ------------------------------------------------------------------
    # _to_endpoint known-status parametrized mapping
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "parasail_status, expected_state",
        [
            ("ONLINE", EndpointState.RUNNING),
            ("STARTING", EndpointState.STARTING),
            ("PAUSED", EndpointState.STOPPED),
            ("STOPPING", EndpointState.STOPPING),
            ("OFFLINE", EndpointState.STOPPED),
        ],
    )
    def test_to_endpoint_maps_all_known_statuses(self, parasail_status, expected_state):
        dep = DeploymentResponse.model_validate(
            _deployment_payload(status=parasail_status)
        )
        endpoint = _to_endpoint(dep)
        assert endpoint.state == expected_state

    # ------------------------------------------------------------------
    # update_endpoint: autoscaling=None preserves replicas
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_endpoint_with_no_autoscaling_preserves_replicas(self):
        client = ParasailDeploymentClient(api_key="test-key")
        original = _deployment_payload(replicas=2)
        get_response = MagicMock()
        get_response.raise_for_status = MagicMock()
        get_response.json.return_value = original

        put_response = MagicMock()
        put_response.raise_for_status = MagicMock()
        put_response.json.return_value = _deployment_payload(replicas=2)

        async with client:
            with (
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=get_response,
                ),
                patch.object(
                    client._client,
                    "put",
                    new_callable=AsyncMock,
                    return_value=put_response,
                ) as mock_put,
            ):
                await client.update_endpoint(endpoint_id="33665", autoscaling=None)

        assert mock_put.call_args[1]["json"]["replicas"] == 2

    # ------------------------------------------------------------------
    # update_endpoint: hardware argument is ignored in PUT payload
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_endpoint_ignores_hardware_in_put_payload(self):
        client = ParasailDeploymentClient(api_key="test-key")
        original = _deployment_payload(replicas=1)
        get_response = MagicMock()
        get_response.raise_for_status = MagicMock()
        get_response.json.return_value = original

        put_response = MagicMock()
        put_response.raise_for_status = MagicMock()
        put_response.json.return_value = _deployment_payload(replicas=1)

        async with client:
            with (
                patch.object(
                    client._client,
                    "get",
                    new_callable=AsyncMock,
                    return_value=get_response,
                ),
                patch.object(
                    client._client,
                    "put",
                    new_callable=AsyncMock,
                    return_value=put_response,
                ) as mock_put,
            ):
                await client.update_endpoint(
                    endpoint_id="33665",
                    hardware=HardwareConfig(accelerator="NVIDIA_A100", count=4),
                )

        put_payload = mock_put.call_args[1]["json"]
        assert put_payload["deviceConfigs"][0]["device"] == "NVIDIA_H100"
        assert "NVIDIA_A100" not in str(put_payload)

    # ------------------------------------------------------------------
    # list_models derives Model records from endpoints
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_derives_from_endpoints(self):
        client = ParasailDeploymentClient(api_key="test-key")
        list_response = MagicMock()
        list_response.raise_for_status = MagicMock()
        list_response.json.return_value = [
            _deployment_payload(
                deployment_id=1, model_name="Qwen/Qwen3-4B", status="ONLINE"
            ),
            _deployment_payload(
                deployment_id=2, model_name="meta-llama/Llama-3-8B", status="PAUSED"
            ),
        ]

        async with client:
            with patch.object(
                client._client,
                "get",
                new_callable=AsyncMock,
                return_value=list_response,
            ):
                models = await client.list_models()

        assert len(models) == 2
        assert models[0].model_id == "Qwen/Qwen3-4B"
        assert models[0].status == "running"
        assert models[0].provider == DeploymentProvider.PARASAIL
        assert models[1].model_id == "meta-llama/Llama-3-8B"
        assert models[1].status == "stopped"

    # ------------------------------------------------------------------
    # HTTP error propagation via raise_for_status
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_http_error_propagates_from_get_endpoint(self):
        """4xx/5xx from the API surfaces as httpx.HTTPStatusError."""
        client = ParasailDeploymentClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        async with client:
            with patch.object(
                client._client,
                "get",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                with pytest.raises(httpx.HTTPStatusError):
                    await client.get_endpoint("99999")
