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

"""Unit tests for base deployment client types."""

import pytest

from oumi.deploy.base_client import (
    AutoscalingConfig,
    DeploymentProvider,
    Endpoint,
    EndpointState,
    HardwareConfig,
    ModelType,
    UploadedModel,
)


class TestDeploymentProvider:
    """Tests for DeploymentProvider enum."""

    def test_deployment_provider_enum_values(self):
        """Test that provider enum has expected values."""
        assert DeploymentProvider.TOGETHER.value == "together"
        assert DeploymentProvider.FIREWORKS.value == "fireworks"

    def test_deployment_provider_is_string_enum(self):
        """Test that provider enum values are strings."""
        assert isinstance(DeploymentProvider.TOGETHER.value, str)
        assert isinstance(DeploymentProvider.FIREWORKS.value, str)


class TestEndpointState:
    """Tests for EndpointState enum."""

    def test_endpoint_state_enum_completeness(self):
        """Test that all expected states are present."""
        states = {s.value for s in EndpointState}
        expected = {"pending", "starting", "running", "stopping", "stopped", "error"}
        assert states == expected

    def test_endpoint_state_is_string_enum(self):
        """Test that state enum values are strings."""
        for state in EndpointState:
            assert isinstance(state.value, str)


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_type_enum_values(self):
        """Test that model type enum has expected values."""
        assert ModelType.FULL.value == "full"
        assert ModelType.ADAPTER.value == "adapter"


class TestHardwareConfig:
    """Tests for HardwareConfig dataclass."""

    def test_hardware_config_defaults(self):
        """Test that hardware config has correct defaults."""
        hw = HardwareConfig(accelerator="nvidia_a100_80gb")
        assert hw.accelerator == "nvidia_a100_80gb"
        assert hw.count == 1

    def test_hardware_config_with_count(self):
        """Test hardware config with explicit count."""
        hw = HardwareConfig(accelerator="nvidia_h100_80gb", count=2)
        assert hw.accelerator == "nvidia_h100_80gb"
        assert hw.count == 2

    def test_hardware_config_equality(self):
        """Test hardware config equality."""
        hw1 = HardwareConfig(accelerator="nvidia_a100_80gb", count=1)
        hw2 = HardwareConfig(accelerator="nvidia_a100_80gb", count=1)
        assert hw1 == hw2


class TestAutoscalingConfig:
    """Tests for AutoscalingConfig dataclass."""

    def test_autoscaling_config_defaults(self):
        """Test that autoscaling config has correct defaults."""
        asc = AutoscalingConfig()
        assert asc.min_replicas == 1
        assert asc.max_replicas == 1

    def test_autoscaling_config_with_values(self):
        """Test autoscaling config with explicit values."""
        asc = AutoscalingConfig(min_replicas=2, max_replicas=5)
        assert asc.min_replicas == 2
        assert asc.max_replicas == 5


class TestUploadedModel:
    """Tests for UploadedModel dataclass."""

    def test_uploaded_model_minimal(self):
        """Test uploaded model with minimal fields."""
        model = UploadedModel(provider_model_id="model-123")
        assert model.provider_model_id == "model-123"
        assert model.job_id is None
        assert model.status == "pending"

    def test_uploaded_model_full(self):
        """Test uploaded model with all fields."""
        model = UploadedModel(
            provider_model_id="model-123",
            job_id="job-456",
            status="ready",
        )
        assert model.provider_model_id == "model-123"
        assert model.job_id == "job-456"
        assert model.status == "ready"


class TestEndpoint:
    """Tests for Endpoint dataclass."""

    def test_endpoint_dataclass_fields(self):
        """Test endpoint dataclass with all fields."""
        ep = Endpoint(
            endpoint_id="ep-123",
            provider=DeploymentProvider.TOGETHER,
            model_id="model-456",
            endpoint_url="https://api.together.xyz/v1/chat/completions",
            state=EndpointState.RUNNING,
            hardware=HardwareConfig(accelerator="nvidia_a100_80gb", count=1),
            autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=2),
        )
        assert ep.endpoint_id == "ep-123"
        assert ep.provider == DeploymentProvider.TOGETHER
        assert ep.model_id == "model-456"
        assert ep.endpoint_url == "https://api.together.xyz/v1/chat/completions"
        assert ep.state == EndpointState.RUNNING
        assert ep.hardware.accelerator == "nvidia_a100_80gb"
        assert ep.autoscaling.max_replicas == 2
        assert ep.display_name is None
        assert ep.created_at is None

    def test_endpoint_with_optional_fields(self):
        """Test endpoint with optional fields filled."""
        from datetime import datetime

        created = datetime.now()
        ep = Endpoint(
            endpoint_id="ep-123",
            provider=DeploymentProvider.FIREWORKS,
            model_id="model-456",
            endpoint_url=None,
            state=EndpointState.PENDING,
            hardware=HardwareConfig(accelerator="nvidia_h100_80gb", count=2),
            autoscaling=AutoscalingConfig(),
            created_at=created,
            display_name="My Endpoint",
        )
        assert ep.created_at == created
        assert ep.display_name == "My Endpoint"
        assert ep.endpoint_url is None
