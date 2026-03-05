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

"""Pydantic models for the Parasail REST API.

Shapes are derived from live API responses captured against
https://api.parasail.io/api/v1  (March 2026).

Two APIs exist:
- Control API  (base: https://api.parasail.io/api/v1)  — manages deployments
- Inference API (base: https://api.parasail.io/v1)     — OpenAI-compatible chat
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ParasailDeploymentStatus(str, Enum):
    """Possible values of ``DeploymentStatusBlock.status``."""

    ONLINE = "ONLINE"
    STARTING = "STARTING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    OFFLINE = "OFFLINE"


class ParasailScaleDownPolicy(str, Enum):
    NONE = "NONE"
    TIMER = "TIMER"
    INACTIVE = "INACTIVE"


class ParasailEngine(str, Enum):
    VLLM = "VLLM"
    SGLANG = "SGLANG"


class ParasailEngineTask(str, Enum):
    AUTO = "AUTO"
    GENERATE = "GENERATE"
    EMBED = "EMBED"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DeviceConfig(BaseModel):
    """A single hardware configuration option for a deployment.

    Returned by ``GET /dedicated/devices`` and embedded in deployment objects.
    The ``selected`` flag must be set to ``True`` for the desired hardware
    before passing the list to ``POST /dedicated/deployments``.
    """

    model_config = ConfigDict(populate_by_name=True)

    device: str
    count: int
    display_name: str | None = Field(None, alias="displayName")
    cost: float | None = None
    estimated_single_user_tps: float | None = Field(
        None, alias="estimatedSingleUserTps"
    )
    estimated_system_tps: float | None = Field(None, alias="estimatedSystemTps")
    recommended: bool = False
    limited_context: bool = Field(False, alias="limitedContext")
    available: bool = False
    selected: bool = False

    def to_api_dict(self) -> dict[str, Any]:
        """Serialises back to the camelCase dict the API expects."""
        return self.model_dump(by_alias=True, exclude_none=False)


class DeploymentStatusBlock(BaseModel):
    """Nested ``status`` object inside a deployment response."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    status: ParasailDeploymentStatus
    status_message: str | None = Field(None, alias="statusMessage")
    status_last_updated_ms: int | None = Field(None, alias="statusLastUpdated")
    instances: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def last_updated_at(self) -> datetime | None:
        """Converts the epoch-millisecond timestamp to an aware ``datetime``."""
        if self.status_last_updated_ms is None:
            return None
        return datetime.fromtimestamp(
            self.status_last_updated_ms / 1000.0, tz=timezone.utc
        )


class ChatTemplate(BaseModel):
    """A single chat template entry inside ``ModelProperties``."""

    model_config = ConfigDict(populate_by_name=True)

    chat_template: str = Field(alias="chat_template")
    name: str | None = None
    source: str | None = None
    description: str | None = None
    is_model_default: bool = Field(False, alias="is_model_default")


class ModelProperties(BaseModel):
    """Model metadata returned in support-check and deployment responses."""

    model_config = ConfigDict(populate_by_name=True)

    context: int | None = None
    parameter_count: int | None = None
    parameter_quantization_bits: int | None = None
    quantization_high_level: str | None = None
    quantization_detail: str | None = None
    parameter_total_bytes: int | None = None
    base_model: str | None = None
    chat_templates: list[ChatTemplate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateDeploymentRequest(BaseModel):
    """Request body for ``POST /dedicated/deployments``."""

    model_config = ConfigDict(populate_by_name=True)

    deployment_name: str = Field(alias="deploymentName")
    model_name: str = Field(alias="modelName")
    device_configs: list[DeviceConfig] = Field(alias="deviceConfigs")
    replicas: int = 1
    scale_down_policy: ParasailScaleDownPolicy | None = Field(
        None, alias="scaleDownPolicy"
    )
    scale_down_threshold: int | None = Field(
        None,
        alias="scaleDownThreshold",
        description="Idle threshold in milliseconds before scaling down.",
    )
    draft_model_name: str | None = Field(default=None, alias="draftModelName")
    generative: bool | None = None
    embedding: bool | None = None
    context_length: int | None = Field(default=None, alias="contextLength")
    model_access_key: str | None = Field(
        default=None,
        alias="modelAccessKey",
        description="HuggingFace token for private models.",
    )

    def to_api_dict(self) -> dict[str, Any]:
        """Serialises to the camelCase payload the API expects, omitting nulls."""
        raw = self.model_dump(by_alias=True, exclude_none=True)
        # device_configs must always use camelCase sub-dicts
        raw["deviceConfigs"] = [d.to_api_dict() for d in self.device_configs]
        return raw


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------


class DeploymentResponse(BaseModel):
    """Full deployment object returned by create / get / list endpoints.

    ``createdAt`` and ``startedAt`` are epoch milliseconds (integer), not ISO
    strings — converted to aware ``datetime`` via properties.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: int
    account_name: str | None = Field(None, alias="accountName")
    deployment_name: str = Field(alias="deploymentName")
    display_name: str | None = Field(None, alias="displayName")
    external_alias: str | None = Field(None, alias="externalAlias")
    custom_aliases: list[str] = Field(default_factory=list, alias="customAliases")
    tags: list[str] = Field(default_factory=list)
    model_name: str = Field(alias="modelName")
    base_url: str | None = Field(None, alias="baseUrl")
    replicas: int = 1
    scale_down_policy: str | None = Field(None, alias="scaleDownPolicy")
    engine_task: str | None = Field(None, alias="engineTask")
    batch: bool = False
    status: DeploymentStatusBlock
    shared: bool = False
    shared_private: bool = Field(False, alias="sharedPrivate")
    engine: str | None = None
    context_length: int | None = Field(None, alias="contextLength")
    autoscaling: bool = False
    created_at_ms: int | None = Field(None, alias="createdAt")
    started_at_ms: int | None = Field(None, alias="startedAt")
    creator_username: str | None = Field(None, alias="creatorUsername")
    creator_email: str | None = Field(None, alias="creatorEmail")
    creator_user_id: int | None = Field(None, alias="creatorUserId")
    device_configs: list[DeviceConfig] = Field(
        default_factory=list, alias="deviceConfigs"
    )
    extra_cmd_args: list[str] = Field(default_factory=list, alias="extraCmdArgs")
    extra_env_vars: dict[str, str] = Field(default_factory=dict, alias="extraEnvVars")
    chart_args: dict[str, Any] = Field(default_factory=dict, alias="chartArgs")
    model_properties: ModelProperties | None = Field(None, alias="modelProperties")

    @property
    def created_at(self) -> datetime | None:
        """Converts ``createdAt`` epoch-milliseconds to an aware ``datetime``."""
        if self.created_at_ms is None:
            return None
        return datetime.fromtimestamp(self.created_at_ms / 1000.0, tz=timezone.utc)

    @property
    def started_at(self) -> datetime | None:
        """Converts ``startedAt`` epoch-milliseconds to an aware ``datetime``."""
        if self.started_at_ms is None:
            return None
        return datetime.fromtimestamp(self.started_at_ms / 1000.0, tz=timezone.utc)

    @property
    def deployment_status(self) -> ParasailDeploymentStatus:
        """Returns the current ``ParasailDeploymentStatus`` from the status block."""
        return self.status.status

    @property
    def selected_device(self) -> DeviceConfig | None:
        """Returns the hardware config that was actually selected."""
        for d in self.device_configs:
            if d.selected:
                return d
        return self.device_configs[0] if self.device_configs else None


class SupportCheckResponse(BaseModel):
    """Response from ``GET /dedicated/support``."""

    model_config = ConfigDict(populate_by_name=True)

    supported: bool
    known: bool = False
    generative: bool = False
    embedding: bool = False
    multimodal: bool = False
    supporting_engines: list[str] = Field(
        default_factory=list, alias="supportingEngines"
    )
    model_redirect: str | None = Field(None, alias="modelRedirect")
    messages: list[str] = Field(default_factory=list)
    properties: ModelProperties | None = None
    error_message: str | None = Field(None, alias="errorMessage")

    @field_validator("supported", mode="before")
    @classmethod
    def _coerce_supported(cls, v: Any) -> bool:
        return bool(v)
