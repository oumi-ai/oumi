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

"""Pydantic data models for the Fireworks.ai REST API.

Generated from the API Documentation:
https://docs.fireworks.ai/getting-started/introduction

These models provide typed request/response schemas used by
:class:`~oumi.deploy.fireworks_client.FireworksDeploymentClient`.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from oumi.deploy.base_client import EndpointState

# ---------------------------------------------------------------------------
# Enums — generated from fireworks.openapi.yaml §components/schemas
# ---------------------------------------------------------------------------


class GatewayModelKind(str, Enum):
    """gatewayModelKind — the kind of model."""

    KIND_UNSPECIFIED = "KIND_UNSPECIFIED"
    HF_BASE_MODEL = "HF_BASE_MODEL"
    HF_PEFT_ADDON = "HF_PEFT_ADDON"
    HF_TEFT_ADDON = "HF_TEFT_ADDON"
    FLUMINA_BASE_MODEL = "FLUMINA_BASE_MODEL"
    FLUMINA_ADDON = "FLUMINA_ADDON"
    DRAFT_ADDON = "DRAFT_ADDON"
    FIRE_AGENT = "FIRE_AGENT"
    LIVE_MERGE = "LIVE_MERGE"
    CUSTOM_MODEL = "CUSTOM_MODEL"
    EMBEDDING_MODEL = "EMBEDDING_MODEL"
    SNAPSHOT_MODEL = "SNAPSHOT_MODEL"


class GatewayModelState(str, Enum):
    """gatewayModelState — state of a model."""

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    UPLOADING = "UPLOADING"
    READY = "READY"


class GatewayDeploymentState(str, Enum):
    """gatewayDeploymentState — state of a deployment."""

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    CREATING = "CREATING"
    READY = "READY"
    DELETING = "DELETING"
    FAILED = "FAILED"
    UPDATING = "UPDATING"
    DELETED = "DELETED"


class GatewayDeployedModelState(str, Enum):
    """gatewayDeployedModelState — state of a deployed model (LoRA)."""

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    UNDEPLOYING = "UNDEPLOYING"
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"
    UPDATING = "UPDATING"


class GatewayAcceleratorType(str, Enum):
    """gatewayAcceleratorType — accelerator hardware type.

    Maps to the ``acceleratorType`` field in the Fireworks REST API:
    https://docs.fireworks.ai/api-reference/create-deployment#body-accelerator-type
    """

    ACCELERATOR_TYPE_UNSPECIFIED = "ACCELERATOR_TYPE_UNSPECIFIED"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"
    AMD_MI300X_192GB = "AMD_MI300X_192GB"
    NVIDIA_A10G_24GB = "NVIDIA_A10G_24GB"
    NVIDIA_A100_40GB = "NVIDIA_A100_40GB"
    NVIDIA_L4_24GB = "NVIDIA_L4_24GB"
    NVIDIA_H200_141GB = "NVIDIA_H200_141GB"
    NVIDIA_B200_180GB = "NVIDIA_B200_180GB"
    AMD_MI325X_256GB = "AMD_MI325X_256GB"
    AMD_MI350X_288GB = "AMD_MI350X_288GB"


class DeploymentPrecision(str, Enum):
    """DeploymentPrecision — precision for model serving."""

    PRECISION_UNSPECIFIED = "PRECISION_UNSPECIFIED"
    FP16 = "FP16"
    FP8 = "FP8"
    FP8_MM = "FP8_MM"
    FP8_AR = "FP8_AR"
    FP8_MM_KV_ATTN = "FP8_MM_KV_ATTN"
    FP8_KV = "FP8_KV"
    FP8_MM_V2 = "FP8_MM_V2"
    FP8_V2 = "FP8_V2"
    FP8_MM_KV_ATTN_V2 = "FP8_MM_KV_ATTN_V2"
    NF4 = "NF4"
    FP4 = "FP4"
    BF16 = "BF16"
    FP4_BLOCKSCALED_MM = "FP4_BLOCKSCALED_MM"
    FP4_MX_MOE = "FP4_MX_MOE"


class BaseModelDetailsCheckpointFormat(str, Enum):
    """BaseModelDetailsCheckpointFormat — checkpoint format for base models."""

    CHECKPOINT_FORMAT_UNSPECIFIED = "CHECKPOINT_FORMAT_UNSPECIFIED"
    NATIVE = "NATIVE"
    HUGGINGFACE = "HUGGINGFACE"
    UNINITIALIZED = "UNINITIALIZED"


class GatewayCode(str, Enum):
    """gatewayCode — RPC status codes."""

    OK = "OK"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    ABORTED = "ABORTED"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    UNIMPLEMENTED = "UNIMPLEMENTED"
    INTERNAL = "INTERNAL"
    UNAVAILABLE = "UNAVAILABLE"
    DATA_LOSS = "DATA_LOSS"


# ---------------------------------------------------------------------------
# Pydantic nested models — building blocks for request/response schemas
# ---------------------------------------------------------------------------


class _FWBaseModel(BaseModel):
    """Shared config for all Fireworks API Pydantic models.

    Allows construction with snake_case kwargs and (de)serialization
    with camelCase JSON via ``by_alias=True``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class GatewayStatus(_FWBaseModel):
    """gatewayStatus — RPC-style status attached to models and deployments."""

    code: GatewayCode | None = None
    message: str | None = None


class GatewayBaseModelDetails(_FWBaseModel):
    """gatewayBaseModelDetails — details for HF_BASE_MODEL kind."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    world_size: int | None = Field(default=None, alias="worldSize")
    checkpoint_format: BaseModelDetailsCheckpointFormat | None = Field(
        default=None, alias="checkpointFormat"
    )
    parameter_count: str | None = Field(default=None, alias="parameterCount")
    moe: bool | None = None
    tunable: bool | None = None
    model_type: str | None = Field(default=None, alias="modelType")
    supports_fireattention: bool | None = Field(
        default=None, alias="supportsFireattention"
    )
    default_precision: DeploymentPrecision | None = Field(
        default=None, alias="defaultPrecision"
    )
    supports_mtp: bool | None = Field(default=None, alias="supportsMtp")
    huggingface_files: list[str] | None = Field(default=None, alias="huggingfaceFiles")


class GatewayPEFTDetails(_FWBaseModel):
    """gatewayPEFTDetails — PEFT addon details (LoRA).

    All three fields (baseModel, r, targetModules) are required by the spec.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    base_model: str = Field(alias="baseModel")
    r: int
    target_modules: list[str] = Field(alias="targetModules")
    base_model_type: str | None = Field(default=None, alias="baseModelType")
    merge_addon_model_name: str | None = Field(
        default=None, alias="mergeAddonModelName"
    )


class GatewayConversationConfig(_FWBaseModel):
    """gatewayConversationConfig — chat template configuration."""

    style: str
    system: str | None = None
    template: str | None = None


class GatewayDeployedModelRef(_FWBaseModel):
    """gatewayDeployedModelRef — reference to a deployed model."""

    name: str | None = None
    deployment: str | None = None
    state: GatewayDeployedModelState | None = None
    default: bool | None = None
    public: bool | None = None


class GatewayAutoscalingPolicy(_FWBaseModel):
    """gatewayAutoscalingPolicy — autoscaling configuration for a deployment."""

    scale_up_window: str | None = Field(default=None, alias="scaleUpWindow")
    scale_down_window: str | None = Field(default=None, alias="scaleDownWindow")
    scale_to_zero_window: str | None = Field(default=None, alias="scaleToZeroWindow")
    load_targets: dict[str, float] | None = Field(default=None, alias="loadTargets")


class GatewayReplicaStats(_FWBaseModel):
    """gatewayReplicaStats — per-replica status counters."""

    pending_scheduling_replica_count: int | None = Field(
        default=None, alias="pendingSchedulingReplicaCount"
    )
    downloading_model_replica_count: int | None = Field(
        default=None, alias="downloadingModelReplicaCount"
    )
    initializing_replica_count: int | None = Field(
        default=None, alias="initializingReplicaCount"
    )
    ready_replica_count: int | None = Field(default=None, alias="readyReplicaCount")


# ---------------------------------------------------------------------------
# Pydantic main models — request bodies and response schemas
# ---------------------------------------------------------------------------


class GatewayModel(_FWBaseModel):
    """gatewayModel — full model resource schema.

    Used in CreateModel requests (nested inside GatewayCreateModelBody)
    and returned by GetModel / ListModels responses.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str | None = None
    display_name: str | None = Field(default=None, alias="displayName")
    description: str | None = None
    create_time: datetime | None = Field(default=None, alias="createTime")
    state: GatewayModelState | None = None
    status: GatewayStatus | None = None
    kind: GatewayModelKind | None = None
    github_url: str | None = Field(default=None, alias="githubUrl")
    hugging_face_url: str | None = Field(default=None, alias="huggingFaceUrl")
    base_model_details: GatewayBaseModelDetails | None = Field(
        default=None, alias="baseModelDetails"
    )
    peft_details: GatewayPEFTDetails | None = Field(default=None, alias="peftDetails")
    public: bool | None = None
    conversation_config: GatewayConversationConfig | None = Field(
        default=None, alias="conversationConfig"
    )
    context_length: int | None = Field(default=None, alias="contextLength")
    supports_image_input: bool | None = Field(default=None, alias="supportsImageInput")
    supports_tools: bool | None = Field(default=None, alias="supportsTools")
    imported_from: str | None = Field(default=None, alias="importedFrom")
    fine_tuning_job: str | None = Field(default=None, alias="fineTuningJob")
    default_draft_model: str | None = Field(default=None, alias="defaultDraftModel")
    default_draft_token_count: int | None = Field(
        default=None, alias="defaultDraftTokenCount"
    )
    deployed_model_refs: list[GatewayDeployedModelRef] | None = Field(
        default=None, alias="deployedModelRefs"
    )
    cluster: str | None = None
    calibrated: bool | None = None
    tunable: bool | None = None
    supports_lora: bool | None = Field(default=None, alias="supportsLora")
    use_hf_apply_chat_template: bool | None = Field(
        default=None, alias="useHfApplyChatTemplate"
    )
    update_time: datetime | None = Field(default=None, alias="updateTime")
    default_sampling_params: dict[str, float] | None = Field(
        default=None, alias="defaultSamplingParams"
    )
    rl_tunable: bool | None = Field(default=None, alias="rlTunable")
    supported_precisions: list[DeploymentPrecision] | None = Field(
        default=None, alias="supportedPrecisions"
    )
    supported_precisions_with_calibration: list[DeploymentPrecision] | None = Field(
        default=None, alias="supportedPrecisionsWithCalibration"
    )
    training_context_length: int | None = Field(
        default=None, alias="trainingContextLength"
    )
    supports_serverless: bool | None = Field(default=None, alias="supportsServerless")
    base_model: str | None = Field(default=None, alias="baseModel")


class GatewayCreateModelBody(_FWBaseModel):
    """GatewayCreateModelBody — request body for POST /accounts/{id}/models.

    ``model_id`` is required. ``model`` carries the model properties.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    model_id: str = Field(alias="modelId")
    model: GatewayModel | None = None
    cluster: str | None = None


class GatewayDeployment(_FWBaseModel):
    """gatewayDeployment — deployment resource schema.

    Used as request body for CreateDeployment / UpdateDeployment and
    returned in GetDeployment / ListDeployments responses.
    ``base_model`` is required per the OpenAPI spec.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str | None = None
    display_name: str | None = Field(default=None, alias="displayName")
    description: str | None = None
    create_time: datetime | None = Field(default=None, alias="createTime")
    expire_time: datetime | None = Field(default=None, alias="expireTime")
    purge_time: datetime | None = Field(default=None, alias="purgeTime")
    delete_time: datetime | None = Field(default=None, alias="deleteTime")
    state: GatewayDeploymentState | None = None
    status: GatewayStatus | None = None
    min_replica_count: int | None = Field(default=None, alias="minReplicaCount")
    max_replica_count: int | None = Field(default=None, alias="maxReplicaCount")
    max_with_revocable_replica_count: int | None = Field(
        default=None, alias="maxWithRevocableReplicaCount"
    )
    desired_replica_count: int | None = Field(default=None, alias="desiredReplicaCount")
    replica_count: int | None = Field(default=None, alias="replicaCount")
    autoscaling_policy: GatewayAutoscalingPolicy | None = Field(
        default=None, alias="autoscalingPolicy"
    )
    base_model: str = Field(alias="baseModel")
    accelerator_count: int | None = Field(default=None, alias="acceleratorCount")
    accelerator_type: GatewayAcceleratorType | None = Field(
        default=None, alias="acceleratorType"
    )
    precision: DeploymentPrecision | None = None
    cluster: str | None = None
    enable_addons: bool | None = Field(default=None, alias="enableAddons")
    draft_token_count: int | None = Field(default=None, alias="draftTokenCount")
    draft_model: str | None = Field(default=None, alias="draftModel")
    ngram_speculation_length: int | None = Field(
        default=None, alias="ngramSpeculationLength"
    )
    enable_session_affinity: bool | None = Field(
        default=None, alias="enableSessionAffinity"
    )
    deployment_template: str | None = Field(default=None, alias="deploymentTemplate")
    max_context_length: int | None = Field(default=None, alias="maxContextLength")
    update_time: datetime | None = Field(default=None, alias="updateTime")
    enable_mtp: bool | None = Field(default=None, alias="enableMtp")
    enable_hot_load: bool | None = Field(default=None, alias="enableHotLoad")
    enable_hot_reload_latest_addon: bool | None = Field(
        default=None, alias="enableHotReloadLatestAddon"
    )
    deployment_shape: str | None = Field(default=None, alias="deploymentShape")
    active_model_version: str | None = Field(default=None, alias="activeModelVersion")
    target_model_version: str | None = Field(default=None, alias="targetModelVersion")
    replica_stats: GatewayReplicaStats | None = Field(
        default=None, alias="replicaStats"
    )


class GatewayGetModelUploadEndpointBody(_FWBaseModel):
    """GatewayGetModelUploadEndpointBody — request for signed upload URLs.

    ``filename_to_size`` is required: maps filenames to their byte sizes.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    filename_to_size: dict[str, str | int] = Field(alias="filenameToSize")
    enable_resumable_upload: bool | None = Field(
        default=None, alias="enableResumableUpload"
    )
    read_mask: str | None = Field(default=None, alias="readMask")


class GatewayPrepareModelBody(_FWBaseModel):
    """GatewayPrepareModelBody — request for preparing a model at a precision."""

    precision: DeploymentPrecision | None = None
    read_mask: str | None = Field(default=None, alias="readMask")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class GatewayGetModelUploadEndpointResponse(_FWBaseModel):
    """Response from getUploadEndpoint — maps filenames to signed URLs."""

    filename_to_signed_urls: dict[str, str] | None = Field(
        default=None, alias="filenameToSignedUrls"
    )
    filename_to_unsigned_uris: dict[str, str] | None = Field(
        default=None, alias="filenameToUnsignedUris"
    )


class GatewayListModelsResponse(_FWBaseModel):
    """gatewayListModelsResponse — paginated list of models."""

    models: list[GatewayModel] | None = None
    next_page_token: str | None = Field(default=None, alias="nextPageToken")
    total_size: int | None = Field(default=None, alias="totalSize")


class GatewayListDeploymentsResponse(_FWBaseModel):
    """gatewayListDeploymentsResponse — paginated list of deployments."""

    deployments: list[GatewayDeployment] | None = None
    next_page_token: str | None = Field(default=None, alias="nextPageToken")
    total_size: int | None = Field(default=None, alias="totalSize")


# ---------------------------------------------------------------------------
# Mapping from GatewayDeploymentState → EndpointState
# ---------------------------------------------------------------------------

FW_STATE_TO_ENDPOINT: dict[GatewayDeploymentState, EndpointState] = {
    GatewayDeploymentState.STATE_UNSPECIFIED: EndpointState.PENDING,
    GatewayDeploymentState.CREATING: EndpointState.STARTING,
    GatewayDeploymentState.READY: EndpointState.RUNNING,
    GatewayDeploymentState.DELETING: EndpointState.STOPPING,
    GatewayDeploymentState.DELETED: EndpointState.STOPPED,
    GatewayDeploymentState.FAILED: EndpointState.ERROR,
    GatewayDeploymentState.UPDATING: EndpointState.STARTING,
}
