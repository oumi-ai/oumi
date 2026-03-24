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

"""TypedDict models for Oumi MCP server data structures."""

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

from oumi.mcp.constants import AcceleratorType, TaskType


class ConfigMetadata(TypedDict):
    """Metadata extracted from an Oumi config file.

    Attributes:
        path: Relative path to the config file.
        description: Description extracted from header comments.
        model_name: HuggingFace model ID or model name.
        task_type: Type of task (sft, dpo, grpo, evaluation, etc.).
        datasets: List of dataset names used in the config.
        reward_functions: List of reward functions for RLHF training.
        peft_type: Type of PEFT (lora/qlora) if applicable.
    """

    path: str
    description: str
    model_name: str
    task_type: TaskType | Literal[""]
    datasets: list[str]
    reward_functions: list[str]
    peft_type: str


class ConfigDetail(ConfigMetadata):
    """Full config details including YAML content.

    Attributes:
        content: Full YAML content.
        error: Error message if config not found (empty string if no error).
    """

    content: str
    error: str


class CategoriesResponse(TypedDict):
    """Response from list_categories tool.

    Attributes:
        categories: Top-level directories.
        model_families: Available model families in recipes/.
        api_providers: Available API providers in apis/.
        total_configs: Total number of configs available.
        oumi_version: Installed oumi library version.
        configs_source: Where configs are loaded from (e.g. "cache:0.7",
            "cache:main", "bundled:0.7", "env:/path").
        version_warning: Non-empty when configs may be mismatched with the
            installed library version.
    """

    categories: list[str]
    model_families: list[str]
    api_providers: list[str]
    total_configs: int
    oumi_version: str
    configs_source: str
    version_warning: str


class HardwareInfo(TypedDict):
    """Detected hardware and installed ML packages on the local machine.

    Attributes:
        accelerator_type: "cuda", "mps", or "none".
        accelerator_count: Number of accelerators detected.
        gpu_name: GPU device name (None if no CUDA GPU).
        gpu_memory_gb: Total GPU memory in GB (None if no CUDA GPU).
        compute_capability: CUDA compute capability e.g. "8.0" (None if no CUDA GPU).
        cuda_version: CUDA toolkit version (None if no CUDA GPU).
        packages: Installed packages relevant to hardware checks,
        e.g. {"torch": "2.3.0"}.
    """

    accelerator_type: AcceleratorType
    accelerator_count: int
    gpu_name: str | None
    gpu_memory_gb: float | None
    compute_capability: str | None
    cuda_version: str | None
    packages: dict[str, str]


class CloudReadiness(TypedDict):
    """Cloud credential and readiness status from SkyPilot.

    Attributes:
        sky_installed: Whether the ``sky`` (SkyPilot) package is importable.
        enabled_clouds: List of cloud providers with valid credentials
            (e.g. ``["AWS", "GCP"]``).
        target_cloud_ready: Whether the specific target cloud (if requested)
            has valid credentials. None if no target cloud was specified.
        target_cloud: The target cloud that was checked (empty if none).
    """

    sky_installed: bool
    enabled_clouds: list[str]
    target_cloud_ready: bool | None
    target_cloud: str


class PreFlightCheckResponse(TypedDict):
    """Response from pre_flight_check tool.

    Attributes:
        blocking: True when errors contain hard blockers that WILL prevent the
            run from succeeding. When True, the user MUST resolve these issues
            before proceeding. Do NOT treat blocking issues as informational.
        summary: One-line human-readable verdict ("ready", "blocked: …", etc.).
            Surface this to the user prominently.
        hf_authenticated: Whether a valid HuggingFace token was found.
        repo_access: Per-repo access status: "valid", "gated", "not_found", or "error".
        hardware: Detected local hardware and installed packages.
        cloud_readiness: SkyPilot cloud credential status.
        errors: Issues that will cause the training run to crash.
        warnings: Potential issues that may be fine if targeting a remote cluster.
        paths: Config paths mapped to validation status: "valid", "not_found",
            "valid_remote", "not_found_warning", "local_machine_path_error",
            "missing_local_source", "unverifiable_remote", or
            "working_dir_suspicious".
        dataset_checks: Per-dataset validation results (e.g. "valid", "not_found",
            "warning_timeout"). Only present when datasets are found in config.
        suggested_configs: Relative config paths relevant to the model in this config.
            Only present when ``cloud`` was specified. Pass these to ``get_config()``
            to retrieve full YAML examples for reference or adaptation.
    """

    blocking: bool
    summary: str
    hf_authenticated: bool
    repo_access: dict[str, str]
    hardware: HardwareInfo
    cloud_readiness: CloudReadiness
    errors: list[str]
    warnings: list[str]
    paths: dict[str, str]
    skypilot_compat_issue: bool | None
    dataset_checks: dict[str, str] | None
    suggested_configs: list[str] | None


class PreFlightSummary(TypedDict):
    """Subset of preflight results embedded in job submission responses.

    Only included for cloud jobs when preflight checks actually ran.

    Attributes:
        summary: One-line human-readable verdict.
        blocking: True when errors contain hard blockers.
        errors: Issues that will cause the run to crash.
        warnings: Potential issues that may be fine for remote clusters.
    """

    summary: str
    blocking: bool
    errors: list[str]
    warnings: list[str]


class JobSubmissionResponse(TypedDict):
    """Response from run_oumi_job tool.

    Returned immediately when a job is submitted (or dry-run previewed).

    Attributes:
        success: Whether the submission/dry-run succeeded.
        job_id: Unique job identifier for use with get_job_status / cancel_job.
        status: "dry_run", "submitted", or an error indicator.
        dry_run: True if this was a dry-run preview (no actual execution).
        command: The Oumi CLI subcommand (train, evaluate, etc.).
        config_path: Absolute path to the config file.
        cloud: Cloud provider (e.g. "local", "gcp", "aws").
        cluster_name: Cluster name (empty if auto-generated).
        model_name: HuggingFace model ID extracted from config.
        message: Human-readable summary of what happened or will happen.
        error: Error message if success is False.
        launch_confirmed: True if the launch was confirmed (cloud only).
        preflight: Nested preflight results (cloud only, when preflight ran).
    """

    success: bool
    job_id: str
    status: str
    dry_run: bool
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    model_name: str
    message: str
    error: str | None
    launch_confirmed: bool | None
    preflight: PreFlightSummary | None


class JobStatusResponse(TypedDict):
    """Response from ``get_job_status`` snapshot tool.

    Attributes:
        success: Whether the status lookup succeeded.
        job_id: The job identifier.
        status: Current status string from the launcher.
        state: Job state enum name (QUEUED, RUNNING, COMPLETED, FAILED, CANCELED).
        command: The Oumi CLI subcommand.
        config_path: Absolute path to the config.
        cloud: Cloud provider name.
        cluster: Cluster name the job is running on.
        model_name: Model being trained/evaluated.
        is_done: True if the job is in a terminal state.
        metadata: Additional metadata from the launcher.
        log_file: Absolute path to the full stdout log file on disk (if available).
        error: Error message if the job failed or lookup failed.
    """

    success: bool
    job_id: str
    status: str
    state: str
    command: str
    config_path: str
    cloud: str
    cluster: str
    model_name: str
    is_done: bool
    metadata: NotRequired[dict[str, Any]]
    log_file: NotRequired[str]
    error: str | None


class JobCancelResponse(TypedDict):
    """Response from cancel_job tool.

    Attributes:
        success: Whether cancellation succeeded.
        message: Human-readable result description.
        error: Error message if cancellation failed.
    """

    success: bool
    message: NotRequired[str]
    error: NotRequired[str]


class JobSummary(TypedDict):
    """Compact job summary for resource listings.

    Attributes:
        job_id: Unique MCP job identifier.
        command: Oumi CLI subcommand.
        status: Current lifecycle state.
        cloud: Cloud provider name.
        cluster: Cluster name.
        model_name: Model being trained/evaluated.
        is_done: Whether the job has finished.
    """

    job_id: str
    command: str
    status: str
    cloud: str
    cluster: str
    model_name: str
    is_done: bool


class JobLogsResponse(TypedDict):
    """Response from ``get_job_logs`` snapshot tool.

    Attributes:
        success: Whether log retrieval succeeded.
        job_id: The MCP job identifier.
        lines_requested: Number of trailing lines requested.
        lines_returned: Number of trailing lines returned.
        log_file: Absolute path to the stdout log file (if available).
        logs: Tail content for the requested number of lines.
        error: Error message if lookup or reading failed.
    """

    success: bool
    job_id: str
    lines_requested: int
    lines_returned: int
    log_file: str
    logs: str
    error: str | None


class FieldDoc(TypedDict):
    """Documentation for a single dataclass field.

    Attributes:
        name: Field name.
        type_str: String representation of the field type.
        description: Docstring extracted from AST (string literal after assignment).
        default: String representation of the default value, or "" if required.
    """

    name: str
    type_str: str
    description: str
    default: str


class DocstringSection(TypedDict):
    """A named section from a parsed Google-style docstring.

    Attributes:
        name: Section header (e.g. "Args", "Returns", "Raises").
        content: Full text content of the section.
    """

    name: str
    content: str


class DocEntry(TypedDict):
    """A single indexed documentation entry (class, function, or method).

    Attributes:
        qualified_name: Fully qualified name (e.g. "oumi.core.configs.TrainingConfig").
        name: Short name (e.g. "TrainingConfig").
        kind: Entry kind: "class", "dataclass", "function", or "method".
        module: Module path (e.g. "oumi.core.configs").
        summary: First line/paragraph of the docstring.
        sections: Parsed docstring sections (Args, Returns, etc.).
        fields: Dataclass field documentation (empty for non-dataclasses).
        signature: Function/method signature string.
        parent_class: Parent class name for methods, "" otherwise.
    """

    qualified_name: str
    name: str
    kind: str
    module: str
    summary: str
    sections: list[DocstringSection]
    fields: list[FieldDoc]
    signature: str
    parent_class: str


class DocsSearchResponse(TypedDict):
    """Response from the get_docs tool.

    Attributes:
        results: Matching documentation entries.
        query: The normalized query terms used for this search.
        total_matches: Total number of matches before limiting.
        index_ready: Whether background indexing has completed.
        oumi_version: Version of the installed oumi library the docs reflect.
        error: Error message, or "" if no error.
    """

    results: list[DocEntry]
    query: list[str]
    total_matches: int
    index_ready: bool
    oumi_version: str
    error: str


class ModuleInfo(TypedDict):
    """Summary information for an indexed module.

    Attributes:
        module: Fully qualified module path.
        description: Human-readable description of the module.
        class_count: Number of classes indexed from this module.
        function_count: Number of module-level functions indexed.
        class_names: Names of classes found in the module.
    """

    module: str
    description: str
    class_count: int
    function_count: int
    class_names: list[str]


class ListModulesResponse(TypedDict):
    """Response from the list_modules tool.

    Attributes:
        modules: Per-module summaries.
        total_entries: Total number of indexed documentation entries.
        index_ready: Whether background indexing has completed.
        oumi_version: Version of the installed oumi library the docs reflect.
    """

    modules: list[ModuleInfo]
    total_entries: int
    index_ready: bool
    oumi_version: str


class ValidateConfigResponse(TypedDict):
    """Response from validate_config tool.

    Attributes:
        valid: True if the config is valid against its schema.
        error: Validation error message, or None if valid.
    """

    valid: bool
    error: str | None


class ClusterLifecycleResponse(TypedDict):
    """Response from stop_cluster and down_cluster tools.

    Attributes:
        success: Whether the operation succeeded.
        message: Human-readable result description.
        error: Error message if the operation failed.
    """

    success: bool
    message: NotRequired[str]
    error: NotRequired[str]


class ConfigSyncResponse(TypedDict):
    """Response from config_sync().

    Attributes:
        ok: Whether the sync succeeded (or was skipped because cache is fresh).
        skipped: True if the cache was fresh and no download was performed.
        error: Error message, or None if no error.
        configs_synced: Number of YAML config files synced (0 if skipped or failed).
        source: Label describing the sync source (e.g. "tag:v0.7", "main").
    """

    ok: bool
    skipped: bool
    error: str | None
    configs_synced: int
    source: str


class CloudJobConfigTemplateResponse(TypedDict):
    """Response from get_cloud_job_config_template tool.

    Attributes:
        cloud: Normalized cloud provider name.
        template_yaml: Complete job config YAML string, ready to customize and save.
        key_fields: List of fields the agent must customize before using.
        notes: Cloud-specific notes and tips.
    """

    cloud: str
    template_yaml: str
    key_fields: list[str]
    notes: str
