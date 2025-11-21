"""Configuration classes for workflow management."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf


class ResourceAllocationType(str, Enum):
    """Resource allocation strategies."""

    FIXED = "fixed"  # Jobs assigned to specific resources
    DYNAMIC = "dynamic"  # Resources assigned as available
    LOAD_BALANCED = "load_balanced"  # Assign to least busy resource
    HYBRID = "hybrid"  # Mix of local and remote


class ExecutionMode(str, Enum):
    """Execution modes for workflows."""

    LOCAL = "local"  # Execute on local machine
    REMOTE = "remote"  # Execute via oumi launch
    DISTRIBUTED = "distributed"  # Execute across multiple machines
    MIXED = "mixed"  # Some jobs local, some remote


@dataclass
class RemoteResourceConfig:
    """Configuration for remote execution resources."""

    name: str
    cluster: Optional[str] = None
    gpus: list[int] = field(default_factory=list)
    max_jobs: int = 1


@dataclass
class ResourceConfig:
    """Resource configuration for workflow execution."""

    # GPU configuration
    gpus: list[int] = field(default_factory=list)

    # Parallelism
    max_parallel: Optional[int] = None

    # Allocation strategy
    allocation: ResourceAllocationType = ResourceAllocationType.DYNAMIC

    # Remote resources
    remote: list[RemoteResourceConfig] = field(default_factory=list)

    # Execution mode
    mode: ExecutionMode = ExecutionMode.LOCAL

    # Cost tracking (USD per GPU-hour)
    cost_per_gpu_hour: Optional[float] = None


@dataclass
class JobConfig:
    """Configuration for a single job in a workflow."""

    # Job identification
    name: str

    # Verb to execute
    verb: str  # train, evaluate, infer, quantize, synth, judge

    # Config file for the verb (can be local path or oumi:// URI)
    config: str

    # Dependencies (job names that must complete before this job)
    depends_on: list[str] = field(default_factory=list)

    # Resource requirements
    resources: dict[str, Any] = field(default_factory=dict)
    # Examples:
    #   {"gpu": 0}  - specific GPU
    #   {"gpu": "auto"}  - auto-assign
    #   {"remote": "aws-cluster"}  - execute remotely
    #   {"cpus": 4, "memory": "32GB"}  - resource limits

    # Additional CLI args to override config
    args: list[str] = field(default_factory=list)

    # Environment variables
    env: dict[str, str] = field(default_factory=dict)

    # Working directory
    workdir: Optional[str] = None

    # Retry configuration
    max_retries: int = 0
    retry_delay: float = 60.0  # seconds

    # Timeout (seconds, None = no timeout)
    timeout: Optional[float] = None


@dataclass
class WorkflowConfig:
    """Configuration for a complete workflow."""

    # Workflow identification
    name: str
    description: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Resource configuration
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Jobs to execute
    jobs: list[JobConfig] = field(default_factory=list)

    # Global environment variables
    env: dict[str, str] = field(default_factory=dict)

    # Output directory for workflow results
    output_dir: Optional[str] = None

    # Enable/disable TUI
    tui: bool = True

    # Workflow-level timeout (seconds)
    timeout: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowConfig":
        """Load workflow configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            WorkflowConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow config not found: {path}")

        with open(path) as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data:
            raise ValueError(f"Empty workflow config: {path}")

        # Convert to OmegaConf for structured config
        conf = OmegaConf.create(yaml_data)

        # Parse resources
        resources_dict = conf.get("resources", {})

        # Parse remote resources if present
        remote_configs = []
        if "remote" in resources_dict:
            for remote_dict in resources_dict["remote"]:
                remote_configs.append(
                    RemoteResourceConfig(
                        name=remote_dict["name"],
                        cluster=remote_dict.get("cluster"),
                        gpus=remote_dict.get("gpus", []),
                        max_jobs=remote_dict.get("max_jobs", 1),
                    )
                )

        resources = ResourceConfig(
            gpus=resources_dict.get("gpus", []),
            max_parallel=resources_dict.get("max_parallel"),
            allocation=ResourceAllocationType(
                resources_dict.get("allocation", "dynamic")
            ),
            remote=remote_configs,
            mode=ExecutionMode(resources_dict.get("mode", "local")),
            cost_per_gpu_hour=resources_dict.get("cost_per_gpu_hour"),
        )

        # Parse jobs
        jobs = []
        for job_dict in conf.get("jobs", []):
            jobs.append(
                JobConfig(
                    name=job_dict["name"],
                    verb=job_dict["verb"],
                    config=job_dict["config"],
                    depends_on=job_dict.get("depends_on", []),
                    resources=job_dict.get("resources", {}),
                    args=job_dict.get("args", []),
                    env=job_dict.get("env", {}),
                    workdir=job_dict.get("workdir"),
                    max_retries=job_dict.get("max_retries", 0),
                    retry_delay=job_dict.get("retry_delay", 60.0),
                    timeout=job_dict.get("timeout"),
                )
            )

        return cls(
            name=conf["name"],
            description=conf.get("description"),
            resources=resources,
            jobs=jobs,
            env=conf.get("env", {}),
            output_dir=conf.get("output_dir"),
            tui=conf.get("tui", True),
            timeout=conf.get("timeout"),
        )

    def validate(self) -> list[str]:
        """Validate workflow configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.name:
            errors.append("Workflow name is required")

        if not self.jobs:
            errors.append("At least one job is required")

        # Validate job names are unique
        job_names = [job.name for job in self.jobs]
        if len(job_names) != len(set(job_names)):
            errors.append("Job names must be unique")

        # Validate dependencies
        for job in self.jobs:
            for dep in job.depends_on:
                if dep not in job_names:
                    errors.append(f"Job '{job.name}' depends on unknown job '{dep}'")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Workflow has circular dependencies")

        # Validate verbs
        valid_verbs = {
            "train",
            "tune",
            "evaluate",
            "eval",
            "infer",
            "quantize",
            "synth",
            "synthesize",
            "judge",
        }
        for job in self.jobs:
            if job.verb not in valid_verbs:
                errors.append(
                    f"Job '{job.name}' has invalid verb '{job.verb}'. "
                    f"Valid verbs: {valid_verbs}"
                )

        # Validate config files exist
        for job in self.jobs:
            # Skip special URIs like oumi:// or http://
            if "://" in job.config:
                continue
            config_path = Path(job.config)
            if not config_path.exists():
                errors.append(f"Job '{job.name}': Config file not found: {job.config}")

        return errors

    def _has_circular_dependencies(self) -> bool:
        """Check if workflow has circular dependencies using DFS.

        Returns:
            True if circular dependencies exist
        """
        # Build adjacency list
        graph = {job.name: job.depends_on for job in self.jobs}

        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for job in self.jobs:
            if job.name not in visited:
                if has_cycle(job.name):
                    return True

        return False
