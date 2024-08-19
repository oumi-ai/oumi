from dataclasses import dataclass, field
from typing import Dict, Optional

from omegaconf import MISSING

from lema.core.types.base_config import BaseConfig


@dataclass
class StorageMount:
    """A storage system mount to attach to a node."""

    #: The remote path to mount the local path to (Required).
    #: e.g. 'gs://bucket/path' for GCS, 's3://bucket/path' for S3, or 'r2://path' for
    #: R2.
    source: str = MISSING

    #: The remote storage solution (Required). Must be one of 's3', 'gcs' or 'r2'.
    store: str = MISSING


@dataclass
class JobResources:
    """Resources required for a single node in a job."""

    #: The cloud used to run the job (required).
    cloud: str = MISSING

    #: The region to use (optional). Supported values vary by environment.
    region: Optional[str] = None

    #: The zone to use (optional). Supported values vary by environment.
    zone: Optional[str] = None

    #: Accelerator type (optional). Supported values vary by environment.
    #: For GCP you may specify the accelerator name and count, e.g. "V100:4".
    accelerators: Optional[str] = None

    #: Number of vCPUs to use per node (optional). Sky-based clouds support strings with
    #: modifiers, e.g. "2+" to indicate at least 2 vCPUs.
    cpus: Optional[str] = None

    #: Memory to allocate per node in GiB (optional). Sky-based clouds support strings
    #: with modifiers, e.g. "256+" to indicate at least 256 GB.
    memory: Optional[str] = None

    #: Instance type to use (optional). Supported values vary by environment.
    #: The instance type is automatically inferred if `accelerators` is specified.
    instance_type: Optional[str] = None

    #: Whether the cluster should use spot instances (optional).
    #: If unspecified, defaults to False (on-demand instances).
    use_spot: bool = False

    #: Disk size in GiB to allocate for OS (mounted at /). Ignored by Polaris. Optional.
    disk_size: Optional[int] = None

    #: Disk tier to use for OS (optional).
    #: For sky-based clouds this Could be one of 'low', 'medium', 'high' or 'best'
    #: (default: 'medium').
    disk_tier: Optional[str] = "medium"


@dataclass
class JobConfig(BaseConfig):
    """Configuration for launching jobs on a cluster."""

    #: Job name (optional). Only used for display purposes.
    name: Optional[str] = None

    #: The user that the job will run as (optional). Required only for Polaris.
    user: Optional[str] = None

    #: The local directory containing the scripts required to execute this job.
    #: This directory will be copied to the remote node before the job is executed.
    working_dir: str = MISSING

    #: The number of nodes to use for the job. Defaults to 1.
    num_nodes: int = 1

    #: The resources required for each node in the job.
    resources: JobResources = field(default_factory=JobResources)

    #: The environment variables to set on the node.
    envs: Dict[str, str] = field(default_factory=dict)

    #: File mounts to attach to the node.
    #: For mounting (copying) local directories, the key is the file path on the remote
    #: and the value is the local path.
    #: The keys of `file_mounts` cannot be shared with `storage_mounts`.
    file_mounts: Dict[str, str] = field(default_factory=dict)

    #: Storage system mounts to attach to the node.
    #: For mounting remote storage solutions, the key is the file path on the remote
    #: and the value is a StorageMount.
    #: The keys of `storage_mounts` cannot be shared with `file_mounts`.
    storage_mounts: Dict[str, StorageMount] = field(default_factory=dict)

    #: The setup script to run on every node. Optional.
    #: `setup` will always be executed before `run`.
    #: ex) pip install -r requirements.txt
    setup: Optional[str] = None

    #: The script to run on every node. Required. Runs after `setup`.
    run: str = MISSING
