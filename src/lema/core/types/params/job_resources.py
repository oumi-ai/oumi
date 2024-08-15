from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


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
