"""Resource management for workflow execution."""

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of resources."""

    GPU = "gpu"
    CPU = "cpu"
    REMOTE = "remote"


@dataclass
class Resource:
    """Represents a computational resource."""

    id: str
    type: ResourceType
    gpu_index: Optional[int] = None
    remote_name: Optional[str] = None
    in_use: bool = False
    current_job: Optional[str] = None

    # Resource metrics
    memory_total: Optional[int] = None  # MB
    memory_used: Optional[int] = None  # MB
    utilization: Optional[float] = None  # 0-100%


class ResourceManager:
    """Manages allocation of computational resources (GPUs, remote machines)."""

    def __init__(
        self,
        local_gpus: Optional[list[int]] = None,
        max_parallel: Optional[int] = None,
        discover_gpus: bool = True,
    ):
        """Initialize resource manager.

        Args:
            local_gpus: List of GPU indices to use (None = discover all)
            max_parallel: Maximum parallel jobs (None = unlimited)
            discover_gpus: Whether to auto-discover available GPUs
        """
        self.max_parallel = max_parallel
        self._resources: dict[str, Resource] = {}
        self._lock = asyncio.Lock()
        self._resource_available = asyncio.Condition(self._lock)

        # Discover and register local GPUs
        if discover_gpus:
            discovered_gpus = self._discover_gpus()
            if local_gpus is not None:
                # Filter to requested GPUs
                discovered_gpus = [g for g in discovered_gpus if g in local_gpus]
            self._register_gpus(discovered_gpus)
        elif local_gpus:
            self._register_gpus(local_gpus)

        logger.info(
            f"ResourceManager initialized with {len(self._resources)} resources"
        )

    def _discover_gpus(self) -> list[int]:
        """Discover available GPUs using nvidia-smi.

        Returns:
            List of GPU indices
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                gpu_indices = [
                    int(line.strip())
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                logger.info(f"Discovered {len(gpu_indices)} GPUs: {gpu_indices}")
                return gpu_indices
            else:
                logger.warning("nvidia-smi failed, no GPUs discovered")
                return []

        except FileNotFoundError:
            logger.warning("nvidia-smi not found, no GPUs discovered")
            return []
        except Exception as e:
            logger.warning(f"Error discovering GPUs: {e}")
            return []

    def _register_gpus(self, gpu_indices: list[int]) -> None:
        """Register GPU resources.

        Args:
            gpu_indices: List of GPU indices to register
        """
        for idx in gpu_indices:
            resource = Resource(
                id=f"gpu:{idx}",
                type=ResourceType.GPU,
                gpu_index=idx,
            )
            self._resources[resource.id] = resource
            logger.debug(f"Registered resource: {resource.id}")

    def register_remote(self, name: str, max_jobs: int = 1) -> None:
        """Register a remote execution resource.

        Args:
            name: Name of remote resource (e.g., cluster name)
            max_jobs: Maximum concurrent jobs on this resource
        """
        for i in range(max_jobs):
            resource = Resource(
                id=f"remote:{name}:{i}",
                type=ResourceType.REMOTE,
                remote_name=name,
            )
            self._resources[resource.id] = resource
            logger.debug(f"Registered remote resource: {resource.id}")

    @property
    def available_gpus(self) -> list[int]:
        """Get list of available (not in use) GPU indices."""
        return [
            r.gpu_index
            for r in self._resources.values()
            if r.type == ResourceType.GPU and not r.in_use and r.gpu_index is not None
        ]

    @property
    def total_gpus(self) -> int:
        """Get total number of GPU resources."""
        return sum(1 for r in self._resources.values() if r.type == ResourceType.GPU)

    @property
    def available_count(self) -> int:
        """Get count of available resources."""
        return sum(1 for r in self._resources.values() if not r.in_use)

    @property
    def in_use_count(self) -> int:
        """Get count of resources currently in use."""
        return sum(1 for r in self._resources.values() if r.in_use)

    async def acquire(
        self,
        requirements: dict[str, any],
        job_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Resource]:
        """Acquire a resource matching requirements.

        Args:
            requirements: Resource requirements dict
                Examples:
                - {"gpu": 0} - specific GPU
                - {"gpu": "auto"} - any available GPU
                - {"remote": "aws-cluster"} - specific remote
            job_id: ID of job requesting resource
            timeout: Maximum time to wait for resource (None = wait forever)

        Returns:
            Acquired resource, or None if timeout

        Raises:
            ValueError: If requirements are invalid
        """

        async def _try_acquire() -> Optional[Resource]:
            """Try to acquire a resource, return None if not available."""
            # Check if we've hit max parallel limit
            if self.max_parallel and self.in_use_count >= self.max_parallel:
                return None

            # Try to find matching resource
            resource = self._find_matching_resource(requirements)
            if resource:
                resource.in_use = True
                resource.current_job = job_id
                logger.info(f"Acquired resource {resource.id} for job {job_id}")
                return resource
            return None

        # Wait on condition variable until resource available or timeout
        async with self._resource_available:
            # First try without waiting
            resource = await _try_acquire()
            if resource:
                return resource

            # Wait for resource to become available
            while True:
                try:
                    if timeout:
                        await asyncio.wait_for(
                            self._resource_available.wait(), timeout=timeout
                        )
                    else:
                        await self._resource_available.wait()
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout acquiring resource for job {job_id}")
                    return None

                # Try to acquire after being notified
                resource = await _try_acquire()
                if resource:
                    return resource
                # If still no resource available, wait again

    def _find_matching_resource(
        self, requirements: dict[str, any]
    ) -> Optional[Resource]:
        """Find a resource matching requirements.

        Args:
            requirements: Resource requirements dict

        Returns:
            Matching available resource, or None
        """
        # GPU requirement
        if "gpu" in requirements:
            gpu_req = requirements["gpu"]

            # Specific GPU requested
            if isinstance(gpu_req, int):
                resource_id = f"gpu:{gpu_req}"
                if (
                    resource_id in self._resources
                    and not self._resources[resource_id].in_use
                ):
                    return self._resources[resource_id]
                return None

            # Auto-assign any GPU
            elif gpu_req == "auto":
                for resource in self._resources.values():
                    if resource.type == ResourceType.GPU and not resource.in_use:
                        return resource
                return None

        # Remote requirement
        if "remote" in requirements:
            remote_name = requirements["remote"]
            for resource in self._resources.values():
                if (
                    resource.type == ResourceType.REMOTE
                    and resource.remote_name == remote_name
                    and not resource.in_use
                ):
                    return resource
            return None

        # No specific requirements - return any available resource
        for resource in self._resources.values():
            if not resource.in_use:
                return resource

        return None

    async def release(self, resource: Resource, job_id: str) -> None:
        """Release a resource.

        Args:
            resource: Resource to release
            job_id: ID of job releasing resource
        """
        async with self._resource_available:
            if resource.id not in self._resources:
                logger.warning(f"Attempting to release unknown resource: {resource.id}")
                return

            if resource.current_job != job_id:
                logger.warning(
                    f"Job {job_id} attempting to release resource {resource.id} "
                    f"owned by {resource.current_job}"
                )
                return

            resource.in_use = False
            resource.current_job = None
            logger.info(f"Released resource {resource.id} from job {job_id}")

            # Notify waiting jobs that a resource is available
            self._resource_available.notify()

    def get_resource_by_id(self, resource_id: str) -> Optional[Resource]:
        """Get resource by ID.

        Args:
            resource_id: Resource ID

        Returns:
            Resource or None if not found
        """
        return self._resources.get(resource_id)

    def get_all_resources(self) -> list[Resource]:
        """Get all registered resources.

        Returns:
            List of all resources
        """
        return list(self._resources.values())

    async def update_gpu_metrics(self) -> None:
        """Update GPU metrics for all GPU resources."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue

                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpu_idx = int(parts[0])
                        memory_used = int(parts[1])
                        memory_total = int(parts[2])
                        utilization = float(parts[3])

                        resource_id = f"gpu:{gpu_idx}"
                        if resource_id in self._resources:
                            self._resources[resource_id].memory_used = memory_used
                            self._resources[resource_id].memory_total = memory_total
                            self._resources[resource_id].utilization = utilization

        except Exception as e:
            logger.debug(f"Error updating GPU metrics: {e}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResourceManager(total={len(self._resources)}, "
            f"available={self.available_count}, "
            f"in_use={self.in_use_count})"
        )
