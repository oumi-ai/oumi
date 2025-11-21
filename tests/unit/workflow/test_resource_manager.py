"""Tests for ResourceManager."""

import asyncio

import pytest

from oumi.workflow.resource_manager import ResourceManager, ResourceType


class TestResourceManager:
    """Tests for ResourceManager."""

    def test_init_with_gpus(self):
        """Test initialization with GPU list."""
        rm = ResourceManager(local_gpus=[0, 1, 2], discover_gpus=False)

        assert rm.total_gpus == 3
        assert rm.available_count == 3
        assert rm.in_use_count == 0

    def test_init_auto_discover(self):
        """Test GPU auto-discovery."""
        # This may discover real GPUs or none
        rm = ResourceManager(discover_gpus=True)

        # Should not crash, may have 0 or more GPUs
        assert rm.total_gpus >= 0
        assert rm.available_count >= 0

    def test_register_remote(self):
        """Test registering remote resources."""
        rm = ResourceManager(discover_gpus=False)
        rm.register_remote("aws-cluster", max_jobs=2)

        resources = rm.get_all_resources()
        remote_resources = [r for r in resources if r.type == ResourceType.REMOTE]

        assert len(remote_resources) == 2
        assert all(r.remote_name == "aws-cluster" for r in remote_resources)

    @pytest.mark.asyncio
    async def test_acquire_specific_gpu(self):
        """Test acquiring a specific GPU."""
        rm = ResourceManager(local_gpus=[0, 1], discover_gpus=False)

        resource = await rm.acquire({"gpu": 0}, "job1")

        assert resource is not None
        assert resource.gpu_index == 0
        assert resource.in_use
        assert resource.current_job == "job1"
        assert rm.in_use_count == 1

    @pytest.mark.asyncio
    async def test_acquire_auto_gpu(self):
        """Test auto-acquiring any available GPU."""
        rm = ResourceManager(local_gpus=[0, 1], discover_gpus=False)

        resource = await rm.acquire({"gpu": "auto"}, "job1")

        assert resource is not None
        assert resource.gpu_index in [0, 1]
        assert resource.in_use
        assert rm.in_use_count == 1

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test acquiring and releasing resources."""
        rm = ResourceManager(local_gpus=[0], discover_gpus=False)

        # Acquire
        resource = await rm.acquire({"gpu": 0}, "job1")
        assert resource is not None
        assert rm.in_use_count == 1

        # Release
        await rm.release(resource, "job1")
        assert not resource.in_use
        assert resource.current_job is None
        assert rm.in_use_count == 0

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquisition timeout when no resources available."""
        rm = ResourceManager(local_gpus=[0], discover_gpus=False)

        # Acquire the only GPU
        resource1 = await rm.acquire({"gpu": 0}, "job1")
        assert resource1 is not None

        # Try to acquire again with timeout
        resource2 = await rm.acquire({"gpu": 0}, "job2", timeout=0.5)
        assert resource2 is None

    @pytest.mark.asyncio
    async def test_max_parallel_limit(self):
        """Test max parallel jobs limit."""
        rm = ResourceManager(local_gpus=[0, 1], max_parallel=1, discover_gpus=False)

        # Acquire first resource
        resource1 = await rm.acquire({"gpu": "auto"}, "job1")
        assert resource1 is not None

        # Try to acquire second (should timeout due to max_parallel=1)
        resource2 = await rm.acquire({"gpu": "auto"}, "job2", timeout=0.5)
        assert resource2 is None

        # Release first
        await rm.release(resource1, "job1")

        # Now can acquire
        resource3 = await rm.acquire({"gpu": "auto"}, "job3", timeout=0.5)
        assert resource3 is not None

    @pytest.mark.asyncio
    async def test_parallel_acquire(self):
        """Test multiple jobs acquiring resources in parallel."""
        rm = ResourceManager(local_gpus=[0, 1, 2], discover_gpus=False)

        async def acquire_job(job_id):
            return await rm.acquire({"gpu": "auto"}, job_id)

        # Acquire 3 resources in parallel
        results = await asyncio.gather(
            acquire_job("job1"),
            acquire_job("job2"),
            acquire_job("job3"),
        )

        assert all(r is not None for r in results)
        assert len({r.gpu_index for r in results}) == 3  # Different GPUs
        assert rm.in_use_count == 3
        assert rm.available_count == 0

    @pytest.mark.asyncio
    async def test_acquire_remote(self):
        """Test acquiring remote resources."""
        rm = ResourceManager(discover_gpus=False)
        rm.register_remote("aws-cluster", max_jobs=2)

        resource = await rm.acquire({"remote": "aws-cluster"}, "job1")

        assert resource is not None
        assert resource.type == ResourceType.REMOTE
        assert resource.remote_name == "aws-cluster"
        assert resource.in_use

    def test_available_gpus(self):
        """Test getting available GPU indices."""
        rm = ResourceManager(local_gpus=[0, 1, 2], discover_gpus=False)

        available = rm.available_gpus
        assert available == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_available_gpus_after_acquire(self):
        """Test available GPUs after acquiring one."""
        rm = ResourceManager(local_gpus=[0, 1, 2], discover_gpus=False)

        await rm.acquire({"gpu": 1}, "job1")

        available = rm.available_gpus
        assert 1 not in available
        assert set(available) == {0, 2}

    def test_get_resource_by_id(self):
        """Test getting resource by ID."""
        rm = ResourceManager(local_gpus=[0, 1], discover_gpus=False)

        resource = rm.get_resource_by_id("gpu:0")
        assert resource is not None
        assert resource.gpu_index == 0

        resource = rm.get_resource_by_id("nonexistent")
        assert resource is None

    def test_repr(self):
        """Test string representation."""
        rm = ResourceManager(local_gpus=[0, 1], discover_gpus=False)
        repr_str = repr(rm)

        assert "ResourceManager" in repr_str
        assert "total=2" in repr_str
