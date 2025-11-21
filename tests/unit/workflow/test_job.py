"""Tests for Job class."""

import pytest

from oumi.workflow.config import JobConfig
from oumi.workflow.job import Job, JobStatus


class TestJob:
    """Tests for Job."""

    def test_job_creation(self):
        """Test creating a job."""
        config = JobConfig(
            name="test-job",
            verb="train",
            config="train.yaml",
        )
        job = Job(config)

        assert job.id == "test-job"
        assert job.status == JobStatus.PENDING
        assert job.is_ready
        assert not job.is_running
        assert not job.is_complete

    def test_job_with_dependencies(self):
        """Test job with dependencies."""
        config = JobConfig(
            name="test-job",
            verb="evaluate",
            config="eval.yaml",
            depends_on=["train-job"],
        )
        job = Job(config)

        assert config.depends_on == ["train-job"]

    def test_build_command(self):
        """Test building command."""
        config = JobConfig(
            name="test-job",
            verb="train",
            config="configs/train.yaml",
            args=["--foo=bar", "--baz=qux"],
        )
        job = Job(config)

        cmd = job._build_command()

        assert "oumi" in cmd
        assert "train" in cmd
        assert "--config" in cmd
        assert "configs/train.yaml" in cmd
        assert "--foo=bar" in cmd
        assert "--baz=qux" in cmd

    def test_parse_progress_step(self):
        """Test parsing step progress from log line."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        job._parse_progress_from_line("Step 100/1000")

        assert job.metrics.current_step == 100
        assert job.metrics.total_steps == 1000
        assert job.metrics.progress_percent == 10.0

    def test_parse_progress_loss(self):
        """Test parsing loss from log line."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        job._parse_progress_from_line("Loss: 0.1234")

        assert job.metrics.loss == 0.1234

    def test_parse_progress_learning_rate(self):
        """Test parsing learning rate from log line."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        job._parse_progress_from_line("lr: 1e-4")

        assert job.metrics.learning_rate == 1e-4

    def test_can_retry(self):
        """Test retry logic."""
        config = JobConfig(
            name="test",
            verb="train",
            config="train.yaml",
            max_retries=3,
        )
        job = Job(config)

        # Initially can't retry (not failed)
        assert not job.can_retry()

        # After failure, can retry
        job.status = JobStatus.FAILED
        assert job.can_retry()

        # After max retries, can't retry
        job._retry_count = 3
        assert not job.can_retry()

    def test_reset(self):
        """Test resetting job for retry."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        # Simulate job execution
        job.status = JobStatus.FAILED
        job.metrics.loss = 0.5

        # Reset
        job.reset()

        assert job.status == JobStatus.PENDING
        assert job.metrics.loss is None
        assert job._retry_count == 1

    @pytest.mark.asyncio
    async def test_run_invalid_state(self):
        """Test running job in invalid state."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        job.status = JobStatus.RUNNING

        with pytest.raises(RuntimeError):
            await job.run()

    def test_job_repr(self):
        """Test job string representation."""
        config = JobConfig(name="test-job", verb="train", config="train.yaml")
        job = Job(config)

        repr_str = repr(job)

        assert "Job" in repr_str
        assert "test-job" in repr_str
        assert "train" in repr_str
        assert "pending" in repr_str.lower()

    def test_status_transitions(self):
        """Test valid status transitions."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        # PENDING -> QUEUED
        assert job.status == JobStatus.PENDING
        job.status = JobStatus.QUEUED
        assert job.is_ready

        # QUEUED -> RUNNING
        job.status = JobStatus.RUNNING
        assert job.is_running
        assert not job.is_complete

        # RUNNING -> COMPLETED
        job.status = JobStatus.COMPLETED
        assert not job.is_running
        assert job.is_complete
        assert job.is_successful

    def test_failed_status(self):
        """Test failed status."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        job.status = JobStatus.FAILED

        assert job.is_complete
        assert not job.is_successful

    def test_cancelled_status(self):
        """Test cancelled status."""
        config = JobConfig(name="test", verb="train", config="train.yaml")
        job = Job(config)

        job.status = JobStatus.CANCELLED

        assert job.is_complete
        assert not job.is_successful

    def test_gpu_assignment(self):
        """Test GPU assignment."""
        config = JobConfig(
            name="test",
            verb="train",
            config="train.yaml",
            resources={"gpu": 2},
        )
        job = Job(config)

        job.assigned_gpu = 2

        assert job.assigned_gpu == 2
        assert job.metrics.gpu_id is None  # Set during execution

    def test_remote_assignment(self):
        """Test remote resource assignment."""
        config = JobConfig(
            name="test",
            verb="train",
            config="train.yaml",
            resources={"remote": "aws-cluster"},
        )
        job = Job(config)

        job.assigned_remote = "aws-cluster"

        assert job.assigned_remote == "aws-cluster"

    def test_timeout_config(self):
        """Test job with timeout."""
        config = JobConfig(
            name="test",
            verb="train",
            config="train.yaml",
            timeout=3600,  # 1 hour
        )
        job = Job(config)

        assert job.config.timeout == 3600

    def test_environment_variables(self):
        """Test job environment variables."""
        config = JobConfig(
            name="test",
            verb="train",
            config="train.yaml",
            env={"CUDA_VISIBLE_DEVICES": "0,1", "WANDB_PROJECT": "test"},
        )
        job = Job(config)

        assert config.env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert config.env["WANDB_PROJECT"] == "test"
