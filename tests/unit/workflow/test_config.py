"""Tests for workflow configuration."""

import tempfile
from pathlib import Path

import pytest

from oumi.workflow.config import (
    ExecutionMode,
    JobConfig,
    ResourceAllocationType,
    ResourceConfig,
    WorkflowConfig,
)


class TestWorkflowConfig:
    """Tests for WorkflowConfig."""

    def test_basic_config_creation(self):
        """Test creating a basic workflow config."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(
                    name="job1",
                    verb="train",
                    config="train.yaml",
                ),
            ],
        )

        assert config.name == "test-workflow"
        assert len(config.jobs) == 1
        assert config.jobs[0].name == "job1"

    def test_config_with_dependencies(self):
        """Test config with job dependencies."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(name="job1", verb="train", config="train.yaml"),
                JobConfig(
                    name="job2",
                    verb="evaluate",
                    config="eval.yaml",
                    depends_on=["job1"],
                ),
            ],
        )

        assert len(config.jobs) == 2
        assert config.jobs[1].depends_on == ["job1"]

    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(name="job1", verb="train", config="train.yaml"),
                JobConfig(
                    name="job2",
                    verb="evaluate",
                    config="eval.yaml",
                    depends_on=["job1"],
                ),
            ],
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_validate_missing_name(self):
        """Test validation catches missing name."""
        config = WorkflowConfig(
            name="",
            jobs=[JobConfig(name="job1", verb="train", config="train.yaml")],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("name is required" in e for e in errors)

    def test_validate_no_jobs(self):
        """Test validation catches missing jobs."""
        config = WorkflowConfig(name="test-workflow", jobs=[])

        errors = config.validate()
        assert len(errors) > 0
        assert any("at least one job" in e.lower() for e in errors)

    def test_validate_duplicate_job_names(self):
        """Test validation catches duplicate job names."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(name="job1", verb="train", config="train.yaml"),
                JobConfig(name="job1", verb="evaluate", config="eval.yaml"),
            ],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("must be unique" in e for e in errors)

    def test_validate_unknown_dependency(self):
        """Test validation catches unknown dependencies."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(
                    name="job1",
                    verb="train",
                    config="train.yaml",
                    depends_on=["unknown-job"],
                ),
            ],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("unknown job" in e.lower() for e in errors)

    def test_validate_circular_dependency(self):
        """Test validation catches circular dependencies."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(
                    name="job1",
                    verb="train",
                    config="train.yaml",
                    depends_on=["job2"],
                ),
                JobConfig(
                    name="job2",
                    verb="evaluate",
                    config="eval.yaml",
                    depends_on=["job1"],
                ),
            ],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("circular" in e.lower() for e in errors)

    def test_validate_invalid_verb(self):
        """Test validation catches invalid verbs."""
        config = WorkflowConfig(
            name="test-workflow",
            jobs=[
                JobConfig(name="job1", verb="invalid-verb", config="train.yaml"),
            ],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("invalid verb" in e.lower() for e in errors)

    def test_from_yaml(self):
        """Test loading config from YAML."""
        yaml_content = """
name: "test-workflow"
description: "Test workflow"

resources:
  gpus: [0, 1]
  max_parallel: 2
  allocation: "dynamic"

jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    resources:
      gpu: 0

  - name: "eval"
    verb: "evaluate"
    config: "eval.yaml"
    depends_on: ["train"]
    resources:
      gpu: 1
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            config = WorkflowConfig.from_yaml(yaml_path)

            assert config.name == "test-workflow"
            assert config.description == "Test workflow"
            assert config.resources.gpus == [0, 1]
            assert config.resources.max_parallel == 2
            assert config.resources.allocation == ResourceAllocationType.DYNAMIC
            assert len(config.jobs) == 2
            assert config.jobs[0].name == "train"
            assert config.jobs[1].depends_on == ["train"]

        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            WorkflowConfig.from_yaml("nonexistent.yaml")

    def test_resource_config_defaults(self):
        """Test resource config defaults."""
        config = ResourceConfig()

        assert config.gpus == []
        assert config.max_parallel is None
        assert config.allocation == ResourceAllocationType.DYNAMIC
        assert config.mode == ExecutionMode.LOCAL

    def test_job_config_defaults(self):
        """Test job config defaults."""
        job = JobConfig(name="test", verb="train", config="train.yaml")

        assert job.depends_on == []
        assert job.resources == {}
        assert job.args == []
        assert job.env == {}
        assert job.max_retries == 0
        assert job.timeout is None
