"""Unit tests for ModalClient."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import JobState
from oumi.launcher.clients.modal_client import ModalClient


#
# Fixtures
#
@pytest.fixture
def mock_thread():
    with patch("oumi.launcher.clients.modal_client.Thread") as thread_mock:
        # Don't actually start the background thread
        thread_mock.return_value.start = Mock()
        yield thread_mock


@pytest.fixture
def mock_popen():
    with patch("oumi.launcher.clients.modal_client.Popen") as popen_mock:
        yield popen_mock


@pytest.fixture
def temp_working_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _get_default_job() -> JobConfig:
    """Create a default JobConfig for testing."""
    resources = JobResources(
        cloud="modal",
        region="us-central1",
        zone=None,
        accelerators="A100:4",
        cpus="4",
        memory="64GB",
        instance_type=None,
        use_spot=False,
        disk_size=512,
        disk_tier=None,
    )
    return JobConfig(
        name="test-modal-job",
        user="user",
        working_dir="./",
        num_nodes=1,
        resources=resources,
        envs={"VAR1": "val1", "VAR2": "val2"},
        file_mounts={},
        storage_mounts={},
        setup="pip install torch transformers",
        run="python train.py",
    )


#
# Tests for GPU conversion
#
def test_convert_accelerator_to_modal_gpu_basic(mock_thread, temp_working_dir):
    """Test basic GPU conversion."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_accelerator_to_modal_gpu("A100") == "A100"
    assert client._convert_accelerator_to_modal_gpu("H100") == "H100"
    assert client._convert_accelerator_to_modal_gpu("L4") == "L4"
    assert client._convert_accelerator_to_modal_gpu("T4") == "T4"


def test_convert_accelerator_to_modal_gpu_with_count(mock_thread, temp_working_dir):
    """Test GPU conversion with count."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_accelerator_to_modal_gpu("A100:4") == "A100:4"
    assert client._convert_accelerator_to_modal_gpu("H100:8") == "H100:8"
    assert client._convert_accelerator_to_modal_gpu("L4:2") == "L4:2"


def test_convert_accelerator_to_modal_gpu_variants(mock_thread, temp_working_dir):
    """Test GPU conversion with memory variants."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_accelerator_to_modal_gpu("A100-80GB") == "A100-80GB"
    assert client._convert_accelerator_to_modal_gpu("A100-40GB") == "A100-40GB"


def test_convert_accelerator_to_modal_gpu_unsupported(mock_thread, temp_working_dir):
    """Test GPU conversion for unsupported types."""
    client = ModalClient(working_dir=temp_working_dir)
    # V100 is not supported on Modal
    assert client._convert_accelerator_to_modal_gpu("V100") is None


def test_convert_accelerator_to_modal_gpu_none(mock_thread, temp_working_dir):
    """Test GPU conversion with None input."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_accelerator_to_modal_gpu(None) is None


def test_convert_accelerator_case_insensitive(mock_thread, temp_working_dir):
    """Test GPU conversion is case insensitive."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_accelerator_to_modal_gpu("a100") == "A100"
    assert client._convert_accelerator_to_modal_gpu("h100:4") == "H100:4"


#
# Tests for memory conversion
#
def test_convert_memory_to_modal_basic(mock_thread, temp_working_dir):
    """Test memory conversion."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_memory_to_modal("64GB") == 64 * 1024  # 65536 MB
    assert client._convert_memory_to_modal("128GB") == 128 * 1024
    assert client._convert_memory_to_modal("256GB") == 256 * 1024


def test_convert_memory_to_modal_with_modifier(mock_thread, temp_working_dir):
    """Test memory conversion with + modifier."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_memory_to_modal("64+") == 64 * 1024
    assert client._convert_memory_to_modal("128GB+") == 128 * 1024


def test_convert_memory_to_modal_none(mock_thread, temp_working_dir):
    """Test memory conversion with None."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._convert_memory_to_modal(None) is None


#
# Tests for Modal app code generation
#
def test_generate_modal_app_code_basic(mock_thread, temp_working_dir):
    """Test basic Modal app code generation."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()
    job_id = "test-job-123"

    code = client._generate_modal_app_code(job, job_id)

    # Check that the generated code contains expected elements
    assert "import modal" in code
    assert f'app = modal.App("oumi-{job_id}")' in code
    assert "def run_oumi_job():" in code
    assert 'gpu="A100:4"' in code
    assert "python train.py" in code
    assert "pip install torch transformers" in code


def test_generate_modal_app_code_no_gpu(mock_thread, temp_working_dir):
    """Test Modal app code generation without GPU."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()
    job.resources.accelerators = None
    job_id = "test-job-nogpu"

    code = client._generate_modal_app_code(job, job_id)

    # Check that no GPU is specified
    assert "gpu=" not in code


def test_generate_modal_app_code_with_envs(mock_thread, temp_working_dir):
    """Test Modal app code generation with environment variables."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()
    job_id = "test-job-envs"

    code = client._generate_modal_app_code(job, job_id)

    # Check that environment variables are included
    assert "VAR1" in code
    assert "val1" in code
    assert "modal.Secret.from_dict" in code


def test_generate_modal_app_code_no_setup(mock_thread, temp_working_dir):
    """Test Modal app code generation without setup script."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()
    job.setup = None
    job_id = "test-job-nosetup"

    code = client._generate_modal_app_code(job, job_id)

    # Should still generate valid code
    assert "import modal" in code
    assert "def run_oumi_job():" in code


#
# Tests for job submission
#
def test_submit_job_creates_status(mock_thread, mock_popen, temp_working_dir):
    """Test that submit_job creates proper initial status."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status = client.submit_job(job)

    assert status is not None
    assert status.name == job.name
    assert status.cluster == "modal"
    assert status.state == JobState.PENDING
    assert not status.done
    assert "modal-" in status.id


def test_submit_job_generates_app_file(mock_thread, mock_popen, temp_working_dir):
    """Test that submit_job generates an app file."""
    import os

    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status = client.submit_job(job)

    # Check that app file was created
    app_dir = os.path.join(temp_working_dir, ".oumi_modal", "apps", status.id)
    app_file = os.path.join(app_dir, "app.py")
    assert os.path.exists(app_file)

    # Verify content
    with open(app_file) as f:
        content = f.read()
    assert "import modal" in content
    assert "python train.py" in content


def test_submit_job_increments_id(mock_thread, mock_popen, temp_working_dir):
    """Test that job IDs are unique."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status1 = client.submit_job(job)
    status2 = client.submit_job(job)

    assert status1.id != status2.id


#
# Tests for job listing and retrieval
#
def test_list_jobs_empty(mock_thread, temp_working_dir):
    """Test list_jobs on empty client."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client.list_jobs() == []


def test_list_jobs_after_submit(mock_thread, mock_popen, temp_working_dir):
    """Test list_jobs after submitting jobs."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status1 = client.submit_job(job)
    status2 = client.submit_job(job)

    jobs = client.list_jobs()
    assert len(jobs) == 2
    job_ids = [j.id for j in jobs]
    assert status1.id in job_ids
    assert status2.id in job_ids


def test_get_job_existing(mock_thread, mock_popen, temp_working_dir):
    """Test get_job for existing job."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status = client.submit_job(job)
    retrieved = client.get_job(status.id)

    assert retrieved is not None
    assert retrieved.id == status.id
    assert retrieved.name == status.name


def test_get_job_nonexistent(mock_thread, temp_working_dir):
    """Test get_job for non-existent job."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client.get_job("nonexistent-job-id") is None


#
# Tests for job cancellation
#
def test_cancel_nonexistent_job(mock_thread, temp_working_dir):
    """Test cancelling a non-existent job."""
    client = ModalClient(working_dir=temp_working_dir)
    result = client.cancel("nonexistent-job-id")
    assert result is None


def test_cancel_job(mock_thread, mock_popen, temp_working_dir):
    """Test cancelling a job."""
    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess for submission
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status = client.submit_job(job)

    # Mock the subprocess for cancellation
    mock_popen.reset_mock()
    mock_cancel_process = Mock()
    mock_cancel_process.communicate.return_value = ("", "")
    mock_cancel_process.returncode = 0
    mock_popen.return_value = mock_cancel_process

    result = client.cancel(status.id)

    assert result is not None
    assert result.state == JobState.CANCELLED
    assert result.done is True


#
# Tests for log retrieval
#
def test_get_logs_nonexistent(mock_thread, temp_working_dir):
    """Test getting logs for non-existent job."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client.get_logs("nonexistent-job-id") is None


def test_get_logs_from_file(mock_thread, mock_popen, temp_working_dir):
    """Test getting logs from log file."""
    from pathlib import Path

    client = ModalClient(working_dir=temp_working_dir)
    job = _get_default_job()

    # Mock the subprocess
    mock_process = Mock()
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    status = client.submit_job(job)

    # Write some logs to the log file
    log_file = client._jobs[status.id].log_file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write("Test log output\nLine 2")

    logs = client.get_logs(status.id)
    assert logs is not None
    assert "Test log output" in logs


#
# Tests for shutdown
#
def test_shutdown(mock_thread, temp_working_dir):
    """Test client shutdown."""
    client = ModalClient(working_dir=temp_working_dir)
    assert client._running is True

    client.shutdown()
    assert client._running is False
